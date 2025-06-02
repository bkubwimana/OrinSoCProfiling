import argparse
import atexit
import sys
import os
from tqdm import tqdm
import time
import re
import pandas as pd
import csv
from datasets import load_dataset, Dataset,get_dataset_config_names
import torch
import subprocess
import signal
sys.path.append(os.path.join(os.getcwd(), "src"))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import transformers
import nvtx

NVTX_RANGE_NAME = "TimeCapture"  
HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


COLOR_RESET = "\033[0m"
COLOR_INFO = "\033[94m"
COLOR_DEBUG = "\033[93m"
COLOR_RESULT = "\033[96m"
COLOR_ERROR = "\033[91m"
MAX_TOKENS = 4096

# Global variables
telemetry_proc: subprocess.Popen = None
output_dir = "./outputs/tbudget/"

prompt_token_budget_list= []

with open("token_budget.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            prompt_token_budget_list.append(int(line))


def extract_predicted_choice(decoded_output: str) -> str:
    """
    Extracts a predicted multiple-choice answer (A, B, C, or D) from a string,
    with further refinement to avoid false positives when options are discussed.
    """
    text = decoded_output.strip()

    answer_pattern = re.compile(
        r"""
        \b(?i:Answer|The\s+(?:final\s+|correct\s+)?(?:answer|choice)\s+is|My\s+choice\s+is|Select(?:ed)?\s+(?:choice|option)\s*is)
        \s*[:\-=\s]*
        (?: 
            (?i:Option|Choice)\s+
        )?
        \s*[\[\(]?   
        ([A-D])      
        [\]\)\.,:]?  
        (?=\s|$|[^\w])
        """,
        re.VERBOSE | re.IGNORECASE
    )
    matches = answer_pattern.findall(text)
    if matches:
        return matches[-1].upper()

    markdown_answer_pattern = re.compile(
        r"""
        \*\*(?:Answer|Selected)
        \s*[:\-=\s]*
        (?:(?i:Option|Choice)\s+)? 
        ([A-D])                   
        \*\*
        """,
        re.VERBOSE | re.IGNORECASE
    )
    matches = markdown_answer_pattern.findall(text)
    if matches:
        return matches[-1].upper()

    markdown_option_pattern = re.compile(
        r"""
        \*\*(?:Option|Choice)\s* 
        (?:[:\-=\s]+\s*)?        
        (?P<choice_letter>[A-D]) 
        \*\*
        (?!\s+(?:is|are|was|were|details|describes|concerns|relates\s+to|refers\s+to|means|entails|states|suggests|provides|offers|talks\sabout|covers|deals\swith|seems|appears|might\sbe|could\sbe|would\sbe|has|contains|involves|represents)\b)
        """,
        re.VERBOSE | re.IGNORECASE
    )
    matches = markdown_option_pattern.finditer(text)
    found_options = [match.group('choice_letter') for match in matches]
    if found_options:
        return found_options[-1].upper()

    explicit_option_pattern = re.compile(
        r"""
        \b(?i:Option|Choice)\s* 
        (?:[:\-=\s])+\s* 
        \s*[\[\(]?
        ([A-D])                 
        [\]\)\.,:]?
        (?=\s|$|[^\w])          
        (?!\s+(?:is|are|was|were|details|describes|concerns|relates\s+to|refers\s+to|means|entails|states|suggests|provides|offers|talks\sabout|covers|deals\swith|seems|appears|might\sbe|could\sbe|would\sbe|has|contains|involves|represents)\b)
        """,
        re.VERBOSE | re.IGNORECASE
    )
    matches = explicit_option_pattern.findall(text)
    if matches:
        return matches[-1].upper()

    if len(text) <= 5:
        isolated_choice_match = re.fullmatch(
            r"\s*[\[\(\{]?([A-D])[\]\)\}\.,:]?\s*",
            text
        )
        if isolated_choice_match:
            return isolated_choice_match.group(1).upper()
    if text in ['A', 'B', 'C', 'D']:
        return text

    end_choice_pattern = re.compile(
        r"""
        (?:
            (?i:is|was|be|denotes|represents|therefore|hence|thus|so|the\sresult\sis|the\sanswer\swould\sbe)
            \s*[:\-=\s]*
        )?
        \s*[\[\(\{]?
        ([A-D])
        [\]\)\}\.,:]?
        \W*$
        """,
        re.VERBOSE | re.IGNORECASE
    )
    match = end_choice_pattern.search(text)
    if match:
        full_match_start_index = match.start(0)
        if full_match_start_index == 0 or not text[full_match_start_index - 1].isalpha():
            return match.group(1).upper()

    return "Invalid"


def predict_local(tokenizer, model, messages, device, tokens=MAX_TOKENS):
    tokenizer.chat_template = open("chat_deepseek.jinja").read()
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # ------------------------------------------------------------------ PREFILL
    with nvtx.annotate("prefill"):
        torch.cuda.synchronize()                  
        prefill_start = time.perf_counter()

        with torch.inference_mode():
            _ = model(**inputs, use_cache=True)   

        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - prefill_start) * 1e3

    # ------------------------------------------------------------------ DECODE
    with nvtx.annotate("decode"):
        torch.cuda.synchronize()
        decode_start = time.perf_counter()

        outputs = model.generate(
            **inputs,               
            max_new_tokens=tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        torch.cuda.synchronize()
        gen_time_ms = (time.perf_counter() - decode_start) * 1e3
        decode_ms = max(gen_time_ms - prefill_ms, 0.0)                     

    # ------------------------------------------------------------------ post-proc
    generated_token_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded_output     = tokenizer.decode(generated_token_ids,
                                          skip_special_tokens=True).strip()
    output_tokens      = generated_token_ids.size(0)
    predicted_choice   = extract_predicted_choice(decoded_output)

    return predicted_choice, decoded_output, (prefill_ms, decode_ms), output_tokens


def terminate_telemetry_process():
    """Function to directly kill the nvidia-smi process using its saved PID."""
    global telemetry_proc
    
    if telemetry_proc and telemetry_proc.poll() is None:
        try:
            telemetry_proc.terminate()
            telemetry_proc.wait(timeout=2)
        except:
            pass
    
    pid_file = os.path.join(output_dir, "nvidia-smi.pid")
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            print(f"{COLOR_INFO}Killing nvidia-smi process with PID {pid}{COLOR_RESET}")
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            subprocess.run(["sudo", "kill", "-9", str(pid)], check=False)
        except Exception as e:
            print(f"{COLOR_ERROR}Error killing nvidia-smi: {e}{COLOR_RESET}")
    
    try:
        subprocess.run(["sudo", "pkill", "-9", "nvidia-smi"], check=False)
        subprocess.run(["sudo", "kill", "-9", "nvidia-smi"], check=False)
    except:
        pass
        
    telemetry_proc = None

def start_telemetry_process():
    """Invoke tegrastats."""
    telemetry_script = os.path.join(os.path.dirname(__file__), "telemetry.sh")
    telemetry_proc = subprocess.Popen(["bash", telemetry_script], preexec_fn=os.setsid)
    atexit.register(terminate_telemetry_process)

def main(args):
    output_dir = "./outputs/profile/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.all:

        subset_list = get_dataset_config_names("edinburgh-dawg/mmlu-redux")
    else:
        subset_list = [args.subset_name]

    print(f"{COLOR_INFO}Loading tokenizer: {args.model_name_or_path}{COLOR_RESET}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    print(f"{COLOR_INFO}Loading model: {args.model_name_or_path}{COLOR_RESET}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        use_flash_attention_2=True
    )
    device = next(model.parameters()).device
    print(f"{COLOR_INFO}Model {args.model_name_or_path} loaded on device map: {model.hf_device_map}{COLOR_RESET}")
    

    # start telemetry in its own group
    
    #============================================#
    
    print(f"{COLOR_INFO}Starting telemetry process...{COLOR_RESET}")

    global telemetry_proc
    telemetry_proc = start_telemetry_process()
        
    #============================================#
    
    # Start NVTX range for all subsets
    print("Pushing NVTX range: TimeCapture")    
    
    nvtx.push_range(NVTX_RANGE_NAME)
    torch.cuda.synchronize()
    try:
        #go through each subset and evaluate
        for subset in subset_list:
            current_subset_name = subset
            
            safe_model_name = args.model_name_or_path.replace("/", "_")
            log_file = os.path.join(output_dir, f"local_log_{safe_model_name}_{current_subset_name}_{args.num_questions}.txt")
            results_file = os.path.join(output_dir, f"results_{safe_model_name}_{current_subset_name}_{args.num_questions}.csv")

            csv_file = open(results_file, 'w', newline='', encoding='utf-8')
            fieldnames = [
                "subset", "question", "choices", "ground_truth_index",
                "predicted_choice_letter", "predicted_index",
                "full_output", "inference_time", "output_tokens",
                "prefill", "decode"
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()


            print(f"{COLOR_INFO}Subset: {current_subset_name}, Questions: {args.num_questions}{COLOR_RESET}")
            print(f"{COLOR_INFO}Log file: {log_file}{COLOR_RESET}")
            print(f"{COLOR_INFO}Results file: {results_file}{COLOR_RESET}\n")
            print("" + "=" * 80)

            with open(log_file, "w") as f:
                f.write(f"Model Name/Path: {args.model_name_or_path}\n")
                f.write(f"Subset: {current_subset_name}\n")
                f.write(f"Number of Questions: {args.num_questions}\n")
                f.write(f"Config: {args.config}\n")
                f.write(f"Max Tokens: {MAX_TOKENS}\n")
                f.flush()

            logf = open(log_file, "a")

            print(f"{COLOR_INFO}Loading dataset: edinburgh-dawg/mmlu-redux, subset: {current_subset_name}{COLOR_RESET}")
            try:
                full_subset_dataset = load_dataset(
                    "edinburgh-dawg/mmlu-redux",
                    name=current_subset_name,
                    split="test",
                    token=HF_READ_TOKEN
                )
            except ValueError as e:
                print(f"{COLOR_DEBUG}Error loading dataset subset '{current_subset_name}': {e}{COLOR_RESET}")
                continue
            except Exception as e:
                print(f"{COLOR_DEBUG}An unexpected error occurred during dataset loading: {e}{COLOR_RESET}")
                continue

            if len(full_subset_dataset) < args.num_questions:
                print(f"{COLOR_DEBUG}Warning: Subset '{current_subset_name}' only has {len(full_subset_dataset)} questions. Evaluating on all available.{COLOR_RESET}")
                eval_dataset = full_subset_dataset
            else:
                eval_dataset = Dataset.from_dict(full_subset_dataset[:args.num_questions])

            print(f"{COLOR_INFO}Evaluating on {len(eval_dataset)} questions.{COLOR_RESET}")

            total_inference_time = 0
            correct_count_debug = 0
            total_output_tokens = 0

            choice_labels = ['A', 'B', 'C', 'D']
            label_to_index = {label: i for i, label in enumerate(choice_labels)}

            
            for idx, item in enumerate(tqdm(eval_dataset, desc="Evaluating Questions")):
                with nvtx.annotate(f"question_{idx}"):
                    choices_str = "\n".join(
                        [f"{choice_labels[i]}. {choice_text}" for i, choice_text in enumerate(item["choices"])]
                    )
                    prompt_content = (
                        f"Choose the single best answer (A, B, C, or D) for the following question:\n\n"
                        f"Question: {item['question']}\n\n"
                        f"Choices:\n{choices_str}\n\n"
                        "Concisely, provide only the letter of the correct answer in the format:\n"
                        "Answer: <A/B/C/D>\n"
                    )

                    messages = [
                        {"role": "user", "content": prompt_content},
                    ]

                    predicted_choice_letter, full_decoded_output, latency, output_tokens = predict_local(tokenizer, model, messages, device, tokens=MAX_TOKENS)
                    inference_time = (latency[0] + latency[1]) / 1000
                    total_inference_time += inference_time
                    total_output_tokens += output_tokens

                    logf.write(f"\n--- Item {idx} ---\n")
                    logf.write("Prompt:\n" + prompt_content + "\n")
                    logf.write("Full model output:\n" + full_decoded_output + "\n")
                    logf.write(f"Output Tokens: {output_tokens}\n")
                    logf.write(f"Prefill Time: {latency[0]:.2f} ms, Decode Time: {latency[1]:.2f} ms\n")
                    logf.flush()

                    ground_truth_index = item["answer"]
                    predicted_index = label_to_index.get(predicted_choice_letter, -1)

                    is_correct = (ground_truth_index == predicted_index)
                    if is_correct:
                        correct_count_debug += 1

                    csv_writer.writerow({
                        "subset": current_subset_name,
                        "question": item["question"],
                        "choices": item["choices"],
                        "ground_truth_index": ground_truth_index,
                        "predicted_choice_letter": predicted_choice_letter,
                        "predicted_index": predicted_index,
                        "full_output": full_decoded_output,
                        "prefill": latency[0],
                        "decode": latency[1],
                        "inference_time": inference_time,
                        "output_tokens": output_tokens
                    })
                    csv_file.flush()

            logf.close()
            csv_file.close()

            print(f"\n{COLOR_INFO}Debug Correct Count: {correct_count_debug}/{len(eval_dataset)}{COLOR_RESET}")

            total_predictions = len(eval_dataset)
            correct_predictions = correct_count_debug

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            avg_inference_time = total_inference_time / total_predictions if total_predictions > 0 else 0

            metrics = {"accuracy": accuracy, "average_inference_time_s": avg_inference_time}

            print(f"\n{COLOR_RESULT}Evaluation Metrics:{COLOR_RESET}")
            print(f"{COLOR_RESULT}  Accuracy: {accuracy:.4f}{COLOR_RESET}")
            print(f"{COLOR_RESULT}  Avg Inference Time: {avg_inference_time:.4f} s/question{COLOR_RESET}")
            print(f"{COLOR_RESULT}  Total Tokens Generated: {total_output_tokens}{COLOR_RESET}")
            print(f"{COLOR_RESULT}  Total Inference Time: {total_inference_time:.4f} s{COLOR_RESET}")

            with open(log_file, "a") as f:
                f.write("\nEvaluation Summary:\n")
                f.write(f"Total Questions Evaluated: {total_predictions}\n")
                f.write(f"Correct Predictions: {correct_predictions}\n")
                f.write(f"Total Tokens Generated: {total_output_tokens}\n")
                f.write(f"Metrics: {metrics}\n")

            print(f"{COLOR_INFO}Detailed results saved to: {results_file}{COLOR_RESET}")
    
    finally:
        torch.cuda.synchronize()
        nvtx.pop_range()
        
        telemetry_proc.terminate()
        terminate_telemetry_process()
        
        print("Popped NVTX range: TimeCapture")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM on MMLU-Redux subset locally.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="Hugging Face model name or local path."
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default="electrical_engineering",
        help="Name of the MMLU-Redux subset to evaluate (e.g., 'electrical_engineering', 'high_school_mathematics')."
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=5,
        help="Number of questions to evaluate from the subset."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="reasoning",
        help="Optional configuration name for logging."
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all MMLU-Redux subsets")
    args = parser.parse_args()
    main(args)

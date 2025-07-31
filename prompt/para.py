import argparse
import atexit
import sys
import os
from tqdm import tqdm
import time
import re
import pandas as pd
import csv
from datasets import load_dataset, Dataset, get_dataset_config_names
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
        (?:[:\-=\s])+
        \s*[\[\(]?([A-D])
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
        (?:(?i:is|was|be|denotes|represents|therefore|hence|thus|so|the\sresult\sis|the\sanswer\swould\sbe)\s*[:\-=\s]*)?
        \s*[\[\(\{]?([A-D])[\]\)\}\.,:]?\W*$
        """,
        re.VERBOSE | re.IGNORECASE
    )
    match = end_choice_pattern.search(text)
    if match:
        idx = match.start(1)
        if idx == 0 or not text[idx - 1].isalpha():
            return match.group(1).upper()

    return "Invalid"


def predict_local(tokenizer, model, inputs, tokens=MAX_TOKENS):
    # inputs is a dict of batched tensors already on CUDA
    # PREFILL
    with nvtx.annotate("prefill"):
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        with torch.inference_mode():
            _ = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - prefill_start) * 1e3

    # DECODE
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

    return outputs, (prefill_ms, decode_ms)


def terminate_telemetry_process():
    """Kill telemetry process (nvidia-smi) if running."""
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
    """Invoke tegrastats via telemetry.sh script."""
    telemetry_script = os.path.join(os.path.dirname(__file__), "telemetry.sh")
    telemetry_proc = subprocess.Popen(["bash", telemetry_script], preexec_fn=os.setsid)
    atexit.register(terminate_telemetry_process)


def main(args):
    output_dir = "./outputs/profile/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"{COLOR_INFO}Detected {num_gpus} GPU(s).{COLOR_RESET}")

    print(f"{COLOR_INFO}Loading tokenizer: {args.model_name_or_path}{COLOR_RESET}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
                attn_implementation="flash_attention_2"
    )

    print(f"{COLOR_INFO}Loading model: {args.model_name_or_path}{COLOR_RESET}")
    # Use device_map="auto" to automatically shard model across all GPUs
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"{COLOR_INFO}Model loaded with device_map: {model.hf_device_map}{COLOR_RESET}")

    # Put model in eval mode
    model.eval()

    # Start NVTX range
    print("Pushing NVTX range: TimeCapture")
    nvtx.push_range(NVTX_RANGE_NAME)
    torch.cuda.synchronize()

    try:
        if args.all:
            subset_list = get_dataset_config_names("edinburgh-dawg/mmlu-redux")
        else:
            subset_list = [args.subset_name]

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

            print(f"{COLOR_INFO}Subset: {current_subset_name}, Questions: {args.num_questions}, Batch size: {args.batch_size}{COLOR_RESET}")
            print(f"{COLOR_INFO}Log file: {log_file}{COLOR_RESET}")
            print(f"{COLOR_INFO}Results file: {results_file}{COLOR_RESET}\n")
            print("" + "=" * 80)

            with open(log_file, "w") as f:
                f.write(f"Model Name/Path: {args.model_name_or_path}\n")
                f.write(f"Subset: {current_subset_name}\n")
                f.write(f"Number of Questions: {args.num_questions}\n")
                f.write(f"Config: {args.config}\n")
                f.write(f"Batch Size: {args.batch_size}\n")
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
                print(f"{COLOR_DEBUG}Error loading subset '{current_subset_name}': {e}{COLOR_RESET}")
                continue
            except Exception as e:
                print(f"{COLOR_DEBUG}Unexpected error loading dataset: {e}{COLOR_RESET}")
                continue

            # Select first N examples directly rather than using Dataset.from_dict
            if len(full_subset_dataset) < args.num_questions:
                print(f"{COLOR_DEBUG}Warning: Subset '{current_subset_name}' has only {len(full_subset_dataset)} questions, evaluating all.{COLOR_RESET}")
                eval_dataset = full_subset_dataset
            else:
                indices = list(range(args.num_questions))
                eval_dataset = full_subset_dataset.select(indices)

            print(f"{COLOR_INFO}Evaluating on {len(eval_dataset)} questions.{COLOR_RESET}")

            total_inference_time = 0
            correct_count_debug = 0
            total_output_tokens = 0
            choice_labels = ['A', 'B', 'C', 'D']
            label_to_index = {label: i for i, label in enumerate(choice_labels)}

            # Process in batches
            for start in range(0, len(eval_dataset), args.batch_size):
                batch = eval_dataset.select(range(start, min(start + args.batch_size, len(eval_dataset))))
                prompts = []
                for item in batch:
                    choices_str = "\n".join(
                        [f"{choice_labels[i]}. {choice_text}" for i, choice_text in enumerate(item["choices"]) ]
                    )
                    prompt_content = (
                        f"Choose the single best answer (A, B, C, or D) for the following question:\n\n"
                        f"Question: {item['question']}\n\n"
                        f"Choices:\n{choices_str}\n\n"
                        "Concisely, provide only the letter of the correct answer in the format:\n"
                        "Answer: <A/B/C/D>\n"
                    )
                    prompts.append(prompt_content)

                # Tokenize batch
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
                outputs, latency = predict_local(tokenizer, model, inputs)
                prefill_ms, decode_ms = latency

                # Decode each sequence in the batch
                for idx in range(len(batch)):
                    output_ids = outputs[idx]
                    decoded_output = tokenizer.decode(output_ids[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    predicted_choice_letter = extract_predicted_choice(decoded_output)
                    output_tokens = output_ids.shape[-1] - inputs['input_ids'].shape[1]
                    inference_time = (prefill_ms + decode_ms) / 1000
                    total_inference_time += inference_time
                    total_output_tokens += output_tokens

                    item = batch[idx]
                    ground_truth_index = item["answer"]
                    predicted_index = label_to_index.get(predicted_choice_letter, -1)
                    if ground_truth_index == predicted_index:
                        correct_count_debug += 1

                    csv_writer.writerow({
                        "subset": current_subset_name,
                        "question": item["question"],
                        "choices": item["choices"],
                        "ground_truth_index": ground_truth_index,
                        "predicted_choice_letter": predicted_choice_letter,
                        "predicted_index": predicted_index,
                        "full_output": decoded_output,
                        "prefill": prefill_ms,
                        "decode": decode_ms,
                        "inference_time": inference_time,
                        "output_tokens": output_tokens
                    })
                    csv_file.flush()

            logf.close()
            csv_file.close()

            print(f"\n{COLOR_INFO}Correct {correct_count_debug}/{len(eval_dataset)}{COLOR_RESET}")
            total_predictions = len(eval_dataset)
            accuracy = correct_count_debug / total_predictions if total_predictions > 0 else 0
            avg_inference_time = total_inference_time / total_predictions if total_predictions > 0 else 0
            metrics = {"accuracy": accuracy, "average_inference_time_s": avg_inference_time}

            print(f"\n{COLOR_RESULT}Metrics:{COLOR_RESET}")
            print(f"{COLOR_RESULT}  Accuracy: {accuracy:.4f}{COLOR_RESET}")
            print(f"{COLOR_RESULT}  Avg Inference Time: {avg_inference_time:.4f} s/question{COLOR_RESET}")
            print(f"{COLOR_RESULT}  Total Tokens: {total_output_tokens}{COLOR_RESET}")
            print(f"{COLOR_RESULT}  Total Time: {total_inference_time:.4f} s{COLOR_RESET}")

            with open(log_file, "a") as f:
                f.write("\nEvaluation Summary:\n")
                f.write(f"Total Questions: {total_predictions}\n")
                f.write(f"Correct: {correct_count_debug}\n")
                f.write(f"Total Tokens: {total_output_tokens}\n")
                f.write(f"Metrics: {metrics}\n")

            print(f"{COLOR_INFO}Results saved: {results_file}{COLOR_RESET}")
    finally:
        torch.cuda.synchronize()
        nvtx.pop_range()
        print("Popped NVTX range: TimeCapture")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM on MMLU-Redux with batching and multi-GPU")
    parser.add_argument(
        "--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="Hugging Face model or path."
    )
    parser.add_argument(
        "--subset_name", type=str, default="electrical_engineering",
        help="MMLU-Redux subset (e.g., 'electrical_engineering')."
    )
    parser.add_argument(
        "--num_questions", type=int, default=10,
        help="Number of questions to evaluate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for inference to leverage larger GPU memory (e.g., 8, 16)."
    )
    parser.add_argument(
        "--config", type=str, default="reasoning",
        help="Optional config name for logging."
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all subsets")
    args = parser.parse_args()
    main(args)

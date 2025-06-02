import argparse
import sys
import os
from tqdm import tqdm
import time
import re
import pandas as pd
import csv
from datasets import load_dataset, Dataset
import torch
from dotenv import load_dotenv
import subprocess
import atexit
import signal
sys.path.append(os.path.join(os.getcwd(), "src"))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

load_dotenv(dotenv_path=".env_example")

HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


COLOR_RESET = "\033[0m"
COLOR_INFO = "\033[94m"
COLOR_DEBUG = "\033[93m"
COLOR_RESULT = "\033[96m"
COLOR_ERROR = "\033[91m"


#constants
MAX_TOKENS = 128
TBUDGET= 128

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
    decoded_output = decoded_output.strip()

    markdown_pattern = r"\*\*(?:[Aa]nswer|[Cc]hoice|[Oo]ption)(?:[:\s]|\s+is\s+)([A-D])\*\*"
    markdown_matches = re.findall(markdown_pattern, decoded_output)
    if markdown_matches:
        return markdown_matches[-1].upper()

    explicit_pattern = r"(?:[Aa]nswer|[Cc]hoice|[Oo]ption)(?: is)?\s*[:\-]?\s*([A-D])"
    explicit_matches = re.findall(explicit_pattern, decoded_output)
    if explicit_matches:
        return explicit_matches[-1].upper()

    match = re.search(r"^[\[\(\{]?([A-D])[\]\)\}\.]?", decoded_output)
    if match:
        return match.group(1).upper()
    match = re.search(r"[\[\(\{]?([A-D])[\]\)\}\.]?$", decoded_output)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-D])\b", decoded_output)
    if match:
        return match.group(1).upper()

    if decoded_output in ['A', 'B', 'C', 'D']:
        return decoded_output

    return "Invalid"


def predict_local(tokenizer, model, messages, device, tokens=MAX_TOKENS):
    start_time = time.time()
    tokenizer.chat_template = open("chat_deepseek.jinja").read()
    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_token_ids = outputs[0][inputs['input_ids'].shape[1]:]
        decoded_output = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
        output_tokens = generated_token_ids.shape[0]                         
        predicted_choice = extract_predicted_choice(decoded_output)

    except Exception as e:
        print(f"{COLOR_DEBUG}Error during inference: {e}{COLOR_RESET}")
        decoded_output = "Error"
        predicted_choice = "Error"
        output_tokens = 0

    inference_time = time.time() - start_time
    return predicted_choice, decoded_output, inference_time, output_tokens

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
        except Exception as e:
            print(f"{COLOR_ERROR}Error killing nvidia-smi: {e}{COLOR_RESET}")
    
    try:
        subprocess.run(["pkill", "-9", "nvidia-smi"], check=False)
    except:
        pass
        
    telemetry_proc = None


def main(args):
    global output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    safe_model_name = args.model_name_or_path.replace("/", "_")
    log_file = os.path.join(output_dir, f"local_log_{safe_model_name}_{args.subset_name}_{args.num_questions}.txt")
    results_file = os.path.join(output_dir, f"results_{safe_model_name}_{args.subset_name}_{args.num_questions}.csv")

    csv_file = open(results_file, 'w', newline='', encoding='utf-8')
    fieldnames = [
        "subset", "question", "choices", "ground_truth_index",
        "predicted_choice_letter", "predicted_index",
        "full_output", "inference_time", "output_tokens"
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    print(f"{COLOR_INFO}Starting evaluation for model: {args.model_name_or_path}{COLOR_RESET}")
    print(f"{COLOR_INFO}Subset: {args.subset_name}, Questions: {args.num_questions}{COLOR_RESET}")
    print(f"{COLOR_INFO}Log file: {log_file}{COLOR_RESET}")
    print(f"{COLOR_INFO}Results file: {results_file}{COLOR_RESET}")

    with open(log_file, "w") as f:
        f.write(f"Model Name/Path: {args.model_name_or_path}\n")
        f.write(f"Subset: {args.subset_name}\n")
        f.write(f"Number of Questions: {args.num_questions}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Max Tokens: {MAX_TOKENS}\n")
        f.flush()

    logf = open(log_file, "a")

    print(f"{COLOR_INFO}Loading dataset: edinburgh-dawg/mmlu-redux, subset: {args.subset_name}{COLOR_RESET}")
    try:
        full_subset_dataset = load_dataset(
            "edinburgh-dawg/mmlu-redux",
            name=args.subset_name,
            split="test",
            token=HF_READ_TOKEN
        )
    except ValueError as e:
        print(f"{COLOR_DEBUG}Error loading dataset subset '{args.subset_name}': {e}{COLOR_RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{COLOR_DEBUG}An unexpected error occurred during dataset loading: {e}{COLOR_RESET}")
        sys.exit(1)

    if len(full_subset_dataset) < args.num_questions:
        print(f"{COLOR_DEBUG}Warning: Subset '{args.subset_name}' only has {len(full_subset_dataset)} questions. Evaluating on all available.{COLOR_RESET}")
        eval_dataset = full_subset_dataset
    else:
        eval_dataset = Dataset.from_dict(full_subset_dataset[:args.num_questions])

    print(f"{COLOR_INFO}Evaluating on {len(eval_dataset)} questions.{COLOR_RESET}")

    print(f"{COLOR_INFO}Loading tokenizer: {args.model_name_or_path}{COLOR_RESET}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    print(f"{COLOR_INFO}Loading model: {args.model_name_or_path}{COLOR_RESET}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 1},
        trust_remote_code=True
    )
    device = next(model.parameters()).device
    print(f"{COLOR_INFO}Model loaded on device map: {model.hf_device_map}{COLOR_RESET}")

    # start telemetry in its own group
    telemetry_script = os.path.join(os.path.dirname(__file__), "telemetry.sh")
    telemetry_proc = subprocess.Popen(
        ["bash", telemetry_script, output_dir],
        preexec_fn=os.setsid
    )
    atexit.register(terminate_telemetry_process)

    total_inference_time = 0
    correct_count_debug = 0
    total_output_tokens = 0

    choice_labels = ['A', 'B', 'C', 'D']
    label_to_index = {label: i for i, label in enumerate(choice_labels)}

    print(f"{COLOR_INFO}Starting evaluation loop...{COLOR_RESET}")
    for idx, item in enumerate(tqdm(eval_dataset, desc="Evaluating Questions")):
        choices_str = "\n".join(
            [f"{choice_labels[i]}. {choice_text}" for i, choice_text in enumerate(item["choices"])]
        )
        prompt_content = (
            f"Choose the single best answer (A, B, C, or D) for the following question:\n\n"
            f"Question: {item['question']}\n\n"
            f"Choices:\n{choices_str}\n\n"
            # f"Please limit your answer to at most {prompt_token_budget_list.pop(0) * 0.5} tokens.\n"
            f"Please limit your answer to at most {TBUDGET} tokens.\n"
            "Provide only the letter of the correct answer in the format:\n"
            "Answer: <A/B/C/D>\n"
        )

        messages = [
            {"role": "user", "content": prompt_content},
        ]

        predicted_choice_letter, full_decoded_output, inference_time, output_tokens = predict_local(tokenizer, model, messages, device, tokens=TBUDGET)

        total_inference_time += inference_time
        total_output_tokens += output_tokens
        
        logf.write(f"\n--- Item {idx} ---\n")
        logf.write("Prompt:\n" + prompt_content + "\n")
        logf.write("Full model output:\n" + full_decoded_output + "\n")
        logf.write(f"Output Tokens: {output_tokens}\n")
        logf.flush()

        ground_truth_index = item["answer"]
        predicted_index = label_to_index.get(predicted_choice_letter, -1)

        is_correct = (ground_truth_index == predicted_index)
        if is_correct:
            correct_count_debug += 1

        if idx < 5 or not is_correct:
            print(f"\n--- Item {idx} ---")
            print(f"  Ground Truth Index: {COLOR_RESULT}{ground_truth_index}{COLOR_RESET} ({choice_labels[ground_truth_index] if 0 <= ground_truth_index < len(choice_labels) else 'Invalid Index'})")
            print(f"  Predicted Letter:   {COLOR_DEBUG if not is_correct else COLOR_RESULT}{predicted_choice_letter}{COLOR_RESET}")
            print(f"  Predicted Index:    {COLOR_DEBUG if not is_correct else COLOR_RESULT}{predicted_index}{COLOR_RESET}")
            if not is_correct:
                print(f"  {COLOR_ERROR}MISMATCH{COLOR_RESET}")

        csv_writer.writerow({
            "subset": args.subset_name,
            "question": item["question"],
            "choices": item["choices"],
            "ground_truth_index": ground_truth_index,
            "predicted_choice_letter": predicted_choice_letter,
            "predicted_index": predicted_index,
            "full_output": full_decoded_output,
            "inference_time": inference_time,
            "output_tokens": output_tokens
        })
        csv_file.flush()

    # on normal shutdown
    terminate_telemetry_process()

    logf.close()
    csv_file.close()

    print(f"\n{COLOR_INFO}Debug Correct Count: {correct_count_debug}/{len(eval_dataset)}{COLOR_RESET}")

    total_predictions = len(eval_dataset)
    correct_predictions = correct_count_debug

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_inference_time = total_inference_time / total_predictions if total_predictions > 0 else 0

    metrics = {"accuracy": accuracy, "average_inference_time_s": round(avg_inference_time, 2)}

    print(f"\n{COLOR_RESULT}Evaluation Metrics:{COLOR_RESET}")
    print(f"{COLOR_RESULT}  Accuracy: {accuracy:.4f}{COLOR_RESET}")
    print(f"{COLOR_RESULT}  Avg Inference Time: {avg_inference_time:.4f} s/question{COLOR_RESET}")
    print(f"{COLOR_RESULT}  Total Inference Time: {total_inference_time:.4f} s{COLOR_RESET}")
    print(f"{COLOR_RESULT}  Total Tokens Generated: {total_output_tokens}{COLOR_RESET}")

    with open(log_file, "a") as f:
        f.write("\nEvaluation Summary:\n")
        f.write(f"Total Questions Evaluated: {total_predictions}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n")
        f.write(f"Total Tokens Generated: {total_output_tokens}\n")
        f.write(f"Total Inference Time: {total_inference_time:.4f} s\n")
        f.write(f"Metrics: {metrics}\n")

    print(f"{COLOR_INFO}Detailed results saved to: {results_file}{COLOR_RESET}")


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
        default=100,
        help="Number of questions to evaluate from the subset."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="reasoning",
        help="Optional configuration name for logging."
    )
    args = parser.parse_args()
    main(args)

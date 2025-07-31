import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
from collections import Counter
import random

# Set GPU constraint
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class MMLUMajorityVoteEvaluator:
    def __init__(self, model_name="gpt2", num_samples=8):
        self.model_name = model_name
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on: {next(self.model.parameters()).device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    def format_mmlu_prompt(self, question, choices):
        """Format MMLU question as a prompt"""
        prompt = f"Question: {question}\n\n"
        choice_labels = ['A', 'B', 'C', 'D']
        
        for i, choice in enumerate(choices):
            prompt += f"{choice_labels[i]}) {choice}\n"
        
        prompt += "\nAnswer: "
        return prompt
    
    def generate_multiple_samples(self, prompt):
        """Generate multiple samples for a single prompt"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        samples = []
        with torch.no_grad():
            # Generate samples in batches to manage memory
            batch_size = min(4, self.num_samples)
            for i in range(0, self.num_samples, batch_size):
                current_batch_size = min(batch_size, self.num_samples - i)
                
                outputs = self.model.generate(
                    inputs,
                    num_return_sequences=current_batch_size,
                    max_new_tokens=20,  # Short answers for multiple choice
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
                
                for output in outputs:
                    # Decode only the new tokens (remove the prompt)
                    generated_text = self.tokenizer.decode(
                        output[len(inputs[0]):], 
                        skip_special_tokens=True
                    )
                    samples.append(generated_text.strip())
        
        return samples
    
    def extract_answer(self, generated_text):
        """Extract the answer choice (A, B, C, or D) from generated text"""
        patterns = [
            r'\b([ABCD])\)',
            r'\b([ABCD])\.',
            r'\b([ABCD])\s',
            r'\(([ABCD])\)',
            r'^([ABCD])$',
            r'\b([ABCD])(?=\s|$)'
        ]
        
        text = generated_text.upper().strip()
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        for char in ['A', 'B', 'C', 'D']:
            if char in text:
                return char
        
        return "Invalid"
    
    def majority_vote(self, answers):
        """Perform majority voting on extracted answers"""
        if not answers:
            return "Invalid"
        
        vote_counts = Counter(answers)
        most_common = vote_counts.most_common(1)
        return most_common[0][0]
    
    def evaluate_single_question(self, question, choices, correct_answer):
        """Evaluate a single MMLU question using majority voting"""
        prompt = self.format_mmlu_prompt(question, choices)
        
        print(f"\nPrompt:\n{prompt}")
        
        # Generate multiple samples
        samples = self.generate_multiple_samples(prompt)
        
        print(f"\nGenerated samples:")
        extracted_answers = []
        for i, sample in enumerate(samples):
            answer = self.extract_answer(sample)
            extracted_answers.append(answer)
            print(f"Sample {i+1}: '{sample}' -> Answer: {answer}")
        
        # Perform majority voting
        final_answer = self.majority_vote(extracted_answers)
        is_correct = final_answer == correct_answer
        
        print(f"\nExtracted answers: {extracted_answers}")
        print(f"Vote counts: {dict(Counter(extracted_answers))}")
        print(f"Majority vote: {final_answer}")
        print(f"Correct answer: {correct_answer}")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        
        return is_correct, final_answer, extracted_answers
    
    def evaluate_mmlu_subset(self, subject="anatomy", num_questions=5):
        """Evaluate a subset of MMLU questions"""
        print(f"Loading MMLU dataset for subject: {subject}")
        
        try:
            # Load MMLU test set
            dataset = load_dataset("cais/mmlu", subject)
            test_data = dataset["test"]
            
            # Sample random questions
            total_questions = len(test_data)
            indices = random.sample(range(total_questions), min(num_questions, total_questions))
            
            correct_count = 0
            results = []
            
            for i, idx in enumerate(indices):
                print(f"\n{'='*80}")
                print(f"Question {i+1}/{num_questions} (Index: {idx})")
                print('='*80)
                
                item = test_data[idx]
                question = item["question"]
                choices = item["choices"]
                correct_idx = item["answer"]
                correct_answer = ['A', 'B', 'C', 'D'][correct_idx]
                
                is_correct, predicted_answer, sample_answers = self.evaluate_single_question(
                    question, choices, correct_answer
                )
                
                if is_correct:
                    correct_count += 1
                
                results.append({
                    'question': question,
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'predicted_answer': predicted_answer,
                    'sample_answers': sample_answers,
                    'is_correct': is_correct
                })
            
            accuracy = correct_count / num_questions
            print(f"\n{'='*80}")
            print(f"FINAL RESULTS")
            print(f"{'='*80}")
            print(f"Subject: {subject}")
            print(f"Model: {self.model_name}")
            print(f"Number of samples per question: {self.num_samples}")
            print(f"Questions evaluated: {num_questions}")
            print(f"Correct answers: {correct_count}")
            print(f"Accuracy: {accuracy:.2%}")
            
            return results, accuracy
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using a sample question for demonstration...")
            
            # Fallback sample question
            sample_question = "What is the powerhouse of the cell?"
            sample_choices = ["Nucleus", "Mitochondria", "Ribosome", "Endoplasmic reticulum"]
            correct_answer = "B"
            
            is_correct, predicted_answer, sample_answers = self.evaluate_single_question(
                sample_question, sample_choices, correct_answer
            )
            
            return [{'question': sample_question, 'predicted_answer': predicted_answer, 'is_correct': is_correct}], int(is_correct)

def main():
    # Initialize evaluator
    evaluator = MMLUMajorityVoteEvaluator(model_name="gpt2", num_samples=8)
    
    # Evaluate on a subset of MMLU
    subjects = ["anatomy", "astronomy", "business_ethics"]  # You can add more subjects
    
    overall_results = []
    overall_accuracy = []
    
    for subject in subjects:
        print(f"\n{'#'*100}")
        print(f"EVALUATING SUBJECT: {subject.upper()}")
        print(f"{'#'*100}")
        
        try:
            results, accuracy = evaluator.evaluate_mmlu_subset(subject=subject, num_questions=3)
            overall_results.extend(results)
            overall_accuracy.append(accuracy)
        except Exception as e:
            print(f"Error evaluating {subject}: {e}")
            continue
    
    if overall_accuracy:
        print(f"\n{'#'*100}")
        print(f"OVERALL RESULTS ACROSS ALL SUBJECTS")
        print(f"{'#'*100}")
        print(f"Average accuracy: {sum(overall_accuracy)/len(overall_accuracy):.2%}")
        print(f"Subjects evaluated: {len(overall_accuracy)}")

if __name__ == "__main__":
    main()

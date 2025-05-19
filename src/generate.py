import argparse
import json
import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.utils import preprocess_and_tokenize_data, get_context, get_text_to_classify, decode
from utils.data_analysis import calculate_self_bleu_score, calculate_average_length, calculate_diversity_for_sample, calculate_flesch_readability
from utils.detector import Detector
import time
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import set_seed
import math

# Functions
def parse_args():
    parser = argparse.ArgumentParser(description="Script to generate text using a pretrained GPT model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on, e.g., 'cpu' or 'cuda:0'.")
    parser.add_argument("--experiment_path", type=str, required=True, help="Path to save generated texts in JSONL format.")
    parser.add_argument("--input_token_length", type=int, default=96, help="Length of input tokens.")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for chunking the dataset.")
    parser.add_argument("--num_samples", type=int, help="Number of samples to generate.")
    parser.add_argument("--model_name", type=str, required=True, help="The model name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_name", type=str, required=False, help="The name of the dataset to use.")
    parser.add_argument("--dataset_config_name", type=str, required=False, help="The configuration name of the dataset to use.")
    parser.add_argument("--dataset_filepath", type=str, required=False, help="The filepath of the dataset to use.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--iteration", type=int, required=True, help="The model iteration")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generating texts.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4,help="The number of processes to use for the preprocessing." )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generating texts.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling temperature for generating texts.")
    parser.add_argument("--top_k", type=int, default=0, help="Top k sampling num for generating texts.")
    parser.add_argument("--beam_search", type=int, default=0, help="Whether to carry out beam search decoding")
    parser.add_argument("--torch_dtype", type=str, default=None, help="Override the default `torch.dtype` and load the model under this dtype.")
    parser.add_argument("--low_cpu_mem_usage", type=str, default=False, help="Whether to use low cpu mem usage")
    parser.add_argument("--self_bleu_n_sample", type=int, default=1000, help="Number of samples to use for self bleu")
    parser.add_argument("--classify_text", type=int, default=0, help="Whether to classify text")
    parser.add_argument("--detector_tokenizer_name", type=str, help="Path to model checkpoint")
    parser.add_argument("--detector_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--detector_threshold", type=float, help="Detector threshold")
    parser.add_argument("--detector_temperature", type=float, default=1.395004153251648, help="Detector temperature")
    return parser.parse_args()

def load_dataset_for_generation(args):
    """Loads the dataset."""
    if args.dataset_name is not None:
        print("Loading dataset from hf")
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        dataset = raw_datasets["train"]
    else:
        assert args.dataset_filepath is not None, "Please provide a dataset filepath."
        print(f"Loading dataset from {args.dataset_filepath}")
        with open(args.dataset_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = Dataset.from_list(data)

    return dataset

def generate_texts(dataloader, tokenizer, model, args):
    """Generates texts from the dataset using the model."""

    if args.temperature==0.0:
        do_sample=False
    else:
        do_sample=True

    new_tokens = args.block_size - args.input_token_length

    generated_texts = []
    tokenized_generated_texts = []

    start_time=time.time()
    # Loop through the dataset in batches
    for batch in tqdm(dataloader, desc="Generating texts"):

        batch_input_ids, batch_attention_mask = batch[0].to(args.device), batch[1].to(args.device)

        with torch.no_grad():

            if args.beam_search:
                generated_ids = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    min_new_tokens=new_tokens,
                    max_new_tokens=new_tokens,
                    # max_length=args.block_size,
                    pad_token_id=tokenizer.eos_token_id,
                    num_beams=5,
                    early_stopping=True
                )
            else:
                # Generate outputs for the entire batch
                generated_ids = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    min_new_tokens=new_tokens,
                    max_new_tokens=new_tokens,
                    # max_length=args.block_size,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )

        # Decode and print each sequence in the batch
        tokenized_generated_texts.extend(generated_ids.tolist())

        decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts.extend(decoded_texts)
    
    total_time = time.time() - start_time
    print(f"Total generation time: {total_time:.2f} seconds")

    # clean up
    del model
    del dataloader
    torch.cuda.empty_cache()

    return generated_texts, tokenized_generated_texts

def main():

    # Initialize wandb
    args = parse_args()

    wandb.init(
        name=f"generate_iteration_{args.iteration}",  # Name for this specific run
        config=vars(args),  # Save arguments to wandb
    )

    set_seed(args.seed)

    save_path = os.path.join(args.experiment_path, str(args.iteration), "data")
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                    torch_dtype=torch_dtype, 
                                                    low_cpu_mem_usage=args.low_cpu_mem_usage)
    model.to(args.device)
    model.eval()

    # Load dataset
    dataset = load_dataset_for_generation(args)

    dataset = preprocess_and_tokenize_data(raw_datasets=dataset,
                                            tokenizer=tokenizer,
                                            column_names=list(dataset.features),
                                            max_length=args.input_token_length,
                                            preprocessing_num_workers=args.preprocessing_num_workers,
                                            overwrite_cache=True,
                                            text_column_name="context")

    # Sample number of samples
    if args.num_samples:
        dataset = dataset.select(range(args.num_samples))

    # Create TensorDataset
    dataset = TensorDataset(torch.tensor(dataset["input_ids"]), torch.tensor(dataset["attention_mask"]))

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,  # Pin memory for faster GPU transfers
        drop_last=False,  # Don't drop the last batch
        shuffle=False  # Keep order consistent, or set to True if desired
    )

    # Generate texts
    generated_texts, tokenized_generated_texts = generate_texts(dataloader, tokenizer, model, args)

    dataset = Dataset.from_dict({
        "text": generated_texts,
        "input_ids": tokenized_generated_texts
    })

    dataset_w_trunc = dataset.map(
        lambda example: get_text_to_classify(example, args.input_token_length), 
        batched=False,
        desc=f"Truncating texts to last {args.input_token_length} tokens",
    )
    
    def filter_eos(example, eos_token_id):
        example["cls_input_ids"] = [token for token in example["cls_input_ids"] if token != eos_token_id]
        return example

    processed_dataset = dataset_w_trunc.map(
        lambda example: filter_eos(example, tokenizer.eos_token_id), 
        batched=False,
        desc=f"Filtering special tokens",
    )
    
    processed_dataset_with_cls_text = processed_dataset.map(
        lambda example: decode(example, tokenizer, 'cls_text', 'cls_input_ids'), 
        batched=False,
        desc=f"Decoding",
    )

    if args.classify_text and args.detector_path is not None:
        detector = Detector(tokenizer_name=args.detector_tokenizer_name, detector_path=args.detector_path, device=args.device)        
        
        processed_dataset_with_cls_text = processed_dataset_with_cls_text.map(
            lambda example: detector.predict_batch(example, cls_text_key="cls_text", max_length=(args.block_size - args.input_token_length), threshold=float(args.detector_threshold), temperature=float(args.detector_temperature)),
            batched=True,
            desc="Classifying texts"
        )

        metrics = detector.evaluate(processed_dataset_with_cls_text, data_label=1)

    else:
        metrics = {}

    processed_dataset_with_cls_text = processed_dataset_with_cls_text.map(calculate_diversity_for_sample)

    output = []
    for row in processed_dataset_with_cls_text:
        output.append({
            'text': row['text'],
            'cls_text': row['cls_text']
        })
        if 'cls_score' in row:
            output[-1]['cls_score'] = row['cls_score']
        if 'cls_confidence' in row:
            output[-1]['cls_confidence'] = row['cls_confidence']
        if 'diversity' in row:
            output[-1]['diversity'] = row['diversity']

    print("Saving data")
    with open(f"{save_path}.json", 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Generated {len(generated_texts)} texts.")
    diversity = processed_dataset_with_cls_text.to_pandas()["diversity"].mean()

    if args.self_bleu_n_sample >= len(processed_dataset_with_cls_text):
        args.self_bleu_n_sample = math.ceil(len(processed_dataset_with_cls_text)/10)

    self_bleu = calculate_self_bleu_score(processed_dataset_with_cls_text, args.self_bleu_n_sample)
    readability = calculate_flesch_readability(processed_dataset_with_cls_text)
    avg_length = calculate_average_length(processed_dataset_with_cls_text)

    metrics = {
        **metrics,
        "diversity": diversity,
        **self_bleu,
        **avg_length,
        **readability
    }

    wandb.log(metrics, step=args.iteration)

    with open(f"{save_path}_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    wandb.finish()

if __name__ == "__main__":
    main()
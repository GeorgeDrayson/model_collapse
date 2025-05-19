from itertools import chain
import json
from datasets import concatenate_datasets, DatasetDict, Dataset
from math import ceil
import os
import random
import csv
import matplotlib.pyplot as plt
import numpy as np

def decode(example, tokenizer, text_key, input_ids_key):
    example[text_key] = tokenizer.decode(example[input_ids_key], skip_special_tokens=True)
    return example

def get_text_to_classify(example, truncate_up_to):
    example["cls_input_ids"] = example["input_ids"][truncate_up_to:]
    return example

def get_context(example, input_token_length, input_ids_key="input_ids", attention_mask_key="attention_mask"):
    """Slice input_ids and attention_mask to the specified length."""
    return {
        input_ids_key: example["input_ids"][:input_token_length],
        attention_mask_key: example["attention_mask"][:input_token_length],
    }

def preprocess_and_tokenize_data(raw_datasets, tokenizer, column_names, max_length, preprocessing_num_workers, overwrite_cache, text_column_name="text"):

    # define tokenize function
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding='max_length', truncation=True, max_length=max_length)
        
    processing_config = {
        'batched': True,
        'num_proc': preprocessing_num_workers,
        'load_from_cache_file': not overwrite_cache,
    }

    # tokenize
    tokenization_config = {**processing_config, 'remove_columns': column_names, 'desc': "Running tokenizer on dataset"}

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        **tokenization_config,
    )

    # make labels equal to input ids
    labelled_datasets = tokenized_datasets.map(add_labels, **processing_config)

    return labelled_datasets

def truncate_dataset(tokenized_dataset, truncate_up_to):

    def truncate_function(example, first_n_tokens):
        """Slice input_ids and attention_mask up to the specified length."""
        if "attention_mask" in example:
            return {
                "input_ids": example["input_ids"][first_n_tokens:],
                "attention_mask": example["attention_mask"][first_n_tokens:],
            }
        else:
            return {
                "input_ids": example["input_ids"][first_n_tokens:]
            }

    processed_dataset = tokenized_dataset.map(
        lambda example: truncate_function(example, truncate_up_to), 
        batched=False,
        desc=f"Truncating texts to last {truncate_up_to} tokens",
    )
    
    return processed_dataset

def group_texts_and_tokenize_data(raw_datasets, tokenizer, column_names, block_size, preprocessing_num_workers, overwrite_cache):

    # subtract 1 from block size to account for the bos token
    block_size -= 1

    def group_texts(examples, block_size):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    text_column_name = "text"

    # define tokenize function
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], add_special_tokens=False)

    processing_config = {
        'batched': True,
        'num_proc': preprocessing_num_workers,
        'load_from_cache_file': not overwrite_cache,
    }

    # tokenize
    tokenization_config = {**processing_config, 'remove_columns': column_names, 'desc': "Running tokenizer on dataset"}

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        **tokenization_config,
    )
  
    block_size = min(block_size, tokenizer.model_max_length)

    processed_datasets = tokenized_datasets.map(
        group_texts,
        **processing_config,
        desc=f"Grouping texts in chunks of {block_size}",
        fn_kwargs={'block_size': block_size}
    )

    # make labels equal to input ids
    labelled_datasets = processed_datasets.map(add_labels, **processing_config)

    return labelled_datasets

def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

def load_dataset_from_json(dataset_filepath):
    """Loads the dataset."""
    print(f"Loading dataset from {dataset_filepath}")
    with open(dataset_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return  Dataset.from_list(data)

def save_to_jsonl(generated_texts, save_path):
    """Saves generated texts to a file in JSONL format."""
    with open(save_path, 'a') as f:
        for text in generated_texts:
            entry = {"text": text}
            f.write(json.dumps(entry) + '\n')

def save_to_json(generated_texts, save_path):
    """Saves generated texts to a file in JSON format."""
    with open(save_path, 'w') as f:
        json.dump([{"text": text} for text in generated_texts], f, indent=4)
        
def convert_jsonl_to_json(input_file, output_file):
    """Converts a JSONL file to a single JSON file."""
    with open(input_file, 'r') as infile:
        data = [json.loads(line) for line in infile]

    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Converted {input_file} to {output_file}")

def load_dataset(base_dir):
    assert os.path.exists(base_dir), f"Directory not found: {base_dir}"

    with open(base_dir, 'r') as f:
        data = json.load(f)

    return data

def plot_metric_over_generations(eval_results, metric_name, y_label, label, color, save_path=None):

    plt.figure(figsize=(10, 6))

    iterations = list(eval_results.keys())

    perplexities = [eval_results[it][metric_name] for it in iterations]

    plt.plot(iterations, perplexities, marker='o', label=f'{label}', color=color)

    plt.title(f"{y_label} over Generations")
    plt.xlabel('Generation')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='png')
    else:
        plt.show()
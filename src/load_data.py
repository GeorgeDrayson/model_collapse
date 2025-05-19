import os
import sys
import json
import torch
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.utils import truncate_dataset, save_to_json, group_texts_and_tokenize_data, get_text_to_classify, decode, get_context
from utils.detector import Detector

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        "./data",
        "./data/wikitext2",
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def load_raw_data():
    """Load WikiText-2 dataset and concatenate every 1000 texts."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    dataset["train"] = dataset["train"].filter(lambda x: len(x["text"]) > 0)
    dataset["test"] = dataset["test"].filter(lambda x: len(x["text"]) > 0)
    dataset["validation"] = dataset["validation"].filter(lambda x: len(x["text"]) > 0)

    return dataset["train"], dataset["validation"],  dataset["test"]

def process_dataset(raw_data, tokenizer, block_size=512, input_token_length=256, train=True):
    """Process and tokenize the dataset."""
    print("Processing and tokenizing dataset...")
    
    # Initial tokenization and processing
    processed_dataset = group_texts_and_tokenize_data(
        raw_datasets=raw_data,
        tokenizer=tokenizer,
        column_names=list(raw_data.features),
        block_size=block_size,
        preprocessing_num_workers=4,
        overwrite_cache=True,
    )
    
    processed_dataset = processed_dataset.map(
        lambda example: decode(example, tokenizer, 'text', 'input_ids'),
        batched=False,
        desc="Decoding full text",
    )
    
    if train:
        processed_dataset = processed_dataset.map(
            lambda example: get_context(example, input_token_length, input_ids_key="context_input_ids", attention_mask_key="context_attention_mask"), 
            batched=False,
            desc=f"Getting context"
        )
        
        processed_dataset = processed_dataset.map(
            lambda example: get_text_to_classify(example, input_token_length),
            batched=False,
            desc=f"Truncating texts to last {input_token_length} tokens"
        )
        
        processed_dataset = processed_dataset.map(
            lambda example: decode(example, tokenizer, 'context', 'context_input_ids'),
            batched=False,
            desc="Decoding context",
        )
        
        processed_dataset = processed_dataset.map(
            lambda example: decode(example, tokenizer, 'cls_text', 'cls_input_ids'),
            batched=False,
            desc="Decoding classification text"
        )

    return processed_dataset

def classify_dataset(dataset, detector, tokenizer, block_size=512, input_token_length=256, cfg=None):
    """Classify the processed dataset using the detector."""
    
    print("Classifying dataset...")
    classified_dataset = dataset.map(
        lambda example: detector.predict_batch(
            example, 
            cls_text_key="cls_text",
            max_length=(block_size - input_token_length),
            threshold=cfg.detector.ai_confidence_threshold,
            temperature=cfg.detector.temperature
        ),
        batched=True,
        desc="Classifying texts"
    )
    
    return classified_dataset

def save_dataset(classified_dataset, output_path):
    """Save the classified dataset to JSON."""
    print(f"Saving classified dataset to {output_path}...")
    output = []
    for row in classified_dataset:
        output_row = {
            'text': row['text'],
        }
        if 'context' in row:
            output_row['context'] = row['context']
        if 'cls_text' in row:
            output_row['cls_text'] = row['cls_text']
        if 'cls_score' in row:
            output_row['cls_score'] = row['cls_score']
        if 'cls_confidence' in row:
            output_row['cls_confidence'] = row['cls_confidence']
        output.append(output_row)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup configuration
    block_size = cfg.train.block_size
    input_token_length = block_size - cfg.train.loss_on_last_n_tokens
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Create directories
    setup_directories()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    
    # Load detector
    print("Loading detector...")
    detector = Detector(
        tokenizer_name=cfg.detector.tokenizer_name,
        detector_path=cfg.detector.model_path,
        device=device
    )
    
    try:
        # Load and save raw data
        train_data, validation_data, test_data = load_raw_data()
        
        # Process training data
        processed_train = process_dataset(train_data, tokenizer, block_size, input_token_length)
        processed_validation = process_dataset(validation_data, tokenizer, block_size, input_token_length, train=False)
        processed_test = process_dataset(test_data, tokenizer, block_size, input_token_length, train=False)
        
        # Classify training data
        classified_train = classify_dataset(processed_train, detector, tokenizer, block_size, input_token_length, cfg)
        
        # Save classified data
        save_dataset(classified_train, "./data/wikitext2/train.json")
        save_dataset(processed_validation, "./data/wikitext2/validation.json")
        save_dataset(processed_test, "./data/wikitext2/test.json")
        
        print("Dataset preparation completed successfully!")
        
    finally:
        # Cleanup
        print("Cleaning up...")
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 
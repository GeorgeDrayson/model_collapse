import os
import sys
import json
import torch
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.utils import truncate_dataset, save_to_json, preprocess_and_tokenize_data
from utils.detector import Detector

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        "./data",
        "./data/wikitext2",
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def load_and_save_raw_data():
    """Load WikiText-2 dataset and save raw versions."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Convert and save datasets in simple format
    for split_name in ["train", "test"]:
        output = []
        for item in dataset[split_name]:
            output.append({"text": item["text"]})
        
        output_path = f"./data/wikitext2/{split_name}.json"
        print(f"Saving {split_name} data to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
    
    return dataset["train"], dataset["test"]

def process_dataset(raw_data, tokenizer, block_size=512, input_token_length=256):
    """Process and tokenize the dataset."""
    print("Processing and tokenizing dataset...")
    
    # Initial tokenization and processing
    processed_dataset = preprocess_and_tokenize_data(
        raw_datasets=raw_data,
        tokenizer=tokenizer,
        column_names=list(raw_data.features),
        group_texts_bool=True,
        block_size=block_size,
        preprocessing_num_workers=4,
        overwrite_cache=True
    )
    
    # Add classification text
    def get_text_to_classify(example, truncate_up_to):
        example["cls_input_ids"] = example["input_ids"][truncate_up_to:]
        return example

    processed_dataset = processed_dataset.map(
        lambda example: get_text_to_classify(example, input_token_length),
        batched=False,
        desc=f"Truncating texts to last {input_token_length} tokens"
    )
    
    # Decode texts
    def decode(example, text_key, input_ids_key):
        example[text_key] = tokenizer.decode(example[input_ids_key], skip_special_tokens=True)
        return example

    processed_dataset = processed_dataset.map(
        lambda example: decode(example, 'text', 'input_ids'),
        batched=False,
        desc="Decoding full text"
    )

    processed_dataset = processed_dataset.map(
        lambda example: decode(example, 'cls_text', 'cls_input_ids'),
        batched=False,
        desc="Decoding classification text"
    )
    
    # Remove unnecessary columns
    processed_dataset = processed_dataset.remove_columns(['input_ids', 'attention_mask', 'labels', 'cls_input_ids'])
    
    return processed_dataset

def classify_dataset(dataset, detector, block_size=512, input_token_length=256, cfg=None):
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

def save_classified_dataset(classified_dataset, output_path):
    """Save the classified dataset to JSON."""
    print(f"Saving classified dataset to {output_path}...")
    output = []
    for row in classified_dataset:
        output_row = {
            'text': row['text'],
            'cls_text': row['cls_text']
        }
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
        train_data, test_data = load_and_save_raw_data()
        
        # Process training data
        processed_train = process_dataset(train_data, tokenizer, block_size, input_token_length)
        
        # Classify training data
        classified_train = classify_dataset(processed_train, detector, block_size, input_token_length, cfg)
        
        # Save classified data
        save_classified_dataset(classified_train, "./data/wikitext2/classified_human_data.json")
        
        print("Dataset preparation completed successfully!")
        
    finally:
        # Cleanup
        print("Cleaning up...")
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 
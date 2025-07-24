"""Language reasoning dataset builder for HRM adaptation."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Add the parent directory to the system path to allow relative imports
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional, Any, Dict, List

import numpy as np
from datasets import load_dataset
from pydantic import BaseModel
from tqdm import tqdm
from transformers import GPT2Tokenizer

from dataset.common import PuzzleDatasetMetadata


class LanguageDataProcessConfig(BaseModel):
    """Configuration for language dataset processing."""
    output_dir: str = "data/language-reasoning-10k"
    tokenizer_name: str = "gpt2"
    seq_len: int = 1024
    subsample_size: Optional[int] = 10000
    
    # Dataset sources
    reasoning_datasets: List[str] = [
        "openai/gsm8k",  # Math reasoning
        "lukaemon/bbh",  # Big Bench Hard
        # Add more reasoning datasets
    ]


def tokenize_example(tokenizer: Any, question: str, answer: str, seq_len: int):
    """Tokenize a question-answer pair for HRM format."""
    # Format: [QUESTION] <sep> [ANSWER] <eos>
    sep_token = tokenizer.sep_token if tokenizer.sep_token else " <sep> "
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    
    text = f"{question}{sep_token}{answer}{eos_token}"
    
    tokens = tokenizer.encode(text, max_length=seq_len, truncation=True, padding="max_length")
    
    # Create input (question + sep) and label (answer + eos)
    question_tokens = tokenizer.encode(question + sep_token, add_special_tokens=False)
    
    input_tokens = tokens.copy()
    label_tokens = tokens.copy()
    
    # Mask question part in labels (only train on answer)
    for i in range(len(question_tokens)):
        if i < len(label_tokens):
            label_tokens[i] = -100  # IGNORE_LABEL_ID
    
    return np.array(input_tokens, dtype=np.int32), np.array(label_tokens, dtype=np.int32)


def load_reasoning_data(config: LanguageDataProcessConfig) -> List[Dict[str, str]]:
    """Load and combine reasoning datasets."""
    all_examples = []
    
    # GSM8K math reasoning
    if "openai/gsm8k" in config.reasoning_datasets:
        print("Loading GSM8K dataset...")
        dataset = load_dataset("openai/gsm8k", "main")
        
        for split in ["train", "test"]:
            if split in dataset:
                split_data = list(dataset[split])  # type: ignore  # datasets library uses dynamic typing
                for example in tqdm(split_data, desc=f"Processing GSM8K {split}"):
                    # Type: ignore for dataset access - datasets library has dynamic typing
                    example_dict = dict(example)  # type: ignore
                    all_examples.append({
                        "question": example_dict["question"],
                        "answer": example_dict["answer"],
                        "source": f"gsm8k_{split}",
                        "difficulty": "medium"
                    })
    
    # Add more datasets here...
    
    return all_examples


def convert_dataset(config: LanguageDataProcessConfig):
    """Convert language reasoning data to HRM format."""
    print(f"Building language reasoning dataset: {config.output_dir}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load reasoning data
    examples = load_reasoning_data(config)
    
    # Subsample if requested
    if config.subsample_size and len(examples) > config.subsample_size:
        random.shuffle(examples)  # Use random.shuffle instead of np.shuffle
        examples = examples[:config.subsample_size]
    
    print(f"Processing {len(examples)} examples...")
    
    # Split train/test
    split_idx = int(0.9 * len(examples))
    splits = {
        "train": examples[:split_idx],
        "test": examples[split_idx:]
    }
    
    for split_name, split_examples in splits.items():
        print(f"Processing {split_name} split ({len(split_examples)} examples)...")
        
        # Process examples
        results = {
            "inputs": [],
            "labels": [], 
            "puzzle_identifiers": [],
            "puzzle_indices": [],
            "group_indices": [0]  # Start with 0
        }
        
        example_id = 0
        puzzle_id = 0
        
        for example in tqdm(split_examples, desc=f"Tokenizing {split_name}"):
            input_tokens, label_tokens = tokenize_example(
                tokenizer, 
                example["question"], 
                example["answer"], 
                config.seq_len
            )
            
            results["inputs"].append(input_tokens)
            results["labels"].append(label_tokens)
            results["puzzle_identifiers"].append(0)  # Single identifier for now
            
            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
        
        results["group_indices"].append(puzzle_id)
        
        # Convert to numpy
        final_results = {
            "inputs": np.stack(results["inputs"]),
            "labels": np.stack(results["labels"]),
            "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
            "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32), 
            "group_indices": np.array(results["group_indices"], dtype=np.int32)
        }
        
        # Save arrays
        save_dir = Path(config.output_dir) / split_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for key, array in final_results.items():
            np.save(save_dir / f"all__{key}.npy", array)
        
        # Create metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=config.seq_len,
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_token_id,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(results["group_indices"]) - 1,
            mean_puzzle_examples=1.0,
            sets=["all"]
        )
        
        # Save metadata
        with open(save_dir / "dataset.json", "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
    
    print(f"✅ Dataset saved to {config.output_dir}")


def process_language_data(args):
    cfg = LanguageDataProcessConfig(
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        seq_len=args.seq_len,
        subsample_size=args.subsample_size,
    )
    convert_dataset(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/language-reasoning-10k")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--subsample-size", type=int, default=10000)
    args = parser.parse_args()
    process_language_data(args) 
import os
import argparse
import numpy as np
from datasets import load_dataset
import tiktoken
import json

def main(args):
    # Load the dataset
    print(f"Loading dataset '{args.dataset_name}'...")
    dataset = load_dataset(args.dataset_name, name=args.dataset_config)
    
    # The gsm8k dataset has 'question' and 'answer' fields. We'll combine them.
    def combine_text(example):
        return {"text": example['question'] + "\n" + example['answer']}
    
    dataset = dataset.map(combine_text, remove_columns=['question', 'answer'])

    # Get the tokenizer
    print(f"Initializing tokenizer '{args.tokenizer_name}'...")
    enc = tiktoken.get_encoding(args.tokenizer_name)
    
    # Process each split
    for split_name in dataset.keys():
        print(f"Processing split: {split_name}...")
        
        # Concatenate all text documents into one large string, separated by EOS token
        # This is a common strategy for training on large corpora
        all_tokens = []
        for example in dataset[split_name]:
            text = example['text']
            if text: # Ensure text is not empty
                tokens = enc.encode_ordinary(text)
                all_tokens.extend(tokens + [enc.eot_token]) # Add EOS token after each document

        # Convert to numpy array
        arr = np.array(all_tokens, dtype=np.uint16)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save to binary file
        bin_path = os.path.join(args.output_dir, f"{split_name}.bin")
        print(f"Saving {len(arr)} tokens to {bin_path}")
        arr.tofile(bin_path)

    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'tokenizer': args.tokenizer_name,
    }
    meta_path = os.path.join(args.output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
        
    print("✅ Pre-tokenization complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-tokenize a dataset and save it to a binary format.")
    parser.add_argument('--dataset-name', type=str, default='gsm8k', help="Name of the Hugging Face dataset to use.")
    parser.add_argument('--dataset-config', type=str, default='main', help="Configuration for the dataset (e.g., 'main' for gsm8k).")
    parser.add_argument('--tokenizer-name', type=str, default='gpt2', help="Name of the tiktoken tokenizer to use (e.g., 'gpt2', 'r50k_base').")
    parser.add_argument('--output-dir', type=str, default='hrm-language/data/gsm8k-tokenized', help="Directory to save the tokenized .bin files and meta.json.")
    
    args = parser.parse_args()
    main(args) 
import argparse
import random
import numpy as np
import pandas as pd
import torch
import ast
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification


# Seed function
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Function for performing inference and returning predicted indices
@torch.no_grad()
def predict(model, tokenizer, tokens, device, max_len=128):
    # Tokenizing the input tokens 
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len
    )
    word_ids = enc.word_ids()           # Getting word-level indices corresponding to tokens
    enc = {k: v.to(device) for k, v in enc.items()}     # Moving encoded inputs to the proper device

    outputs = model(**enc)          # Performing forward pass through the model
    pred_ids = torch.argmax(outputs.logits, dim=2)[0].tolist()      # Getting predicted label indices

    # Selecting word indices with label B (1) or I (2)
    indices = [word_ids[i] for i in range(len(word_ids))
               if word_ids[i] is not None and pred_ids[i] in [1, 2]]
    
    return sorted(set(indices)) if indices else [-1]        # Returning sorted unique indices, or [-1] if none were found


# Main function to load data, models, and perform predictions
def main(args):
    set_seed(42)

    # Defining label mappings
    label2id = {"O": 0, "B": 1, "I": 2}
    id2label = {v: k for k, v in label2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading test data from CSV
    test_df = pd.read_csv(args.test_path)
    # Converting tokenized sentence string into Python list
    test_df["tokenized_sentence"] = test_df["tokenized_sentence"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Splitting test set by language
    test_tr = test_df[test_df["language"] == "tr"].reset_index(drop=True)
    test_it = test_df[test_df["language"] == "it"].reset_index(drop=True)

    # Loading Turkish tokenizer and model
    tokenizer_tr = AutoTokenizer.from_pretrained(args.model_name_tr)
    model_tr = AutoModelForTokenClassification.from_pretrained(
        args.model_name_tr, num_labels=3, id2label=id2label, label2id=label2id
    )
    model_tr.load_state_dict(torch.load(os.path.join(args.model_dir_tr, "model_tr.pt"), map_location=device))
    model_tr.to(device).eval()

    # Loading Italian tokenizer and model
    tokenizer_it = AutoTokenizer.from_pretrained(args.model_name_it)
    model_it = AutoModelForTokenClassification.from_pretrained(
        args.model_name_it, num_labels=3, id2label=id2label, label2id=label2id
    )
    model_it.load_state_dict(torch.load(os.path.join(args.model_dir_it, "model_it.pt"), map_location=device))
    model_it.to(device).eval()

    # Predicting entity indices for Turkish test data
    test_tr["indices"] = test_tr["tokenized_sentence"].apply(lambda x: predict(model_tr, tokenizer_tr, x, device))
    # Predicting entity indices for Italian test data
    test_it["indices"] = test_it["tokenized_sentence"].apply(lambda x: predict(model_it, tokenizer_it, x, device))

    final_df = pd.concat([test_tr, test_it]).sort_values(by="id")       # Combining both language results and sort by ID
    final_df[["id", "indices"]].to_csv(args.output_path, index=False)   # Saving only the ID and predicted indices to output CSV
    print(f"Predictions saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--model_dir_tr", type=str, required=True, help="Directory of saved Turkish model")
    parser.add_argument("--model_dir_it", type=str, required=True, help="Directory of saved Italian model")
    parser.add_argument("--model_name_tr", type=str, required=True, help="HuggingFace model name for Turkish")
    parser.add_argument("--model_name_it", type=str, required=True, help="HuggingFace model name for Italian")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save prediction CSV")
    args = parser.parse_args()

    main(args)
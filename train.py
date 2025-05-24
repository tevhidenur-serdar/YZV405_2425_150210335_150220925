import argparse
import os
import ast
import random
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset


# Seed function
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Converting entity indices into BIO labels for token classification
def create_bio_labels(row):
    tokens = row["tokenized_sentence"]
    indices = row["indices"]
    if isinstance(indices, str):        # Converting stringified list to Python list
        indices = ast.literal_eval(indices)     
    labels = ["O"] * len(tokens)        
    if indices != [-1]:         
        labels[indices[0]] = "B"
        for i in indices[1:]:
            labels[i] = "I"         # Remaining indices are "I"
    return tokens, labels


# Encoding tokens and labeling 
def encode_dataset(df, tokenizer, label2id):
    tokenized_inputs = []
    for tokens, labels in zip(df["tokens"], df["labels"]):

        # Tokenizing the sentence
        enc = tokenizer(tokens, is_split_into_words=True, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        word_ids = enc.word_ids(batch_index=0)

        # Creating label ids aligned with tokens 
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[labels[word_id]])

        # Converting tensors to plain lists
        enc = {k: v.squeeze().tolist() for k, v in enc.items()}
        enc["labels"] = label_ids
        tokenized_inputs.append(enc)
    return Dataset.from_list(tokenized_inputs)


# Training function
def train_model(model, tokenizer, train_data, val_data, output_dir, lang):
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer
    )
    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"model_{lang}.pt"))


# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--synthetic_path", type=str, required=True, help="Path to synthetic_data.csv")
    parser.add_argument("--val_path", type=str, required=True, help="Path to eval.csv")
    parser.add_argument("--model_dir_tr", type=str, required=True, help="Directory to save Turkish model")
    parser.add_argument("--model_dir_it", type=str, required=True, help="Directory to save Italian model")
    parser.add_argument("--model_name_tr", type=str, default="dbmdz/bert-base-turkish-cased", help="Huggingface model name for Turkish")
    parser.add_argument("--model_name_it", type=str, default="dbmdz/bert-base-italian-cased", help="Huggingface model name for Italian")
    args = parser.parse_args()

    set_seed(42)

    # Loading and merging real + synthetic training data
    train_df = pd.read_csv(args.train_path)
    synthetic_df = pd.read_csv(args.synthetic_path)
    df = pd.concat([train_df, synthetic_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # Converting stringified lists to actual lists
    df["indices"] = df["indices"].apply(ast.literal_eval)
    df[["tokenized_sentence"]] = df[["tokenized_sentence"]].applymap(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Generating BIO labels for training
    df[["tokens", "labels"]] = df.apply(create_bio_labels, axis=1, result_type="expand")

    label2id = {"O": 0, "B": 1, "I": 2}
    id2label = {v: k for k, v in label2id.items()}

    # Training a separate model for each language
    for lang, model_name, model_dir in zip(
        ["tr", "it"],
        [args.model_name_tr, args.model_name_it],
        [args.model_dir_tr, args.model_dir_it]
    ):
        # Filtering train and validation data by language
        df_lang = df[df["language"] == lang].reset_index(drop=True)
        val_df = pd.read_csv(args.val_path)
        val_df = val_df[val_df["language"] == lang].reset_index(drop=True)
        val_df[["tokenized_sentence"]] = val_df[["tokenized_sentence"]].applymap(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        val_df[["tokens", "labels"]] = val_df.apply(create_bio_labels, axis=1, result_type="expand")

        # Loading tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3, id2label=id2label, label2id=label2id)

        # Encoding datasets
        train_encoded = encode_dataset(df_lang, tokenizer, label2id)
        val_encoded = encode_dataset(val_df, tokenizer, label2id)

        train_model(model, tokenizer, train_encoded, val_encoded, model_dir, lang)
        print(f"{lang.upper()} model saved to {os.path.join(model_dir, f'model_{lang}.pt')}")
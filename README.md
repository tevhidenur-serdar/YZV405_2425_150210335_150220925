# Multiword Expression (MWE) Identification

This project addresses the identification of idiomatic expressions — a subclass of Multiword Expressions (MWEs) — in Turkish and Italian sentences. The task involves locating the token indices of idioms in tokenized sentences, or returning `[-1]` if no idiom is present.

## Project Structure

```
.
├── train.py                             # Training script
├── run.py                               # Synthetic data script
├── synthetic.py                         # Inference script
├── requirements.txt                     # Python dependencies
├── data/
│   ├── train.csv                        # Training dataset
│   ├── synthetic_data.csv               # Synthetic dataset
│   ├── external_synthetic_data.csv      # External Synthetic dataset
│   ├── eval.csv                         # Evaluation dataset
│   └── test_w_o_labels.csv              # Test dataset (no labels)
└── model/
    ├── model_tr.pt                      # Trained Turkish model weights (You can download it from drive link)
    └── model_it.pt                      # Trained Italian model weights (You can download it from drive link)
```

---

## How to Run

### 1. Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

---

### 2. Training

```bash
python train.py --train_path data/train.csv --synthetic_path data/synthetic_data.csv --val_path data/eval.csv --model_dir_tr model --model_dir_it model
```

This script:
- Loads and combines real and synthetic training data.
- Converts idiom indices to BIO format.
- Trains and saves two separate token classification models (Turkish & Italian) under `model` folder.

---

### 3. Prediction

```bash
python run.py --test_path data/test_w_o_labels.csv --model_dir_tr model --model_dir_it model --model_name_tr dbmdz/bert-base-turkish-cased --model_name_it dbmdz/bert-base-italian-cased --output_path prediction.csv
```

This script:
- Loads the trained models for each language.
- Applies them to tokenized sentences in `test.csv`.
- Outputs predictions to `prediction.csv` in the expected format.

---

### 4. Synthetic.py

```bash
python synthetic.py --eval_path data/eval.csv --output_path data/synthetic_data.csv
```

This script:
- Loads idioms and their language tags from eval.csv.
- For each idiom, generates two idiomatic and two literal example sentences using predefined templates.
- Saves all generated data into synthetic_data.csv in the expected format.

---

### Input and Output Format

### Input (test_w_o_labels.csv)

| id  | language | sentence | tokenized_sentence |
|-----|----------|----------|--------------------|
| 123 | tr       | ...      | ["Bir", "gün", ...] |

### Output (prediction.csv)
| id  | indices       |
|-----|---------------|
| 123 | [1, 2, 3]     |

- If an idiom is detected, `indices` will contain token positions (0-based).
- If no idiom is present, output `[-1]`.

---

## Model Weights

Pretrained model weights are stored in:

- `model/model_tr.pt` — Turkish model
- `model/model_it.pt` — Italian model

You **do not need to re-train** the models for inference, unless you modify the training data or model architecture.

You can download the models from:  
**https://drive.google.com/drive/folders/1jYo1nuZTsdenXTgu8-EeaJPu-Pa_YjpY?usp=drive_link**

---

## External Resources

- Token classification models:  
  - [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased)  
  - [`dbmdz/bert-base-italian-cased`](https://huggingface.co/dbmdz/bert-base-italian-cased)

These models are automatically downloaded by the Huggingface Transformers library.

---

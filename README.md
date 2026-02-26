# NLI Comparison on IndicXNLI (Hindi)

Compares three models on the **IndicXNLI Hindi** Natural Language Inference dataset.

**Task:** Given a premise and hypothesis in Hindi, classify the relationship as `Entailment`, `Neutral`, or `Contradiction`.

## Results

| Model | Accuracy | Macro F1 | Type |
|---|---|---|---|
| **XLM-RoBERTa** (`joeddav/xlm-roberta-large-xnli`) | **0.9699** | **0.9699** | Transformer (pre-trained on XNLI-15 languages) |
| Logistic Regression (TF-IDF) | 0.4846 | 0.4831 | Classical — trained on IndicXNLI train set |
| Linear SVM (TF-IDF) | 0.4603 | 0.4589 | Classical — trained on IndicXNLI train set |

**Dataset split sizes:** Train: 117,811 | Dev: 2,490 | Test: 5,010 (balanced, 1,670 per class)

---

## Setup

### 1. Clone the repo

```bash
git clone git@github.com:Mayankmalhotra772/nlp.git
cd nlp
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the **IndicXNLI** dataset and place it in the project root so the folder structure looks like:

```
IndicXNLI/
  train/xnli_hi.json
  dev/xnli_hi.json
  test/xnli_hi.json
```

Download from: https://huggingface.co/datasets/Divyanshu/indicxnli

Quick download using the `datasets` library:

```python
from datasets import load_dataset

for split in ['train', 'validation', 'test']:
    ds = load_dataset('Divyanshu/indicxnli', 'hi', split=split)
    ds.to_json(f'IndicXNLI/{"dev" if split == "validation" else split}/xnli_hi.json')
```

---

## Run the notebook

```bash
jupyter notebook nli_comparison.ipynb
```

Then run all cells top-to-bottom (**Kernel → Restart & Run All**).

---

## Requirements

See [requirements.txt](requirements.txt). Key packages:

- `torch` — GPU/CPU inference for XLM-RoBERTa
- `transformers` — HuggingFace model & tokenizer
- `scikit-learn` — Logistic Regression, SVM, TF-IDF
- `pandas`, `numpy`, `matplotlib`, `seaborn` — data handling & visualisation

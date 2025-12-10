# IHLT Final Project
## SemEval 2012 Task 6: Semantic Textual Similarity (STS)

**Course:** Introduction to Human Language Technology (IHLT)  
**Task:** Measure the degree of semantic equivalence between two sentences (0-5 scale).

## Project Overview
This project implements a Supervised Machine Learning approach to detect paraphrases and semantic similarity without using modern Deep Learning embeddings (BERT, GloVe, etc.).

Instead, we utilize a **"feature engineering"** approach combining:
* **Lexical Overlap:** Jaccard, Dice, Length differences.
* **Semantics:** WordNet Path Similarity (capturing synonyms).
* **Syntax:** spaCy Dependency Parsing (Subject/Object/Root alignment).
* **Domain Specifics:** Negation detection, Number matching, and Translation metrics (BLEU, LCS).

## Setup & Installation

The project is designed to be **self-setting**.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Resource Download:**
    You do not need to manually download NLTK data or spaCy models. Running the notebook or importing features.py will automatically check and download missing resources (like en_core_web_sm).

## Project Structure

```
## Project Structure

```text
IHLT_Final_Project/
│
├── train/                        # Training Data Folder
│   ├── STS.input.MSRpar.txt
│   ├── STS.gs.MSRpar.txt
│   └── ... 
│
├── test-gold/                    # Test Data Folder (Inputs + Gold)
│   ├── STS.input.MSRpar.txt
│   ├── STS.gs.MSRpar.txt
│   └── ... 
│
├── src/                          # Python Source Modules
│   ├── data_loader.py            # Handles loading training and test datasets
│   ├── eval_utils.py             # Pearson correlation calculation
│   ├── features.py               # Feature extraction (Lexical, WordNet, spaCy)
│   └── models.py                 # ML Pipelines (Ridge, SVR, Random Forest)
│
├── output/                       # (Generated) System Predictions
│   ├── STS.output.MSRpar.mySystem.txt
│   └── ... (Created after running the notebook)
│
├── sts-ZoeFinelli-OnatBitirgen.ipynb  # MAIN NOTEBOOK (Entry Point)
├── evaluate.sh                   # Evaluation Shell Script
├── correlation.pl                # Official Perl Evaluation Helper
├── requirements.txt              # Project dependencies
└── README.md                     # Project Documentation
```

* `sts-ZoeFinelli-OnatBitirgen.ipynb`: The main notebook. It trains the models, performs feature analysis, and generates predictions.

* `features.py`: The feature extraction engine (contains WordNet, spaCy, and Lexical logic).

* `models.py`: Contains the Machine Learning pipelines (Ridge, SVR, Random Forest).

* `data_loader.py`: Handles loading training and test datasets (including "surprise" datasets).

* `evaluate.sh`: A shell script that automates the official Perl evaluation script.

## How to Run

### Step 1: Train & Predict
Open **`sts-ZoeFinelli-OnatBitirgen.ipynb`** in Jupyter Lab or VS Code.

1. **Restart Kernel and Run All Cells.**
2. The notebook will:
   * Load data from `train/` and `test-gold/`.
   * Train Random Forest, SVR, and Ridge models.
   * Generate predictions for all 5 datasets (`MSRpar`, `MSRvid`, `SMTeuroparl`, `OnWN`, `SMTnews`).
   * Save output files named `STS.output.[dataset].mySystem.txt`.

### Step 2: Evaluate
Once the notebook finishes, run the evaluation script in your terminal to get the official Pearson correlations:

```bash
./evaluate.sh
```
## Performance

Our "Combined Model" (Ensemble) achieves the following Pearson Correlations ($r$):

| Dataset | Pearson ($r$) |
| :--- | :--- |
| **MSRvid** | `0.833` |
| **OnWN** | `0.663` |
| **MSRpar** | `0.661` |
| **SMTnews** | `0.434` |
| **SMTeuroparl**| `0.448` |

**Final Macro-Average Pearson:** `0.60794`

>*Note: MSRpar performance was significantly boosted by adding negation, number matching, and LCS features.*
# 🧠 BERT-Based Sentiment Classification

### ELEC0141 – Deep Learning for Natural Language Processing (2025)

This project implements a fine-tuned BERT model for binary sentiment classification on the IMDB movie review dataset. Developed as part of the ELEC0141 coursework at UCL, it investigates key architectural components of transformer-based models and demonstrates how transfer learning can be applied effectively to natural language understanding.

---

## 📁 Project Structure

- `main.py` – Entrypoint for training and testing
- `A/` – Core codebase
    - `data_process.py` – Preprocessing and dataset tokenization
    - `model.py` – BERT classifier implementations (manual + AutoModel)
    - `training.py` – Training loop and metric tracking
    - `testing.py` – Evaluation pipeline (classification report, confusion matrix)
    - `utils.py` – Reproducibility and logging utilities
- `B/` – Reserved for future extensions or experimental modules
- `Datasets/` – Raw and preprocessed dataset files
    - `IMDB_Dataset.csv` – Binary-labeled dataset
    - `Sentiment_Analysis.csv` – Multi-label emotion classification dataset
    - `test_dataset.pt` – Serialized test set
- `env/` – Environment and reproducibility files
    - `environment.yml` – Conda environment specification
    - `requirements.txt` – Python package dependencies
- `logs/` – Logs and model outputs
    - `metrics_log.csv` – Training/validation metric tracking
    - `test_classification_report.txt` – Final model performance
    - `checkpoints/` – Saved model weights at each epoch
    - `plots/` – Generated figures (loss, F1, confusion matrix)

---

## ⚙️ How to Run

1. **Set up the environment**

```bash
conda env create -f env/environment.yml
conda activate dlnlp
```

(Alternative) Use pip:

```
pip install -r env/requirements.txt
```

And then, start the traning by:

```
python main.py
```


All metrics and plots (loss, F1-curve, confusion matrix) are available under logs/plots/.

## 📚 Coursework Report

The complete project methodology, source code explanations, model architecture comparison, dataset exploration, and experimental reflections are documented in the report: Report_DLNLP_25_SN24070891.pdf

## 🔍 Key Highlights

Transformer architecture explained via \cite{vaswani_attention_2023}

BERT classification and loss logic analyzed from source \cite{devlin_bert_2019}

Custom DataLoader + Manual loss computation vs Hugging Face abstraction

Reflections on dataset quality and multi-class classification challenges

Evaluation based on F1, macro-avg, confusion matrix, and training dynamics

## 💡 Lessons Learned

Clean and balanced datasets are crucial for successful NLP training

Understanding model internals (loss computation, forward pass) enhances trust and debuggability

Fine-tuning pre-trained models with even small classifier heads can yield strong results

## 📌 Notes

Ensure that the datasets are placed in the Datasets/ directory before execution.

Use GPU for faster training. Confirm availability via torch.cuda.is_available().

```
\DLNLP_25_SN24070891
|   .gitignore
|   main.py
|   README.md
|
+---A
|       data_process.py
|       model.py
|       testing.py
|       training.py
|       utils.py
|       __init__.py
|
+---B
+---Datasets
|       IMDB_Dataset.csv
|       Sentiment_Analysis.csv
|       test_dataset.pt
|
+---env
|       environment.yml
|       requirements.txt
|
\---logs
    |   metrics_log.csv
    |   test_classification_report.txt
    |
    +---checkpoints
    |       classifier_epoch_1.pt
    |       classifier_epoch_10.pt
    |       classifier_epoch_2.pt
    |       classifier_epoch_3.pt
    |       classifier_epoch_4.pt
    |       classifier_epoch_5.pt
    |       classifier_epoch_6.pt
    |       classifier_epoch_7.pt
    |       classifier_epoch_8.pt
    |       classifier_epoch_9.pt
    |
    \---plots
            loss_curve.png
            metric_curves.png
            test_confusion_matrix.png
```

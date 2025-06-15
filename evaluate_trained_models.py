# evaluate_trained_models.py

import os
import logging
import torch
import pandas as pd
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, RobertaTokenizer
from model_evaluation import evaluate_model
from data_loader import load_and_split_data
from data_preprocessing import preprocess_data


# ΡΥΘΜΙΣΗ LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("evaluate_models.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# ΕΛΕΓΧΟΣ CUDA / CPU

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Χρήση συσκευής: {device}")


# ΕΝΤΟΠΙΣΜΟΣ ΑΠΟΘΗΚΕΥΜΕΝΩΝ ΜΟΝΤΕΛΩΝ

BERT_MODEL_PATH = "saved_models/bert"
ROBERTA_MODEL_PATH = "saved_models/roberta"

if not os.path.exists(BERT_MODEL_PATH) or not os.path.exists(ROBERTA_MODEL_PATH):
    logging.critical("Δεν βρέθηκαν τα εκπαιδευμένα μοντέλα! Βεβαιωθείτε ότι έχουν αποθηκευτεί σωστά.")
    exit(1)

logging.info("Βρέθηκαν τα αποθηκευμένα μοντέλα BERT & RoBERTa.")


# ΦΟΡΤΩΣΗ ΤΩΝ ΜΟΝΤΕΛΩΝ

try:
    logging.info("Φόρτωση BERT...")
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

    logging.info("Φόρτωση RoBERTa...")
    roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH).to(device)
    roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_PATH)

    logging.info("Η φόρτωση των μοντέλων ολοκληρώθηκε επιτυχώς.")
except Exception as e:
    logging.critical(f"Σφάλμα στη φόρτωση των εκπαιδευμένων μοντέλων: {str(e)}")
    exit(1)


# ΦΟΡΤΩΣΗ TEST DATASET

DATASET_FILENAME = "training.1600000.processed.noemoticon.csv"

if not os.path.exists(DATASET_FILENAME):
    logging.critical(f"Το dataset {DATASET_FILENAME} δεν βρέθηκε! Παρακαλώ βεβαιωθείτε ότι υπάρχει.")
    exit(1)

try:
    _, _, df_test = load_and_split_data(DATASET_FILENAME)
    df_test = preprocess_data(df_test)
    logging.info(f"Το test dataset φορτώθηκε επιτυχώς με {len(df_test)} δείγματα.")
except Exception as e:
    logging.critical(f"Σφάλμα κατά τη φόρτωση του test dataset: {str(e)}")
    exit(1)


# ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΩΝ

try:
    logging.info("[EVALUATION] Αξιολόγηση BERT στο test subset.")
    bert_metrics = evaluate_model(bert_model, bert_tokenizer, df_test, max_length=64)
    logging.info(f"[EVALUATION] BERT Metrics: {bert_metrics}")

    logging.info("[EVALUATION] Αξιολόγηση RoBERTa στο test subset.")
    roberta_metrics = evaluate_model(roberta_model, roberta_tokenizer, df_test, max_length=64)
    logging.info(f"[EVALUATION] RoBERTa Metrics: {roberta_metrics}")

    logging.info("Η αξιολόγηση ολοκληρώθηκε επιτυχώς!")
except Exception as e:
    logging.critical(f"Σφάλμα κατά την αξιολόγηση: {str(e)}")
    exit(1)

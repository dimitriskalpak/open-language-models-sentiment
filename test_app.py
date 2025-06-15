# test_app.py

# Αρχείο που εκτελεί μια μικρή δοκιμή (test) της ροής:
# - Φόρτωση και split δεδομένων
# - Προεπεξεργασία
# - Download προ-εκπαιδευμένων μοντέλων
# - Fine-tuning για 1 εποχή (BERT & RoBERTa)
# - Αποθήκευση μοντέλων
# - Φόρτωση αποθηκευμένων μοντέλων
# - Καταγραφή χρήσης μνήμης


import sys
import logging
import os
import torch
import psutil
import pandas as pd
from transformers import BertForSequenceClassification, RobertaForSequenceClassification

from data_loader import load_and_split_data
from data_preprocessing import preprocess_data
from model_downloader import download_pretrained_models
from bert_fine_tuning import train_bert
from roberta_fine_tuning import train_roberta
from model_evaluation import evaluate_model


# ΡΥΘΜΙΣΗ LOGGING

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "test_app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)


# ΣΥΝΑΡΤΗΣΗ ΕΛΕΓΧΟΥ ΜΝΗΜΗΣ

def log_memory_usage(stage):
    """
    Καταγράφει τη χρήση μνήμης RAM & GPU σε κάθε βασικό στάδιο (stage).
    """
    vram = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    vram_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
    ram = psutil.virtual_memory().used / (1024**3)  # RAM σε GB

    logging.info(
        f"[MEMORY] {stage}: "
        f"GPU Allocated: {vram:.2f}GB | "
        f"GPU Reserved: {vram_reserved:.2f}GB | "
        f"RAM Used: {ram:.2f}GB"
    )

def test_huggingface_app(csv_filename="training.1600000.processed.noemoticon.csv"):
    """
    Εκτελεί μια συνοπτική δοκιμή της εφαρμογής, 
    ελέγχοντας μεταξύ άλλων και τη χρήση μνήμης.
    """
    logging.info("=== Έναρξη Testing της Εφαρμογής ===")
    
    # 1. Φόρτωση + Διαχωρισμός Δεδομένων
    try:
        df_train, df_val, df_test = load_and_split_data(csv_filename)
        df_train = df_train.sample(n=min(3000, len(df_train)), random_state=42)
        df_val = df_val.sample(n=min(1000, len(df_val)), random_state=42)
        df_test = df_test.sample(n=min(1000, len(df_test)), random_state=42)
        logging.info(
            f"[TEST] Δεδομένα μετά το sampling -> "
            f"train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
        )
    except Exception as e:
        logging.critical(f"[TEST] Σφάλμα στη φόρτωση δεδομένων: {str(e)}")
        return

    # 2. Προεπεξεργασία
    try:
        df_train = preprocess_data(df_train)
        df_val = preprocess_data(df_val)
        df_test = preprocess_data(df_test)
    except Exception as e:
        logging.critical(f"[TEST] Σφάλμα στην προεπεξεργασία: {str(e)}")
        return

    log_memory_usage("Μετά την προεπεξεργασία")

    # 3. Λήψη Μοντέλων
    try:
        download_pretrained_models(["bert-base-uncased", "roberta-base"])
    except Exception as e:
        logging.critical(f"[TEST] Σφάλμα στη λήψη των μοντέλων: {str(e)}")
        return

    log_memory_usage("Μετά τη λήψη μοντέλων")

    # 4. Fine-Tuning (1 εποχή)
    try:
        logging.info("[TEST] Fine-tuning BERT (1 εποχή).")
        
        # train_bert τώρα επιστρέφει (bert_model, bert_tokenizer)
        bert_model, bert_tokenizer = train_bert(
            df_train, df_val, 
            epochs=1, 
            batch_size=16, 
            gradient_accumulation_steps=2
        )
        
        logging.info("[TEST] Fine-tuning RoBERTa (1 εποχή).")
        # train_roberta επιστρέφει (roberta_model_path, roberta_tokenizer),
        # αλλά εδώ κρατάμε μόνο το path, 
        # γιατί το αποθηκευμένο μοντέλο το φορτώνουμε στο βήμα 7.
        roberta_model_path, _ = train_roberta(
            df_train, df_val, 
            epochs=1, 
            batch_size=16, 
            gradient_accumulation_steps=2
        )
    except Exception as e:
        logging.critical(f"[TEST] Σφάλμα κατά το fine-tuning: {str(e)}")
        return

    log_memory_usage("Μετά το fine-tuning")

    # 5. Αποθήκευση Μοντέλων (BERT)
    try:
        os.makedirs("saved_models/bert", exist_ok=True)
        os.makedirs("saved_models/roberta", exist_ok=True)

        bert_model.save_pretrained("saved_models/bert")
        bert_tokenizer.save_pretrained("saved_models/bert")
    except Exception as e:
        logging.critical(f"[TEST] Σφάλμα στην αποθήκευση των μοντέλων: {str(e)}")
        return

    log_memory_usage("Μετά την αποθήκευση των μοντέλων")

    # 6. Αποδέσμευση μνήμης GPU
    del bert_model, bert_tokenizer
    torch.cuda.empty_cache()

    log_memory_usage("Μετά την αποδέσμευση μνήμης GPU")

    # 7. Φόρτωση αποθηκευμένων μοντέλων
    try:
        # Επαναφόρτωση του BERT
        bert_model = BertForSequenceClassification.from_pretrained("saved_models/bert")

        # Επαναφόρτωση του RoBERTa
        roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path)

        bert_model.to("cuda" if torch.cuda.is_available() else "cpu")
        roberta_model.to("cuda" if torch.cuda.is_available() else "cpu")

    except Exception as e:
        logging.critical(f"[TEST] Σφάλμα στη φόρτωση αποθηκευμένων μοντέλων: {str(e)}")
        return

    log_memory_usage("Μετά τη φόρτωση αποθηκευμένων μοντέλων")

    logging.info("=== Το testing ολοκληρώθηκε επιτυχώς! ===")

if __name__ == "__main__":
    test_huggingface_app()

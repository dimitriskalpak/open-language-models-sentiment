# main.py

# Κεντρικό αρχείο εκτέλεσης (entry point)
# για τη διαδικασία προετοιμασίας δεδομένων,



import sys
import os
import logging
import torch

from data_loader import load_and_split_data
from data_preprocessing import preprocess_data
from model_downloader import download_pretrained_models
from bert_fine_tuning import train_bert
from roberta_fine_tuning import train_roberta
from model_evaluation import evaluate_model
from results_analysis import plot_metrics
from transformers import RobertaForSequenceClassification, RobertaTokenizer


# ΡΥΘΜΙΣΗ LOGGING ΜΕ ΔΥΝΑΜΙΚΟ ΦΑΚΕΛΟ

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    logging.info("=== Έναρξη της εφαρμογής ===")

    torch.backends.cudnn.benchmark = True  # Βελτιστοποίηση CUDA

    csv_filename = "training.1600000.processed.noemoticon.csv"
    try:
        df_train, df_val, df_test = load_and_split_data(csv_filename)
    except Exception as e:
        logging.critical(f"Σφάλμα κατά τη φόρτωση/διαχωρισμό των δεδομένων: {str(e)}")
        return

    try:
        df_train = preprocess_data(df_train)
        df_val = preprocess_data(df_val)
        df_test = preprocess_data(df_test)
    except Exception as e:
        logging.critical(f"Σφάλμα στην προεπεξεργασία δεδομένων: {str(e)}")
        return

    # Επιλέγουμε 600k δείγματα για το training
    df_train = df_train.sample(n=600000, random_state=42)  # 600k δείγματα

    models_list = ["bert-base-uncased", "roberta-base"]
    try:
        download_pretrained_models(models_list)
    except Exception as e:
        logging.critical(f"Σφάλμα κατά τη λήψη των μοντέλων: {str(e)}")
        return

    
    # Fine-tuning BERT
    
    try:
        logging.info("=== Fine-tuning BERT ===")
        torch.cuda.empty_cache()  # Καθαρισμός μνήμης πριν ξεκινήσει η εκπαίδευση
        
        # Σημείωση: train_bert τώρα επιστρέφει (model, tokenizer)
        bert_model, bert_tokenizer = train_bert(
            df_train, 
            df_val, 
            epochs=3, 
            batch_size=8,  # batch_size=8
            max_length=64, 
            gradient_accumulation_steps=4  # gradient_accumulation_steps=4
        )

        # Αποθήκευση BERT
        bert_model_path = "saved_models/bert"
        os.makedirs(bert_model_path, exist_ok=True)
        bert_model.save_pretrained(bert_model_path)
        bert_tokenizer.save_pretrained(bert_model_path)
        logging.info(f"BERT αποθηκεύτηκε στο {bert_model_path}")

        # Αξιολόγηση BERT
        torch.cuda.empty_cache()
        logging.info("=== Αξιολόγηση BERT στο Test set ===")
        bert_metrics = evaluate_model(bert_model, bert_tokenizer, df_test)
        logging.info(f"BERT Metrics: {bert_metrics}")

        # Εκτύπωση metrics στην κονσόλα
        print("\n=== BERT ΑΠΟΤΕΛΕΣΜΑΤΑ ===")
        for metric, value in bert_metrics.items():
            print(f"{metric}: {value:.4f}")
            logging.info(f"BERT {metric}: {value:.4f}")

        # Καθαρισμός μνήμης
        del bert_model, bert_tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        logging.critical(f"Σφάλμα στο fine-tuning/αποθήκευση/αξιολόγηση του BERT: {str(e)}")
        return

    
    # Ερώτηση στον χρήστη αν θέλει να συνεχίσει
    
    proceed = input("\nΘέλετε να συνεχίσετε στο fine-tuning του RoBERTa; (y/n): ").strip().lower()
    if proceed != "y":
        logging.info("Ο χρήστης επέλεξε να τερματίσει την εκτέλεση μετά το BERT.")
        print("Η διαδικασία ολοκληρώθηκε μετά το BERT. Έξοδος...")
        return

    
    # Fine-tuning RoBERTa
    
    try:
        logging.info("=== Fine-tuning RoBERTa ===")
        torch.cuda.empty_cache()

        # train_roberta επιστρέφει το path (model_path, tokenizer)
        roberta_model_path, _ = train_roberta(
            df_train, 
            df_val, 
            epochs=3, 
            batch_size=8,  # batch_size=8
            max_length=64, 
            gradient_accumulation_steps=4  # gradient_accumulation_steps=4
        )

        # Επαναφόρτωση RoBERTa μετά το fine-tuning (εφόσον χρειάζεται μνήμη)
        roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path)
        roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path)

        logging.info(f"RoBERTa αποθηκεύτηκε στο {roberta_model_path}")

        # Αξιολόγηση RoBERTa
        torch.cuda.empty_cache()
        logging.info("=== Αξιολόγηση RoBERTa στο Test set ===")
        roberta_metrics = evaluate_model(roberta_model, roberta_tokenizer, df_test)
        logging.info(f"RoBERTa Metrics: {roberta_metrics}")

        print("\n=== RoBERTa ΑΠΟΤΕΛΕΣΜΑΤΑ ===")
        for metric, value in roberta_metrics.items():
            print(f"{metric}: {value:.4f}")
            logging.info(f"RoBERTa {metric}: {value:.4f}")

        del roberta_model, roberta_tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        logging.critical(f"Σφάλμα στο fine-tuning/αποθήκευση/αξιολόγηση του RoBERTa: {str(e)}")
        return

    
    # Δημιουργία Γραφημάτων
    
    try:
        logging.info("=== Δημιουργία γραφημάτων ===")
        plot_metrics(bert_metrics, title="BERT Αποτελέσματα Test")
        plot_metrics(roberta_metrics, title="RoBERTa Αποτελέσματα Test")
    except Exception as e:
        logging.error(f"Σφάλμα στη δημιουργία γραφημάτων: {str(e)}")

    logging.info("=== Τέλος της εφαρμογής ===")
    print("\nΗ διαδικασία ολοκληρώθηκε με επιτυχία!")

if __name__ == "__main__":
    main()

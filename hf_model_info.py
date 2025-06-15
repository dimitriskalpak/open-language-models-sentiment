import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
import time
import logging
from huggingface_hub import HfApi  # Εισαγωγή από το huggingface_hub

# Ρύθμιση logging για παρακολούθηση της διαδικασίας
logging.basicConfig(
    filename='model_download.log',  # Αποθήκευση των logs σε αρχείο
    level=logging.INFO,  # Το επίπεδο καταγραφής (INFO για κανονική εκτέλεση, ERROR για σφάλματα)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_progress_bar(total_size):
    """
    Εμφανίζει την πρόοδο κατά τη διάρκεια της λήψης αρχείων.
    :param total_size: Ολική μέγεθος των δεδομένων που κατεβαίνουν
    """
    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading model")
    start_time = time.time()

    def update_progress(downloaded_size):
        pbar.update(downloaded_size)
        elapsed_time = time.time() - start_time
        speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
        logging.info(f"Downloaded {downloaded_size}B of {total_size}B, Speed: {speed:.2f} B/s")
        print(f"Downloaded {downloaded_size}B of {total_size}B, Speed: {speed:.2f} B/s")
        return elapsed_time  # Χρόνος για να υπολογίσουμε τη συνολική διάρκεια

    return update_progress, pbar

def request_huggingface_token():
    """
    Ζητάει από τον χρήστη το Hugging Face token, αν είναι απαραίτητο.
    :return: Το Hugging Face token
    """
    token = input("Για να κατεβάσετε το μοντέλο από το Hugging Face, παρακαλώ εισάγετε το token σας (ή αφήστε κενό για κατέβασμα χωρίς token): ")
    if not token:
        logging.info("Κατέβασμα μοντέλου χωρίς Hugging Face token.")
    else:
        logging.info("Χρησιμοποιείται Hugging Face token για το κατέβασμα.")
    return token

def download_model(model_name, save_dir, token=None):
    """
    Κατεβάζει το μοντέλο από το Hugging Face με δυνατότητα χρήσης token.
    :param model_name: Όνομα του μοντέλου (π.χ. 'bert-base-uncased')
    :param save_dir: Φάκελος αποθήκευσης του μοντέλου
    :param token: Token από τον Hugging Face (αν χρειάζεται)
    """
    try:
        if token:
            api = HfApi()
            model_info = api.model_info(model_name, token=token)
        else:
            model_info = None
        logging.info(f"Αρχίζω το κατέβασμα του μοντέλου {model_name} στην διαδρομή {save_dir}...")
        model = BertForSequenceClassification.from_pretrained(model_name, cache_dir=save_dir)
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=save_dir)
        logging.info(f"Το μοντέλο {model_name} κατέβηκε και αποθηκεύτηκε στο {save_dir}.")
    except Exception as e:
        logging.error(f"Σφάλμα κατά την λήψη του μοντέλου: {str(e)}")

def select_model_and_download():
    """
    Επιλέγει το μοντέλο που θέλει ο χρήστης και το κατεβάζει.
    """
    model_name = input("Παρακαλώ εισάγετε το όνομα του μοντέλου (π.χ., bert-base-uncased, roberta-base): ").strip()
    save_dir = input("Παρακαλώ εισάγετε την διαδρομή για αποθήκευση του μοντέλου: ").strip()

    # Ζητάμε το Hugging Face token, αν είναι απαραίτητο
    token = request_huggingface_token()

    # Κατεβάζουμε το μοντέλο
    download_model(model_name, save_dir, token)

if __name__ == "__main__":
    select_model_and_download()

# model_downloader.py

import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

def download_pretrained_models(models_list):
    """
    Κατεβάζει (αν δεν υπάρχουν) τα προεπιλεγμένα μοντέλα από το Hugging Face.
    
    models_list: λίστα με ονόματα μοντέλων, ["bert-base-uncased", "roberta-base"].
    """
    for model_name in models_list:
        try:
            logging.info(f"Έλεγχος / Λήψη του μοντέλου {model_name}...")
            _ = AutoTokenizer.from_pretrained(model_name)
            _ = AutoModelForSequenceClassification.from_pretrained(model_name)
            logging.info(f"Το μοντέλο '{model_name}' έχει επιτυχώς εντοπιστεί / ληφθεί.")
        except (RepositoryNotFoundError, EntryNotFoundError) as not_found_err:
            error_msg = f"Δεν βρέθηκε το μοντέλο {model_name} στο Hugging Face: {str(not_found_err)}"
            logging.error(error_msg)
            raise not_found_err
        except Exception as e:
            error_msg = f"Σφάλμα κατά τη λήψη του μοντέλου {model_name}: {str(e)}"
            logging.error(error_msg)
            raise e

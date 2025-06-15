"""
predict_tweets.py

Script για την πρόβλεψη συναισθήματος σε 1.600.000 tweets, 
χρησιμοποιώντας ήδη εκπαιδευμένα (fine-tuned) μοντέλα BERT και RoBERTa.
Δεν πραγματοποιείται νέα εκπαίδευση, αλλά μόνο φόρτωση των μοντέλων, 
πρόβλεψη, αποθήκευση των αποτελεσμάτων και ανάλυση.

Στάδια:
1) Φόρτωση του μεγάλου dataset με 1.600.000 tweets
2) Φόρτωση των αποθηκευμένων μοντέλων BERT και RoBERTa
3) Πρόβλεψη συναισθήματος (θετικό/αρνητικό) για κάθε tweet
4) Αποθήκευση σε νέο CSV (tweet_predictions.csv) και ενδιάμεση αποθήκευση
5) Δημιουργία γραφημάτων (Confusion Matrix & διαγράμματα κατανομής)

Εκτέλεση:
    python predict_tweets.py
"""

import sys
import logging
import os
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Από τη βιβλιοθήκη transformers
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification


# Ρύθμιση advanced logging: Καταγραφή σε αρχείο και στην κονσόλα

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("prediction_results.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

def predict_sentiment(text, model, tokenizer, device):
    """
    Συνάρτηση που δέχεται ένα κείμενο (tweet) και επιστρέφει την πρόβλεψη
    συναισθήματος ως 'Θετικό' ή 'Αρνητικό'. 
    Χρησιμοποιεί το μοντέλο και τον tokenizer που έχουν ήδη φορτωθεί.

    Στάδια:
      1) Τοποθέτηση του μοντέλου σε κατάσταση αξιολόγησης (model.eval()).
      2) Tokenization του κειμένου (με truncation=True, max_length=64).
      3) Μεταφορά των δεδομένων στη συσκευή (CPU/GPU).
      4) Προβλέψεις (logits) χωρίς backpropagation (with torch.no_grad()).
      5) Λογική argmax για εντοπισμό κατηγορίας: 0=Αρνητικό, 1=Θετικό.
    """
    try:
        model.eval()
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=64
        )
        # Μεταφορά στη συσκευή
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        prediction_id = torch.argmax(logits, dim=-1).item()
        return "Θετικό" if prediction_id == 1 else "Αρνητικό"

    except Exception as e:
        logging.error(f"[PREDICTION ERROR] Κείμενο: '{text[:50]}...' | Σφάλμα: {str(e)}")
        return "Άγνωστο"


def main():
    """
    Η κεντρική συνάρτηση που φορτώνει το dataset, τα μοντέλα, παράγει τις
    προβλέψεις για κάθε tweet, αποθηκεύει τα αποτελέσματα και παράγει γραφήματα
    για την ανάλυση των τελικών προβλέψεων.

    Έλεγχος σφαλμάτων σε κάθε στάδιο με logging σε επίπεδα INFO, ERROR, CRITICAL.
    """
    logging.info("=== Έναρξη Διαδικασίας Πρόβλεψης ===")

    csv_filename = "training.1600000.processed.noemoticon.csv"
    
    # Ο φάκελος του τρέχοντος script, για να αποθηκεύονται δυναμικά τα γραφήματα
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) Φόρτωση του dataset
    try:
        df = pd.read_csv(csv_filename, encoding="ISO-8859-1", header=None)
        df.columns = ["Sentiment", "ID", "Date", "Query", "User", "Text"]
        logging.info(f"Το dataset φορτώθηκε επιτυχώς με σχήμα: {df.shape}")
    except FileNotFoundError:
        logging.critical(f"[CRITICAL] Δεν βρέθηκε το αρχείο {csv_filename}. Τερματισμός.")
        return
    except Exception as e:
        logging.critical(f"[CRITICAL] Αποτυχία φόρτωσης των δεδομένων: {str(e)}")
        return

    # 2) Φόρτωση αποθηκευμένων μοντέλων
    try:
        logging.info("Φόρτωση εκπαιδευμένου BERT από 'bert_fine_tuned/'...")
        bert_model = BertForSequenceClassification.from_pretrained("bert_fine_tuned")
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        logging.info("Φόρτωση εκπαιδευμένου RoBERTa από 'roberta_fine_tuned/'...")
        roberta_model = RobertaForSequenceClassification.from_pretrained("roberta_fine_tuned")
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    except Exception as e:
        logging.critical(f"[CRITICAL] Αδυναμία φόρτωσης των εκπαιδευμένων μοντέλων: {str(e)}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Χρήση συσκευής: {device}")
    bert_model.to(device)
    roberta_model.to(device)

    logging.info("Ξεκινά η ταξινόμηση συναισθήματος σε όλο το dataset.")

    results = []  # Λίστα που θα περιέχει [Original, Text, BERT, RoBERTa]
    total_tweets = len(df)

    # 3) Πρόβλεψη συναισθήματος για κάθε tweet
    for i, row in df.iterrows():
        tweet_text = str(row["Text"])
        original_label = row["Sentiment"]  # 0 ή 4 στο αρχικό dataset (Sentiment140)

        # Πρόβλεψη με BERT
        bert_pred = predict_sentiment(tweet_text, bert_model, bert_tokenizer, device)
        # Πρόβλεψη με RoBERTa
        roberta_pred = predict_sentiment(tweet_text, roberta_model, roberta_tokenizer, device)

        # Καταγραφή (προαιρετικά ανά tweet ή κάθε x tweets)
        logging.info(f"[{i+1}/{total_tweets}] BERT: {bert_pred}, RoBERTa: {roberta_pred} | Text: {tweet_text[:60]}...")
        
        results.append([original_label, tweet_text, bert_pred, roberta_pred])

        # Ενδιάμεση αποθήκευση των προβλέψεων κάθε 50.000 tweets
        if (i + 1) % 50000 == 0:
            partial_df = pd.DataFrame(
                results,
                columns=["Original Sentiment", "Tweet", "BERT Prediction", "RoBERTa Prediction"]
            )
            partial_filename = f"tweet_predictions_partial_{i+1}.csv"
            partial_df.to_csv(partial_filename, index=False, encoding="utf-8")
            logging.info(f"[INFO] Αποθηκεύτηκαν προσωρινά {i+1} προβλέψεις ({partial_filename}).")

    logging.info("Ολοκληρώθηκε η ταξινόμηση όλων των tweets. Προχωράμε σε αποθήκευση τελικού αρχείου.")

    # 4) Αποθήκευση αποτελεσμάτων σε CSV
    try:
        final_df = pd.DataFrame(
            results,
            columns=["Original Sentiment", "Tweet", "BERT Prediction", "RoBERTa Prediction"]
        )
        final_df.to_csv("tweet_predictions.csv", index=False, encoding="utf-8")
        logging.info("Οι τελικές προβλέψεις αποθηκεύτηκαν στο tweet_predictions.csv.")
    except Exception as e:
        logging.error(f"[ERROR] Αποτυχία αποθήκευσης του τελικού αποτελέσματος: {str(e)}")
        return
    
    # 5) Δημιουργία γραφημάτων
    try:
        logging.info("Ξεκινά η δημιουργία γραφημάτων για την ανάλυση των αποτελεσμάτων...")

        # Μετατρέπουμε την Original Sentiment: 0->Αρνητικό, 4->Θετικό
        # και τις BERT/Roberta σε 0/1 για τη δημιουργία Confusion Matrix
        final_df["Original_Sent"] = final_df["Original Sentiment"].apply(lambda x: 0 if x==0 else 1)
        final_df["BERT_Sent"] = final_df["BERT Prediction"].apply(lambda x: 1 if x=="Θετικό" else 0)
        final_df["RoBERTa_Sent"] = final_df["RoBERT Prediction"].apply(lambda x: 1 if x=="Θετικό" else 0)

        # Confusion Matrix για BERT
        cm_bert = confusion_matrix(final_df["Original_Sent"], final_df["BERT_Sent"])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_bert, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - BERT")
        plt.xlabel("Πρόβλεψη BERT")
        plt.ylabel("Αρχική Ετικέτα")
        plt.savefig(os.path.join(script_dir, "bert_confusion_matrix.png"))
        plt.show()
        plt.close()
        logging.info("Αποθηκεύτηκε το διάγραμμα Confusion Matrix για BERT (bert_confusion_matrix.png).")

        # Confusion Matrix για RoBERTa
        cm_roberta = confusion_matrix(final_df["Original_Sent"], final_df["RoBERTa_Sent"])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_roberta, annot=True, fmt="d", cmap="Greens")
        plt.title("Confusion Matrix - RoBERTa")
        plt.xlabel("Πρόβλεψη RoBERTa")
        plt.ylabel("Αρχική Ετικέτα")
        plt.savefig(os.path.join(script_dir, "roberta_confusion_matrix.png"))
        plt.show()
        plt.close()
        logging.info("Αποθηκεύτηκε το διάγραμμα Confusion Matrix για RoBERTa (roberta_confusion_matrix.png).")

        # Απλή ανάλυση κατανομής
        # (πόσα tweets προέβλεψε θετικά vs αρνητικά το κάθε μοντέλο)
        plt.figure(figsize=(6,4))
        bert_counts = final_df["BERT Prediction"].value_counts()
        sns.barplot(x=bert_counts.index, y=bert_counts.values, palette="Blues")
        plt.title("BERT - Κατανομή Θετικών/Αρνητικών Προβλέψεων")
        plt.ylabel("Αριθμός Tweets")
        plt.savefig(os.path.join(script_dir, "bert_distribution.png"))
        plt.show()
        plt.close()

        roberta_counts = final_df["RoBERT Prediction"].value_counts()
        plt.figure(figsize=(6,4))
        sns.barplot(x=roberta_counts.index, y=roberta_counts.values, palette="Greens")
        plt.title("RoBERTa - Κατανομή Θετικών/Αρνητικών Προβλέψεων")
        plt.ylabel("Αριθμός Tweets")
        plt.savefig(os.path.join(script_dir, "roberta_distribution.png"))
        plt.show()
        plt.close()

        logging.info("Ολοκληρώθηκε η δημιουργία γραφημάτων κατανομής προβλέψεων.")
    except Exception as e:
        logging.error(f"[ERROR] Αποτυχία δημιουργίας/αποθήκευσης γραφημάτων: {str(e)}")

    logging.info("=== Η διαδικασία ολοκληρώθηκε επιτυχώς! ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"[CRITICAL] Απρόβλεπτο σφάλμα εκτέλεσης: {str(e)}")

# model_evaluation.py

# Αρχείο που περιλαμβάνει τη συνάρτηση αξιολόγησης ενός μοντέλου 
# σε δεδομένα test/validation, με υπολογισμό 
# (accuracy, precision, recall, f1).


import torch
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

def evaluate_model(model, tokenizer, df, max_length=64, batch_size=64):
    """
    Συνάρτηση αξιολόγησης του μοντέλου σε ένα DataFrame (df).
    Υπολογίζει τις μετρήσεις αξιολόγησης accuracy, precision, recall, f1 
    και τις επιστρέφει σε ένα λεξικό.

    :param model: Το (fine-tuned) μοντέλο ταξινόμησης.
    :param tokenizer: Ο tokenizer που αντιστοιχεί στο μοντέλο.
    :param df: pandas DataFrame με στήλες ['Tweet', 'Sentiment'].
    :param max_length: Μέγιστο μήκος sequence για tokenization (int).
    :param batch_size: Μέγεθος batch κατά την αξιολόγηση (int).
    :return: dict -> {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
    """
    logging.info("Ξεκινά η αξιολόγηση του μοντέλου στο δοθέν σετ.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = df['Tweet'].tolist()
    labels = df['Sentiment'].tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"]
    attention_masks = encodings["attention_mask"]
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_preds = []
    all_true = []

    # Χρήση torch.inference_mode() για καλύτερη απόδοση
    with torch.inference_mode():
        for batch in data_loader:
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device, non_blocking=True) for b in batch]

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = batch_labels.cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(true_labels)

    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, average='binary')
    recall = recall_score(all_true, all_preds, average='binary')
    f1 = f1_score(all_true, all_preds, average='binary')

    logging.info(
        f"Αποτελέσματα Αξιολόγησης -> "
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    # Καθαρισμός μνήμης
    del input_ids, attention_masks, labels_tensor, dataset, data_loader
    torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# bert_fine_tuning.py

# Σε αυτό το αρχείο υλοποιείται το fine-tuning του BERT 


import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

def train_bert(
    df_train,
    df_val,
    epochs=3,
    batch_size=8,  
    max_length=64,
    lr=2e-5,
    warmup_ratio=0.1,
    early_stopping_patience=2,
    gradient_accumulation_steps=4 
):
    """
    Πραγματοποιεί fine-tuning του BERT (bert-base-uncased) για ανάλυση συναισθήματος.
    Περιλαμβάνει Mixed Precision Training (FP16), Gradient Accumulation και Early Stopping.
    
    :param df_train: DataFrame με στήλες ['Tweet', 'Sentiment'] για το training set.
    :param df_val: DataFrame με στήλες ['Tweet', 'Sentiment'] για το validation set.
    :param epochs: Αριθμός εποχών εκπαίδευσης (int).
    :param batch_size: Μέγεθος batch (int).
    :param max_length: Μέγιστο μήκος sequence (int).
    :param lr: Learning rate (float).
    :param warmup_ratio: Ποσοστό warmup steps (float).
    :param early_stopping_patience: Υπομονή χωρίς βελτίωση στο val accuracy 
                                    πριν ενεργοποιηθεί το early stopping (int).
    :param gradient_accumulation_steps: Αθροιστική ενημέρωση βαθμίδων (int).
    :return: (model, tokenizer) - το fine-tuned BERT μοντέλο και ο tokenizer του.
    """

    logging.info("Ξεκινά το fine-tuning του BERT...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Χρήση συσκευής: {device}")

    # Προσαρμογή batch size με βάση τη διαθέσιμη VRAM
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # VRAM σε GB
        if vram < 8:
            batch_size = min(batch_size, 8)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    #  Προετοιμασία δεδομένων 
    def encode_texts(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        dataset = TensorDataset(
            encodings['input_ids'], 
            encodings['attention_mask'], 
            torch.tensor(labels)
        )
        return dataset

    train_dataset = encode_texts(df_train['Tweet'].tolist(), df_train['Sentiment'].tolist())
    val_dataset = encode_texts(df_val['Tweet'].tolist(), df_val['Sentiment'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    #  Ορισμός optimizer και scheduler 
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler()  # Mixed Precision Training

    best_val_accuracy = 0.0
    no_improve_epochs = 0
    best_model_state = None

    #  Βρόχος εκπαίδευσης (train + validation) 
    for epoch in range(epochs):
        logging.info(f"=== Εποχή {epoch+1}/{epochs} ===")
        model.train()
        
        total_train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
            
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )
                loss = outputs.loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_train_loss += loss.item() * gradient_accumulation_steps
        
        avg_train_loss = total_train_loss / len(train_loader)
        logging.info(f"Μέσο Training Loss: {avg_train_loss:.4f}")

        #  Validation 
        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())

        val_accuracy = accuracy_score(val_true, val_preds)
        logging.info(f"Validation Accuracy: {val_accuracy:.4f}")

        #  Early Stopping Check 
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logging.info(f"Καμία βελτίωση. no_improve_epochs={no_improve_epochs}")
            if no_improve_epochs >= early_stopping_patience:
                logging.info("Early stopping λόγω μη βελτίωσης.")
                break

    # Επαναφόρτωση της καλύτερης κατάστασης (αν υπάρχει)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    logging.info("Η εκπαίδευση του BERT ολοκληρώθηκε.")
    return model, tokenizer

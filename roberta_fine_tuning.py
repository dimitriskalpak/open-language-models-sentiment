# roberta_fine_tuning.py

# Σε αυτό το αρχείο υλοποιείται το fine-tuning του RoBERTa 
# (π.χ. "roberta-base") για ανάλυση συναισθήματος.


import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

def train_roberta(
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
    Fine-tuning του RoBERTa (roberta-base) για ταξινόμηση (π.χ. Sentiment Analysis).

    - Χρησιμοποιεί Mixed Precision Training (FP16).
    - Gradient Accumulation Steps για οικονομία VRAM.
    - Αυτόματη προσαρμογή batch size για GPU <8GB.

    Επιστρέφει ένα tuple: (model_path, tokenizer)
      όπου model_path είναι το path όπου σώζεται το fine-tuned μοντέλο.
    """

    logging.info("Ξεκινά το fine-tuning του RoBERTa...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Χρήση συσκευής: {device}")

    # Προσαρμογή batch size αν η GPU έχει <8GB VRAM
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram < 8:
            batch_size = min(batch_size, 8)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.to(device)

    # Προετοιμασία δεδομένων 
    def encode_texts(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        dataset = TensorDataset(
            encodings["input_ids"], 
            encodings["attention_mask"], 
            torch.tensor(labels)
        )
        return dataset

    train_dataset = encode_texts(df_train["Tweet"].tolist(), df_train["Sentiment"].tolist())
    val_dataset = encode_texts(df_val["Tweet"].tolist(), df_val["Sentiment"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Ορισμός optimizer & scheduler 
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler()
    best_val_accuracy = 0.0
    no_improve_epochs = 0
    best_model_state = None

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

    logging.info("Η εκπαίδευση του RoBERTa ολοκληρώθηκε.")

    # Αποθήκευση του μοντέλου σε τοπικό φάκελο 
    model_path = "saved_models/roberta"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logging.info(f"Το fine-tuned RoBERTa αποθηκεύτηκε στο: {model_path}")

    # Αποδέσμευση μνήμης
    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    
    return model_path, tokenizer

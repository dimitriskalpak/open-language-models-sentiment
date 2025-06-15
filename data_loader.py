# data_loader.py

import os
import logging
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_and_split_data(
    csv_filename,
    column_names=['Sentiment', 'ID', 'Timestamp', 'Query', 'User', 'Tweet'],
    test_size=0.1,
    val_size=0.1,
    random_state=42
):
    """
    Συνάρτηση που φορτώνει το dataset από το CSV αρχείο, ορίζει ονόματα στηλών
    (εφόσον δεν υπάρχουν), και το διαχωρίζει σε train/validation/test.
    """
    logging.info("Φόρτωση του dataset και διαχωρισμός σε train/validation/test σετ.")
    
    if not os.path.exists(csv_filename):
        error_msg = f"Το αρχείο {csv_filename} δεν βρέθηκε."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        df = pd.read_csv(csv_filename, names=column_names, header=None, encoding='latin-1')
    except Exception as e:
        error_msg = f"Σφάλμα κατά τη φόρτωση του CSV αρχείου: {str(e)}"
        logging.error(error_msg)
        raise e
    
    logging.info(f"Το dataset φορτώθηκε με σχήμα: {df.shape}")
    
    # Φιλτράρουμε μόνο τις γραμμές που έχουν Sentiment 0 ή 4
    df = df[(df['Sentiment'] == 0) | (df['Sentiment'] == 4)]
    df['Sentiment'] = df['Sentiment'].replace(4, 1)
    
    # Διαχωρισμός σε train/test
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Sentiment'])
    
    # Διαχωρισμός του train σε train/validation
    relative_val_size = val_size / (1 - test_size)
    df_train, df_val = train_test_split(df_train, test_size=relative_val_size, random_state=random_state, stratify=df_train['Sentiment'])
    
    logging.info(f"Train set: {df_train.shape}, Validation set: {df_val.shape}, Test set: {df_test.shape}")
    
    del df  # Αποδέσμευση μνήμης για το αρχικό dataset
    torch.cuda.empty_cache()

    return df_train, df_val, df_test

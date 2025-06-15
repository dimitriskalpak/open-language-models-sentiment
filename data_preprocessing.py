# data_preprocessing.py

import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm


nltk.download('stopwords')

def preprocess_data(df):
    """
    Συνάρτηση που εκτελεί προεπεξεργασία των tweets:
    - Μετατροπή σε πεζά
    - Αφαίρεση URL, mentions, hashtags, punctuation
    - Αφαίρεση stopwords
    - Stemming
    
    Επιστρέφει το DataFrame με την καθαρισμένη στήλη 'Tweet'.
    """
    logging.info("Ξεκινά η προεπεξεργασία των δεδομένων (καθαρισμός, tokenization, κ.λπ.).")
    
    # Αρχικοποίηση του PorterStemmer και των stopwords
    stemmer = PorterStemmer()
    english_stopwords = set(stopwords.words('english'))
    
    # Χρησιμοποιούμε list comprehension για αποδοτικότητα και αναγνωσιμότητα
    def clean_tweet(tweet):
        tweet = tweet.lower()  # Σε πεζά
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)  # Αφαίρεση URLs
        tweet = re.sub(r'@\w+|#\w+', '', tweet)  # Αφαίρεση mentions και hashtags
        tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)  # Αφαίρεση punctuation / ειδικών χαρακτήρων
        tokens = tweet.split()  # Tokenization
        tokens = [t for t in tokens if t not in english_stopwords]  # Αφαίρεση stopwords
        tokens = [stemmer.stem(t) for t in tokens]  # Stemming
        return " ".join(tokens)  # Ενοποίηση

    # Χρησιμοποιούμε την apply για καλύτερη αναγνωσιμότητα και αποδοτικότητα
    df['Tweet'] = df['Tweet'].apply(clean_tweet)
    
    logging.info("Ο καθαρισμός και η προεπεξεργασία ολοκληρώθηκαν.")
    return df

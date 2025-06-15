import os
import pandas as pd
import logging

# Ρύθμιση logging για παρακολούθηση σε αρχείο
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_file(file_name):
    """
    Ψάχνει το αρχείο στον τρέχοντα φάκελο και επιστρέφει τη διαδρομή του.
    :param file_name: Όνομα του αρχείου
    :return: Διαδρομή του αρχείου αν βρεθεί, αλλιώς None
    """
    try:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, file_name)

        if os.path.exists(file_path):
            print(f"Αρχείο {file_name} βρέθηκε: {file_path}")
            logging.info(f"Αρχείο {file_name} βρέθηκε: {file_path}")
            return file_path
        else:
            print(f"Το αρχείο {file_name} ΔΕΝ βρέθηκε στον φάκελο {current_directory}.")
            logging.error(f"Το αρχείο {file_name} δεν βρέθηκε στον φάκελο {current_directory}")
            return None
    except Exception as e:
        print(f"Σφάλμα κατά την αναζήτηση του αρχείου: {str(e)}")
        logging.error(f"Σφάλμα κατά την αναζήτηση του αρχείου: {str(e)}")
        return None

def load_and_process_data(file_path):
    """
    Φορτώνει το dataset και προσθέτει τα κατάλληλα ονόματα στηλών.
    :param file_path: Η διαδρομή του αρχείου CSV
    :return: DataFrame με τα δεδομένα, ή None αν προκύψει σφάλμα
    """
    try:
        print(f"Φορτώνω το αρχείο: {file_path}...")
        # Φορτώνουμε το dataset με 'ISO-8859-1' για να χειριστούμε καλύτερα τις κωδικοποιήσεις
        df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1', dtype={0: 'int32', 1: 'str'})
        print("Τα δεδομένα φορτώθηκαν επιτυχώς.")
        logging.info("Τα δεδομένα φορτώθηκαν επιτυχώς.")
        
        # Προσθήκη ονομάτων στηλών (αν δεν υπάρχουν)
        df.columns = ['Sentiment', 'ID', 'Timestamp', 'Query', 'User', 'Tweet']
        print("Τα ονόματα των στηλών προστέθηκαν.")
        logging.info("Τα ονόματα των στηλών προστέθηκαν.")
        
        return df
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση ή επεξεργασία του dataset: {str(e)}")
        logging.error(f"Σφάλμα κατά τη φόρτωση ή επεξεργασία του dataset: {str(e)}")
        return None

def save_processed_data(df, output_file_name):
    """
    Αποθηκεύει τα επεξεργασμένα δεδομένα σε νέο αρχείο CSV.
    :param df: DataFrame με τα επεξεργασμένα δεδομένα
    :param output_file_name: Όνομα του νέου αρχείου
    """
    try:
        output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file_name)
        df.to_csv(output_file_path, index=False)  # Μην αποθηκεύεις τη στήλη index
        print(f"Τα δεδομένα αποθηκεύτηκαν στο αρχείο {output_file_name}.")
        logging.info(f"Τα δεδομένα αποθηκεύτηκαν στο αρχείο {output_file_name}.")
    except Exception as e:
        print(f"Σφάλμα κατά την αποθήκευση των δεδομένων: {str(e)}")
        logging.error(f"Σφάλμα κατά την αποθήκευση των δεδομένων: {str(e)}")

def verify_data_integrity(original_df, processed_df):
    """
    Επαληθεύει ότι το αρχικό και το επεξεργασμένο dataset είναι ταυτόσημα.
    :param original_df: Το αρχικό dataset
    :param processed_df: Το επεξεργασμένο dataset
    :return: True αν είναι ταυτόσημα, αλλιώς False
    """
    try:
        print("Επαληθεύω την ταυτότητα των δεδομένων...")

        # Αφαιρούμε το index για τη σύγκριση (έτσι δεν επηρεάζει)
        original_df = original_df.reset_index(drop=True)
        processed_df = processed_df.reset_index(drop=True)
        
        # Ελέγχουμε αν τα δεδομένα είναι ταυτόσημα (ιδία περιεχόμενα και τάξη)
        if original_df.shape == processed_df.shape and original_df.equals(processed_df):
            print("Τα δεδομένα είναι ταυτόσημα.")
            logging.info("Τα δεδομένα είναι ταυτόσημα.")
            return True
        else:
            print("Τα δεδομένα ΔΕΝ είναι ταυτόσημα.")
            logging.warning("Τα δεδομένα ΔΕΝ είναι ταυτόσημα.")
            
            # Εκτύπωση διαφορών
            diff = original_df.compare(processed_df)
            print("Διαφορές μεταξύ των datasets:")
            print(diff)
            logging.info("Διαφορές μεταξύ των datasets:")
            logging.info(f"{diff}")
            return False
    except Exception as e:
        print(f"Σφάλμα κατά τη σύγκριση των δεδομένων: {str(e)}")
        logging.error(f"Σφάλμα κατά τη σύγκριση των δεδομένων: {str(e)}")
        return False

def main():
    file_name = "training.1600000.processed.noemoticon.csv"
    
    # Βρίσκουμε το αρχείο CSV στον ίδιο φάκελο
    file_path = find_file(file_name)
    if file_path is None:
        print("Δεν βρέθηκε το αρχείο για επεξεργασία. Τερματισμός.")
        logging.error("Δεν βρέθηκε το αρχείο για επεξεργασία. Τερματισμός.")
        return
    
    # Φορτώνουμε και επεξεργαζόμαστε το αρχικό dataset
    original_df = load_and_process_data(file_path)
    if original_df is None:
        print("Η φόρτωση του αρχικού dataset απέτυχε.")
        logging.error("Η φόρτωση του αρχικού dataset απέτυχε.")
        return
    
    # Αποθηκεύουμε τα επεξεργασμένα δεδομένα με νέο όνομα
    save_processed_data(original_df, "processed_sentiment140.csv")
    
    # Φορτώνουμε και επεξεργαζόμαστε ξανά το dataset για τη σύγκριση
    processed_df = load_and_process_data("processed_sentiment140.csv")
    if processed_df is None:
        print("Η φόρτωση του επεξεργασμένου dataset απέτυχε.")
        logging.error("Η φόρτωση του επεξεργασμένου dataset απέτυχε.")
        return
    
    # Επαλήθευση αν τα datasets είναι ταυτόσημα
    if not verify_data_integrity(original_df, processed_df):
        print("Τα datasets ΔΕΝ είναι ταυτόσημα!")
        logging.error("Τα datasets ΔΕΝ είναι ταυτόσημα!")
    else:
        print("Η επεξεργασία ολοκληρώθηκε με επιτυχία και τα datasets είναι ταυτόσημα.")
        logging.info("Η επεξεργασία ολοκληρώθηκε με επιτυχία και τα datasets είναι ταυτόσημα.")

if __name__ == "__main__":
    main()

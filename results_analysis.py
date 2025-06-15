# results_analysis.py

import matplotlib.pyplot as plt
import logging

def plot_metrics(metrics_dict, title="Αποτελέσματα Αξιολόγησης"):
    """
    Δημιουργεί ένα bar chart για τις μετρήσεις που περιέχονται στο metrics_dict.
    """
    logging.info("Δημιουργία γραφήματος για τις μετρήσεις αξιολόγησης.")
    
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, values, color=['blue', 'green', 'orange', 'red'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height * 1.01,
            f"{height:.2f}",
            ha='center',
            va='bottom'
        )
    
    plt.ylim([0, 1.2])
    plt.title(title)
    plt.ylabel("Τιμή Μέτρησης αξιολόγησης (0 έως 1)")
    plt.xlabel("Μετρήσεις αξιολόγησης")
    plt.show()

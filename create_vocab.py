import os
import pickle
import json
from collections import Counter

def build_vocab(dataset_path, vocab_path, min_freq=1):
    """
    Erstelle eine Vokabular-Datei basierend auf den Log-Daten.

    :param dataset_path: Pfad zur Log-Datei (z. B. hdfs.log).
    :param vocab_path: Pfad, wo die Vokabular-Datei gespeichert werden soll.
    :param min_freq: Minimale Häufigkeit eines Tokens, um ins Vokabular aufgenommen zu werden.
    """
    counter = Counter()

    # Tokens aus den Log-Daten zählen
    with open(dataset_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)
            
    # Neue Methode, Tokens aus der Log Datei zählen: JSON-basierte Log-Datei verarbeiten -> try except um ungültige JSon Zeilen abbzufangen😀️
     with open(dataset_path, 'r') as f:
         for line in f:
             try:
                 log_entry = json.loads(line.strip())  # Jede Zeile als JSON parsen
                 
                 # Relevante Felder extrahieren
                 severity = log_entry.get('severity', '')  # z. B. "DEBUG", "ERROR"
                 reporter = log_entry.get('reporter', '')  # z. B. Modulnamen
                 msg = log_entry.get('msg', '')            # Nachrichtentext
                 
                 # Tokenisierung für das Feld "msg"
                 msg_tokens = msg.split()  # Tokens im "msg"-Feld
                 
                 # Zähle Tokens
                 counter.update([severity, reporter])  # Füge severity & reporter hinzu
                 counter.update(msg_tokens)           # Füge die Tokens aus msg hinzu
             except json.JSONDecodeError:
                 print(f"Fehler beim Parsen der Zeile: {line}")

    # Tokens mit IDs speichern
    vocab = {token: idx for idx, (token, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab["<PAD>"] = len(vocab)
    vocab["<UNK>"] = len(vocab)

    # Vokabular-Datei speichern
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print(f"Vokabular mit {len(vocab)} Tokens erstellt und in {vocab_path} gespeichert.")

# Beispielaufruf
build_vocab('./datasets/hdfs.log', './datasets/HDFS_vocab.pkl')
build_vocab('./datasets/mt_controller_log', './datasets/MtController_vocab.pkl')😀️

import pickle
import json

# Lade das aktuelle Vokabular
vocab_path = './datasets/HDFS_vocab.pkl'   
vocab_path = './datasets/Mt_controller_vocab.pkl'😀️
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# Erstelle `stoi` (String-to-Index) und `itos` (Index-to-String)
stoi = vocab

# Neue Tokens aus einer JSON-Log-Datei extrahieren😀️
dataset_path = './datasets/MtController.log'  # Pfad zu deiner Log-Datei
with open(dataset_path, 'r') as f:
    for line in f:
        try:
            # Lade die JSON-Daten
           log_entry = json.loads(line.strip())
            
            # Relevante Felder extrahieren
            severity = log_entry.get('severity', '')  # Feld `severity`
            reporter = log_entry.get('reporter', '')  # Feld `reporter`
            msg = log_entry.get('msg', '')  # Feld `msg`
            
            # Tokens aus `severity` und `reporter` hinzufügen
            if severity not in stoi:
                stoi[severity] = len(stoi)
            if reporter not in stoi:
                stoi[reporter] = len(stoi)
            
            # Tokens aus `msg` hinzufügen
            msg_tokens = msg.split()  # Nachricht in Tokens zerlegen
            for token in msg_tokens:
                if token not in stoi:
                    stoi[token] = len(stoi)
        except json.JSONDecodeError:
            print(f"Fehler beim Parsen der Zeile: {line}")

itos = {index: token for token, index in vocab.items()}

# Speichere das neue Vokabular-Objekt mit `stoi` und `itos`
new_vocab = {"stoi": stoi, "itos": itos}

with open(vocab_path, 'wb') as f:
    pickle.dump(new_vocab, f)

print(f"Vokabular-Datei mit `stoi` und `itos` aktualisiert: {vocab_path}")

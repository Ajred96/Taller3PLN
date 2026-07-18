"""
classification_report por clase (N/NEU/P) de los 3 modelos TASS publicados en HF.
Solo inferencia sobre el conjunto de test local. Corre en CPU/MPS (Mac M2).

Uso:
    python src/predict/tass/classification_report_tass.py

Salida: imprime + guarda en outputs/reports/tass_classification_report.txt
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[3]
TEST_JSON = ROOT / "data" / "processed" / "tass" / "test.json"
OUT = ROOT / "outputs" / "reports" / "tass_classification_report.txt"

LABELS = ["N", "NEU", "P"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

REPOS = {
    "BETO":  "gustavoa6791/beto-tass-sentiment-univalle-pln",
    "XLNet": "gustavoa6791/xlnet-tass-sentiment-univalle-pln",
    "XLM-R": "gustavoa6791/xlmroberta-tass-sentiment-univalle-pln",
}

# MPS (GPU integrada M2) si está; si no CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Dispositivo: {device}")

# Cargar test
rows = json.load(open(TEST_JSON, encoding="utf-8"))
texts = [r["text"] for r in rows]
y_true = [LABEL2ID[r["label"]] for r in rows]
print(f"Test: {len(texts)} ejemplos")


def predict(repo, batch_size=32):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, truncation=True, max_length=128,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            print(f"  {min(i+batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return preds


salida = []
for nombre, repo in REPOS.items():
    print(f"\n===== {nombre} ({repo}) =====")
    preds = predict(repo)
    rep = classification_report(y_true, preds, target_names=LABELS, digits=4)
    bloque = f"\n===== {nombre} ({repo}) =====\n{rep}"
    print(rep)
    salida.append(bloque)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(salida), encoding="utf-8")
print(f"\nGuardado en: {OUT}")

"""
Inferencia interactiva del clasificador TASS (3 clases: N=Negativo, NEU=Neutro, P=Positivo).

Uso:
    python src/predict/tass/predict_tass.py                      # usa BETO bs32 por defecto
    MODEL=outputs/models/tass/xlmroberta_bs16/best python ...    # otro modelo
"""
import os
from transformers import pipeline

MODEL_PATH = os.environ.get("MODEL", "./outputs/models/tass/beto_bs32/best")

EMOJI = {"N": "🔴 Negativo", "NEU": "🟡 Neutro", "P": "🟢 Positivo"}

print(f"Cargando modelo TASS desde: {MODEL_PATH}")
try:
    classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)
    print("¡Modelo cargado!")
    print("=" * 64)
    print("Escribe una frase para clasificar su sentimiento (N / NEU / P).")
    print("(Escribe 'salir' para terminar)")
    print("=" * 64, "\n")
except Exception as e:
    print(f"Error al cargar el modelo desde '{MODEL_PATH}'. Detalle: {e}")
    raise SystemExit(1)

while True:
    entrada = input("Tu oración > ").strip()
    if entrada.lower() == "salir":
        print("\n¡Hasta luego!")
        break
    if not entrada:
        continue

    pred = classifier(entrada)[0]
    etiqueta = pred["label"]  # id2label ya devuelve N / NEU / P
    legible = EMOJI.get(etiqueta, etiqueta)
    print(f"Resultado: {legible} | Confianza: {pred['score']*100:.2f}%\n")

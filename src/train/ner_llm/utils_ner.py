"""
utils_ner.py
============
Funciones compartidas para el fine-tuning de modelos NER
sobre el dataset de próstata (formato BIO).

Usado por train_beto_ner.py y train_xlmroberta_ner.py
"""

import json
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# ── Etiquetas del dataset próstata ───────────────────────────────────────────

LABEL_LIST = [
    "O",
    "B-CANCER", "I-CANCER",
    "B-FECHA", "I-FECHA",
    "B-BIOMARCADOR", "I-BIOMARCADOR",
    "B-GLEASON", "I-GLEASON",
    "B-DOSIS", "I-DOSIS",
    "B-TRATAMIENTO", "I-TRATAMIENTO",
    "B-MEDICAMENTO", "I-MEDICAMENTO",
    "B-TNM", "I-TNM",
    "B-EDAD", "I-EDAD",
    "B-CIRUGIA", "I-CIRUGIA",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_json(path: Path) -> list[dict]:
    """Lee un archivo JSON con lista de {tokens, labels}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_dataset(data_dir: Path) -> DatasetDict:
    """
    Carga los tres splits del dataset próstata desde JSON
    y los convierte a DatasetDict de HuggingFace.

    Args:
        data_dir: ruta a data/processed/prostata_json/

    Returns:
        DatasetDict con splits train, validation, test
    """
    splits = {}
    for split in ["train", "validation", "test"]:
        ruta = data_dir / f"{split}.json"
        datos = cargar_json(ruta)
        splits[split] = Dataset.from_list(datos)
        print(f"✔ {split}: {len(datos)} oraciones")

    return DatasetDict(splits)


# ── Tokenización y alineamiento de etiquetas ─────────────────────────────────

def tokenizar_y_alinear(ejemplos: dict, tokenizer, max_length: int = 512) -> dict:
    """
    Tokeniza las oraciones y alinea las etiquetas NER con los subtokens.

    El problema: BETO/XLM-RoBERTa dividen palabras en subtokens.
    Por ejemplo "adenocarcinoma" → ["adeno", "##car", "##cinoma"].
    Solo el primer subtoken de cada palabra hereda la etiqueta original;
    los demás reciben -100 para que el loss los ignore.

    Args:
        ejemplos: batch con campos "tokens" y "labels"
        tokenizer: tokenizador del modelo
        max_length: longitud máxima de secuencia

    Returns:
        dict con input_ids, attention_mask y labels alineados
    """
    tokenized = tokenizer(
        ejemplos["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,   # entrada ya tokenizada en palabras
        padding=False,              # padding dinámico en DataCollator
    )

    labels_alineadas = []
    for i, labels in enumerate(ejemplos["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # token especial [CLS] o [SEP]
                label_ids.append(-100)
            elif word_id != prev_word_id:
                # primer subtoken de la palabra → hereda la etiqueta
                etiqueta = labels[word_id]
                label_ids.append(LABEL2ID.get(etiqueta, 0))
            else:
                # subtoken adicional → ignorado en el loss
                label_ids.append(-100)
            prev_word_id = word_id

        labels_alineadas.append(label_ids)

    tokenized["labels"] = labels_alineadas
    return tokenized


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    """
    Calcula F1, precisión y recall usando seqeval (evaluación a nivel de entidad).

    seqeval ignora automáticamente los -100 y evalúa secuencias BIO completas,
    no token a token, que es el estándar para NER.
    """
    logits, labels = eval_pred
    predicciones = np.argmax(logits, axis=-1)

    # Convertir IDs a etiquetas, ignorando -100
    pred_labels = [
        [ID2LABEL[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predicciones, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for l in label_row if l != -100]
        for label_row in labels
    ]

    return {
        "f1":        f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall":    recall_score(true_labels, pred_labels),
    }


def reporte_completo(pred_labels: list[list[str]], true_labels: list[list[str]]) -> str:
    """Genera el reporte de clasificación completo por tipo de entidad."""
    return classification_report(true_labels, pred_labels, digits=4)


# ── Inspección rápida del dataset ─────────────────────────────────────────────

def mostrar_estadisticas(dataset: DatasetDict) -> None:
    """Imprime estadísticas básicas del dataset cargado."""
    print("\n" + "=" * 55)
    print("  ESTADÍSTICAS DEL DATASET PRÓSTATA")
    print("=" * 55)

    from collections import Counter

    for split_name, split_data in dataset.items():
        total_tokens = sum(len(e["tokens"]) for e in split_data)
        counter = Counter()
        for e in split_data:
            counter.update(e["labels"])

        print(f"\n[{split_name.upper()}]")
        print(f"  Oraciones : {len(split_data)}")
        print(f"  Tokens    : {total_tokens}")
        print(f"  Distribución de etiquetas:")
        for label, count in sorted(counter.items()):
            if label != "O":
                print(f"    {label:<25} {count}")

    print("=" * 55)

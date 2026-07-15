import json
from collections import Counter
from pathlib import Path

from src.config.config import PROCESSED_DIR, REPORTS_DIR


DATASETS = {
    "tass": {
        "type": "Clasificación multiclase",
        "task": "Análisis de sentimientos",
        "splits": ["train", "test"],
        "description": "Dataset TASS en español con tres clases de sentimiento: negativo, neutro y positivo.",
    },
    "sarcasmo": {
        "type": "Clasificación binaria",
        "task": "Detección de sarcasmo",
        "splits": ["train", "validation"],
        "description": "Dataset en español para clasificación binaria entre sarcasmo y no sarcasmo.",
    },
    "conll2002": {
        "type": "NER",
        "task": "Reconocimiento de entidades nombradas",
        "splits": ["train", "validation", "test"],
        "description": "Dataset CoNLL2002 en español con entidades PER, LOC, ORG y MISC en formato BIO.",
    },
    "prostata_json": {
        "card_name": "prostata",
        "type": "NER biomédico",
        "task": "Extracción de entidades clínicas relacionadas con cáncer de próstata",
        "splits": ["train", "validation", "test"],
        "description": "Dataset clínico anotado en formato BIO con entidades biomédicas como CANCER, FECHA, BIOMARCADOR, GLEASON, DOSIS, TRATAMIENTO, MEDICAMENTO, TNM, EDAD y CIRUGIA.",
    },
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def count_classification_labels(data):
    return Counter(item["label"] for item in data)


def count_ner_labels(data):
    counter = Counter()

    for item in data:
        counter.update(item["labels"])

    return counter


def get_example(data):
    if not data:
        return {}

    return data[0]


def format_counter(counter: Counter):
    lines = []

    for label, count in counter.most_common():
        lines.append(f"| {label} | {count} |")

    return "\n".join(lines)


def export_card(dataset_key: str, config: dict):
    dataset_dir = PROCESSED_DIR / dataset_key
    card_name = config.get("card_name", dataset_key)
    output_path = REPORTS_DIR / f"dataset_card_{card_name}.md"

    total_records = 0
    global_counter = Counter()
    split_summaries = []
    examples = {}

    for split in config["splits"]:
        file_path = dataset_dir / f"{split}.json"

        if not file_path.exists():
            split_summaries.append(
                {
                    "split": split,
                    "records": 0,
                    "status": "No encontrado",
                }
            )
            continue

        data = load_json(file_path)
        total_records += len(data)

        if config["type"].startswith("Clasificación"):
            label_counter = count_classification_labels(data)
        else:
            label_counter = count_ner_labels(data)

        global_counter.update(label_counter)

        split_summaries.append(
            {
                "split": split,
                "records": len(data),
                "status": "OK",
                "labels": label_counter,
            }
        )

        examples[split] = get_example(data)

    with open(output_path, "w", encoding="utf-8") as card:
        card.write(f"# Dataset Card: {card_name.upper()}\n\n")
        card.write(f"## Descripción\n\n")
        card.write(f"{config['description']}\n\n")

        card.write("## Tarea\n\n")
        card.write(f"- Tipo: {config['type']}\n")
        card.write(f"- Objetivo: {config['task']}\n")
        card.write(f"- Formato procesado: JSON\n\n")

        card.write("## Splits disponibles\n\n")
        card.write("| Split | Registros | Estado |\n")
        card.write("|---|---:|---|\n")

        for summary in split_summaries:
            card.write(
                f"| {summary['split']} | {summary['records']} | {summary['status']} |\n"
            )

        card.write(f"\nTotal de registros: **{total_records}**\n\n")

        card.write("## Distribución global de etiquetas\n\n")
        card.write("| Etiqueta | Cantidad |\n")
        card.write("|---|---:|\n")
        card.write(format_counter(global_counter))
        card.write("\n\n")

        card.write("## Ejemplos\n\n")

        for split, example in examples.items():
            card.write(f"### {split}\n\n")
            card.write("```json\n")
            card.write(json.dumps(example, ensure_ascii=False, indent=2))
            card.write("\n```\n\n")

        card.write("## Observaciones de preprocesamiento\n\n")

        if card_name == "prostata":
            card.write(
                "- Se conservaron los archivos originales en `data/raw/prostata`.\n"
            )
            card.write(
                "- Se aplicaron correcciones puntuales sobre anotaciones inconsistentes detectadas durante la validación BIO.\n"
            )
            card.write(
                "- Los archivos corregidos fueron exportados a `data/processed/prostata` y convertidos posteriormente a JSON en `data/processed/prostata_json`.\n"
            )
        elif card_name == "conll2002":
            card.write(
                "- El dataset se conservó sin modificaciones, ya que las anomalías BIO detectadas fueron mínimas y corresponden a diferencias de anotación del corpus.\n"
            )
        else:
            card.write(
                "- El dataset fue normalizado a una estructura común con campos `text` y `label`.\n"
            )

    print(f"Dataset card creada: {output_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_key, config in DATASETS.items():
        export_card(dataset_key, config)


if __name__ == "__main__":
    main()
import json
from pathlib import Path

import pandas as pd

from src.config.config import (
    TASS_DIR,
    SARCASMO_DIR,
    CONLL2002_DIR,
    PROCESSED_DIR,
)


PROCESSED_PROSTATA_DIR = PROCESSED_DIR / "prostata"


def save_json(data, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Archivo creado: {output_path}")


def read_csv_safely(path: Path) -> pd.DataFrame:
    for sep in [",", ";"]:
        for encoding in ["utf-8", "latin1", "cp1252"]:
            try:
                return pd.read_csv(path, encoding=encoding, sep=sep)
            except Exception:
                continue

    raise ValueError(f"No se pudo leer el archivo CSV: {path}")


def export_classification_csv(
        input_path: Path,
        output_path: Path,
        text_column: str,
        label_column: str,
):
    df = read_csv_safely(input_path)

    data = []

    for _, row in df.iterrows():
        text = str(row[text_column]).strip()
        label = str(row[label_column]).strip()

        if text and label:
            data.append(
                {
                    "text": text,
                    "label": label,
                }
            )

    save_json(data, output_path)


def read_bio_sentences(path: Path):
    sentences = []
    tokens = []
    labels = []

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line:
                if tokens:
                    sentences.append(
                        {
                            "tokens": tokens,
                            "labels": labels,
                        }
                    )
                    tokens = []
                    labels = []
                continue

            parts = line.split()

            if len(parts) >= 2:
                token = parts[0]
                label = parts[-1]

                tokens.append(token)
                labels.append(label)

        if tokens:
            sentences.append(
                {
                    "tokens": tokens,
                    "labels": labels,
                }
            )

    return sentences


def export_bio_file(input_path: Path, output_path: Path):
    data = read_bio_sentences(input_path)
    save_json(data, output_path)


def export_tass():
    output_dir = PROCESSED_DIR / "tass"

    export_classification_csv(
        input_path=TASS_DIR / "cr-tass.csv",
        output_path=output_dir / "train.json",
        text_column="sentencia original",
        label_column="label",
    )

    export_classification_csv(
        input_path=TASS_DIR / "cr1-tass.csv",
        output_path=output_dir / "test.json",
        text_column="sentencia original",
        label_column="label",
    )


def export_sarcasmo():
    output_dir = PROCESSED_DIR / "sarcasmo"

    export_classification_csv(
        input_path=SARCASMO_DIR / "train.csv",
        output_path=output_dir / "train.json",
        text_column="Texto",
        label_column="Sarcasmo",
    )

    export_classification_csv(
        input_path=SARCASMO_DIR / "validation.csv",
        output_path=output_dir / "validation.json",
        text_column="Texto",
        label_column="Sarcasmo",
    )


def export_conll2002():
    output_dir = PROCESSED_DIR / "conll2002"

    export_bio_file(
        input_path=CONLL2002_DIR / "train.txt",
        output_path=output_dir / "train.json",
    )

    export_bio_file(
        input_path=CONLL2002_DIR / "valid.txt",
        output_path=output_dir / "validation.json",
    )

    export_bio_file(
        input_path=CONLL2002_DIR / "test.txt",
        output_path=output_dir / "test.json",
    )


def export_prostata():
    output_dir = PROCESSED_DIR / "prostata_json"

    export_bio_file(
        input_path=PROCESSED_PROSTATA_DIR / "training.bio",
        output_path=output_dir / "train.json",
    )

    export_bio_file(
        input_path=PROCESSED_PROSTATA_DIR / "validation_cleaned.bio",
        output_path=output_dir / "validation.json",
    )

    export_bio_file(
        input_path=PROCESSED_PROSTATA_DIR / "testing_cleaned.bio",
        output_path=output_dir / "test.json",
    )


def main():
    export_tass()
    export_sarcasmo()
    export_conll2002()
    export_prostata()


if __name__ == "__main__":
    main()
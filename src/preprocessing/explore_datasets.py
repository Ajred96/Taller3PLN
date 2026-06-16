from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

from src.config.config import (
    TASS_DIR,
    SARCASMO_DIR,
    CONLL2002_DIR,
    PROSTATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)


def ensure_output_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_safely(path: Path) -> pd.DataFrame:
    for sep in [",", ";"]:
        for encoding in ["utf-8", "latin1", "cp1252"]:
            try:
                return pd.read_csv(path, encoding=encoding, sep=sep)
            except Exception:
                continue

    raise ValueError(f"No se pudo leer el archivo CSV: {path}")


def save_label_plot(title: str, counts: pd.Series, output_name: str):
    if counts.empty:
        return

    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Etiqueta")
    plt.ylabel("Cantidad")
    plt.tight_layout()

    output_path = FIGURES_DIR / output_name
    plt.savefig(output_path)
    plt.close()

    print(f"Gráfica creada: {output_path}")


def detect_label_column(df: pd.DataFrame):
    possible_label_columns = [
        "label",
        "labels",
        "sentiment",
        "sentimiento",
        "polarity",
        "polaridad",
        "etiqueta",
        "class",
        "clase",
        "sarcasmo",
    ]

    for column in df.columns:
        if column.lower() in possible_label_columns:
            return column

    return None


def explore_tabular_dataset(dataset_name: str, dataset_dir: Path):
    report_path = REPORTS_DIR / f"{dataset_name.lower()}_report.txt"

    with open(report_path, "w", encoding="utf-8") as report:
        report.write(f"=== REPORTE DATASET {dataset_name.upper()} ===\n\n")
        report.write(f"Ruta: {dataset_dir}\n\n")

        files = list(dataset_dir.glob("*"))

        if not files:
            report.write("No se encontraron archivos.\n")
            print(f"Sin archivos: {dataset_name}")
            return

        global_rows = 0
        global_label_counts = Counter()

        for file in files:
            if file.suffix.lower() not in [".csv", ".xlsx", ".xls"]:
                continue

            report.write(f"\n--- Archivo: {file.name} ---\n")

            try:
                if file.suffix.lower() == ".csv":
                    df = read_csv_safely(file)
                else:
                    df = pd.read_excel(file)

                global_rows += df.shape[0]

                report.write(f"Filas: {df.shape[0]}\n")
                report.write(f"Columnas: {df.shape[1]}\n")
                report.write(f"Nombres de columnas: {list(df.columns)}\n\n")

                report.write("Primeras 5 filas:\n")
                report.write(df.head().to_string())
                report.write("\n\n")

                label_column = detect_label_column(df)

                if label_column:
                    counts = df[label_column].value_counts(dropna=False)
                    global_label_counts.update(df[label_column].dropna().tolist())

                    report.write(f"Distribución de etiquetas usando columna '{label_column}':\n")
                    report.write(counts.to_string())
                    report.write("\n")

                    save_label_plot(
                        title=f"Distribución de etiquetas - {dataset_name} - {file.stem}",
                        counts=counts,
                        output_name=f"{dataset_name.lower()}_{file.stem}_labels.png",
                    )
                else:
                    report.write("No se detectó automáticamente una columna de etiquetas.\n")

            except Exception as e:
                report.write(f"Error leyendo archivo: {e}\n")

        report.write("\n=== RESUMEN GLOBAL ===\n")
        report.write(f"Total de filas leídas: {global_rows}\n")

        if global_label_counts:
            global_counts_series = pd.Series(global_label_counts).sort_values(ascending=False)
            report.write("Distribución global de etiquetas:\n")
            report.write(global_counts_series.to_string())
            report.write("\n")

            save_label_plot(
                title=f"Distribución global de etiquetas - {dataset_name}",
                counts=global_counts_series,
                output_name=f"{dataset_name.lower()}_global_labels.png",
            )

    print(f"Reporte creado: {report_path}")


def read_bio_file(path: Path):
    sentences = []
    current_sentence = []

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            parts = line.split()

            if len(parts) >= 2:
                token = parts[0]
                label = parts[-1]
                current_sentence.append((token, label))

        if current_sentence:
            sentences.append(current_sentence)

    return sentences


def explore_bio_dataset(dataset_name: str, dataset_dir: Path):
    report_path = REPORTS_DIR / f"{dataset_name.lower()}_report.txt"

    with open(report_path, "w", encoding="utf-8") as report:
        report.write(f"=== REPORTE DATASET {dataset_name.upper()} ===\n\n")
        report.write(f"Ruta: {dataset_dir}\n\n")

        bio_files = list(dataset_dir.glob("*.bio")) + list(dataset_dir.glob("*.txt"))

        if not bio_files:
            report.write("No se encontraron archivos BIO o TXT.\n")
            print(f"Sin archivos BIO/TXT: {dataset_name}")
            return

        total_sentences = 0
        total_tokens = 0
        global_labels = Counter()

        for file in bio_files:
            sentences = read_bio_file(file)

            label_counter = Counter()
            token_count = 0

            for sentence in sentences:
                token_count += len(sentence)
                for _, label in sentence:
                    label_counter[label] += 1
                    global_labels[label] += 1

            total_sentences += len(sentences)
            total_tokens += token_count

            counts_series = pd.Series(label_counter).sort_values(ascending=False)

            report.write(f"\n--- Archivo: {file.name} ---\n")
            report.write(f"Oraciones: {len(sentences)}\n")
            report.write(f"Tokens: {token_count}\n")
            report.write("Distribución de etiquetas:\n")
            report.write(counts_series.to_string())
            report.write("\n\n")

            report.write("Ejemplo primera oración:\n")
            if sentences:
                report.write(str(sentences[0][:20]))
            else:
                report.write("Sin oraciones detectadas.")
            report.write("\n")

            save_label_plot(
                title=f"Distribución de etiquetas - {dataset_name} - {file.stem}",
                counts=counts_series,
                output_name=f"{dataset_name.lower()}_{file.stem}_labels.png",
            )

        global_counts_series = pd.Series(global_labels).sort_values(ascending=False)

        report.write("\n=== RESUMEN GLOBAL ===\n")
        report.write(f"Total oraciones: {total_sentences}\n")
        report.write(f"Total tokens: {total_tokens}\n")
        report.write("Distribución global de etiquetas:\n")
        report.write(global_counts_series.to_string())
        report.write("\n")

        save_label_plot(
            title=f"Distribución global de etiquetas - {dataset_name}",
            counts=global_counts_series,
            output_name=f"{dataset_name.lower()}_global_labels.png",
        )

    print(f"Reporte creado: {report_path}")


def create_general_summary():
    summary_path = REPORTS_DIR / "datasets_summary.txt"

    report_files = [
        "tass_report.txt",
        "sarcasmo_report.txt",
        "conll2002_report.txt",
        "prostata_report.txt",
    ]

    with open(summary_path, "w", encoding="utf-8") as summary:
        summary.write("=== RESUMEN GENERAL DE DATASETS ===\n\n")

        for report_file in report_files:
            path = REPORTS_DIR / report_file

            summary.write(f"\n--- {report_file} ---\n")

            if not path.exists():
                summary.write("Reporte no encontrado.\n")
                continue

            with open(path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            for line in lines:
                if (
                        "Filas:" in line
                        or "Oraciones:" in line
                        or "Tokens:" in line
                        or "Total" in line
                        or "Nombres de columnas:" in line
                ):
                    summary.write(line)

    print(f"Resumen general creado: {summary_path}")


def main():
    ensure_output_dirs()

    explore_tabular_dataset("TASS", TASS_DIR)
    explore_tabular_dataset("SARCASMO", SARCASMO_DIR)
    explore_bio_dataset("CONLL2002", CONLL2002_DIR)
    explore_bio_dataset("PROSTATA", PROSTATA_DIR)

    create_general_summary()


if __name__ == "__main__":
    main()
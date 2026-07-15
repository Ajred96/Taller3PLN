from pathlib import Path
from collections import Counter

from src.config.config import CONLL2002_DIR, PROSTATA_DIR, REPORTS_DIR


VALID_PREFIXES = {"B", "I", "O"}


def validate_bio_file(path: Path):
    invalid_lines = []
    invalid_labels = Counter()
    transition_errors = []

    previous_label = "O"
    line_number = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line_number += 1
            line = raw_line.strip()

            if not line:
                previous_label = "O"
                continue

            parts = line.split()

            if len(parts) < 2:
                invalid_lines.append((line_number, line))
                previous_label = "O"
                continue

            token = parts[0]
            label = parts[-1]

            if label == "O":
                previous_label = label
                continue

            if "-" not in label:
                invalid_labels[label] += 1
                previous_label = label
                continue

            prefix, entity = label.split("-", 1)

            if prefix not in VALID_PREFIXES:
                invalid_labels[label] += 1

            if prefix == "I":
                if previous_label == "O":
                    transition_errors.append(
                        (line_number, token, label, previous_label, "I después de O")
                    )
                elif "-" in previous_label:
                    previous_prefix, previous_entity = previous_label.split("-", 1)
                    if previous_entity != entity:
                        transition_errors.append(
                            (
                                line_number,
                                token,
                                label,
                                previous_label,
                                "I con entidad diferente",
                            )
                        )

            previous_label = label

    return invalid_lines, invalid_labels, transition_errors


def validate_dataset(dataset_name: str, dataset_dir: Path):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / f"{dataset_name.lower()}_bio_validation.txt"
    files = list(dataset_dir.glob("*.bio")) + list(dataset_dir.glob("*.txt"))

    with open(report_path, "w", encoding="utf-8") as report:
        report.write(f"=== VALIDACIÓN BIO: {dataset_name.upper()} ===\n\n")
        report.write(f"Ruta: {dataset_dir}\n\n")

        if not files:
            report.write("No se encontraron archivos BIO/TXT.\n")
            print(f"Sin archivos para validar: {dataset_name}")
            return

        for file in files:
            invalid_lines, invalid_labels, transition_errors = validate_bio_file(file)

            report.write(f"\n--- Archivo: {file.name} ---\n")
            report.write(f"Líneas inválidas: {len(invalid_lines)}\n")
            report.write(f"Etiquetas inválidas: {sum(invalid_labels.values())}\n")
            report.write(f"Errores de transición BIO: {len(transition_errors)}\n\n")

            if invalid_labels:
                report.write("Detalle etiquetas inválidas:\n")
                for label, count in invalid_labels.items():
                    report.write(f"{label}: {count}\n")
                report.write("\n")

            if invalid_lines:
                report.write("Primeras líneas inválidas:\n")
                for line_number, line in invalid_lines[:20]:
                    report.write(f"Línea {line_number}: {line}\n")
                report.write("\n")

            if transition_errors:
                report.write("Primeros errores de transición BIO:\n")
                for error in transition_errors[:20]:
                    line_number, token, label, previous_label, reason = error
                    report.write(
                        f"Línea {line_number}: token='{token}', "
                        f"label='{label}', anterior='{previous_label}', razón='{reason}'\n"
                    )
                report.write("\n")

    print(f"Reporte creado: {report_path}")


def main():
    validate_dataset("CONLL2002", CONLL2002_DIR)
    validate_dataset("PROSTATA", PROSTATA_DIR)


if __name__ == "__main__":
    main()
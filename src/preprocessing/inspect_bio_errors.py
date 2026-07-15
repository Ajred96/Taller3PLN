from pathlib import Path

from src.config.config import CONLL2002_DIR, PROSTATA_DIR, REPORTS_DIR


ERRORS_TO_INSPECT = {
    "CONLL2002": [
        ("test.txt", 9291),
        ("train.txt", 221619),
        ("valid.txt", 30882),
    ],
    "PROSTATA": [
        ("testing_cleaned.bio", 4977),
        ("testing_cleaned.bio", 12471),
        ("training.bio", 1773),
        ("training.bio", 15496),
    ],
}


INVALID_LABELS_TO_FIND = {
    "PROSTATA": [
        ("testing_cleaned.bio", "0"),
        ("validation_cleaned.bio", ","),
    ]
}


def read_lines(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.readlines()


def extract_context(lines, target_line, window=5):
    start = max(target_line - window - 1, 0)
    end = min(target_line + window, len(lines))

    context = []

    for idx in range(start, end):
        line_number = idx + 1
        marker = ">>> " if line_number == target_line else "    "
        context.append(f"{marker}{line_number}: {lines[idx].rstrip()}")

    return context


def find_label_lines(lines, label_to_find):
    results = []

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            continue

        parts = stripped.split()

        if len(parts) < 2:
            continue

        label = parts[-1]

        if label == label_to_find:
            results.append(idx + 1)

    return results


def inspect_known_errors(dataset_name, dataset_dir, report):
    report.write(f"\n\n=== ERRORES BIO DETECTADOS: {dataset_name} ===\n")

    for filename, target_line in ERRORS_TO_INSPECT.get(dataset_name, []):
        path = dataset_dir / filename

        report.write(f"\n--- Archivo: {filename}, línea objetivo: {target_line} ---\n")

        if not path.exists():
            report.write("Archivo no encontrado.\n")
            continue

        lines = read_lines(path)
        context = extract_context(lines, target_line)

        for item in context:
            report.write(item + "\n")


def inspect_invalid_labels(dataset_name, dataset_dir, report):
    report.write(f"\n\n=== ETIQUETAS INVÁLIDAS DETECTADAS: {dataset_name} ===\n")

    for filename, label in INVALID_LABELS_TO_FIND.get(dataset_name, []):
        path = dataset_dir / filename

        report.write(f"\n--- Archivo: {filename}, etiqueta inválida: {label} ---\n")

        if not path.exists():
            report.write("Archivo no encontrado.\n")
            continue

        lines = read_lines(path)
        label_lines = find_label_lines(lines, label)

        if not label_lines:
            report.write("No se encontró la etiqueta.\n")
            continue

        for target_line in label_lines:
            report.write(f"\nContexto línea {target_line}:\n")
            context = extract_context(lines, target_line)

            for item in context:
                report.write(item + "\n")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "bio_errors_context.txt"

    with open(output_path, "w", encoding="utf-8") as report:
        report.write("=== INSPECCIÓN CONTEXTUAL DE ERRORES BIO ===\n")

        inspect_known_errors("CONLL2002", CONLL2002_DIR, report)
        inspect_known_errors("PROSTATA", PROSTATA_DIR, report)

        inspect_invalid_labels("PROSTATA", PROSTATA_DIR, report)

    print(f"Reporte creado: {output_path}")


if __name__ == "__main__":
    main()
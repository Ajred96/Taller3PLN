from pathlib import Path
import shutil

from src.config.config import PROSTATA_DIR, PROCESSED_DIR, REPORTS_DIR


OUTPUT_DIR = PROCESSED_DIR / "prostata"

CORRECTIONS = {
    "testing_cleaned.bio": {
        4977: "B-DOSIS",
        12471: "B-BIOMARCADOR",
        17566: "O",
    },
    "training.bio": {
        1773: "B-TRATAMIENTO",
        15496: "I-EDAD",
    },
    "validation_cleaned.bio": {
        1401: "O",
    },
}


def fix_line_label(line: str, new_label: str) -> str:
    stripped = line.rstrip("\n")

    if not stripped:
        return line

    parts = stripped.split()

    if len(parts) < 2:
        return line

    token = parts[0]
    return f"{token}\t{new_label}\n"


def fix_file(input_path: Path, output_path: Path, corrections_by_line: dict):
    applied = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    new_lines = []

    for index, line in enumerate(lines, start=1):
        if index in corrections_by_line:
            old_line = line.rstrip("\n")
            new_label = corrections_by_line[index]
            new_line = fix_line_label(line, new_label)
            new_lines.append(new_line)

            applied.append(
                {
                    "line": index,
                    "old": old_line,
                    "new": new_line.rstrip("\n"),
                }
            )
        else:
            new_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as file:
        file.writelines(new_lines)

    return applied


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "prostata_annotation_fixes.txt"

    expected_files = [
        "training.bio",
        "validation_cleaned.bio",
        "testing_cleaned.bio",
    ]

    all_applied = {}

    for filename in expected_files:
        input_path = PROSTATA_DIR / filename
        output_path = OUTPUT_DIR / filename

        if not input_path.exists():
            print(f"No encontrado: {input_path}")
            continue

        corrections_by_line = CORRECTIONS.get(filename, {})

        if corrections_by_line:
            applied = fix_file(input_path, output_path, corrections_by_line)
        else:
            shutil.copy(input_path, output_path)
            applied = []

        all_applied[filename] = applied
        print(f"Archivo procesado: {output_path}")

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("=== CORRECCIONES PUNTUALES DATASET PRÓSTATA ===\n\n")
        report.write(f"Origen: {PROSTATA_DIR}\n")
        report.write(f"Destino: {OUTPUT_DIR}\n\n")

        for filename, fixes in all_applied.items():
            report.write(f"\n--- Archivo: {filename} ---\n")

            if not fixes:
                report.write("Sin correcciones aplicadas.\n")
                continue

            for fix in fixes:
                report.write(f"Línea {fix['line']}:\n")
                report.write(f"Antes:   {fix['old']}\n")
                report.write(f"Después: {fix['new']}\n\n")

    print(f"Reporte creado: {report_path}")


if __name__ == "__main__":
    main()
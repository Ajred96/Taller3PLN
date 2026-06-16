from config.config import (
    ROOT_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TASS_DIR,
    SARCASMO_DIR,
    CONLL2002_DIR,
    PROSTATA_DIR,
)


def main():
    print("=== PROYECTO TALLER 3 PLN ===")
    print(f"Raíz del proyecto: {ROOT_DIR}")
    print(f"Datos crudos: {RAW_DIR}")
    print(f"Datos procesados: {PROCESSED_DIR}")
    print(f"Reportes: {REPORTS_DIR}")
    print(f"Figuras: {FIGURES_DIR}")

    print("\n=== DATASETS ESPERADOS ===")
    print(f"TASS: {TASS_DIR}")
    print(f"Sarcasmo: {SARCASMO_DIR}")
    print(f"CoNLL2002: {CONLL2002_DIR}")
    print(f"Próstata: {PROSTATA_DIR}")


if __name__ == "__main__":
    main()
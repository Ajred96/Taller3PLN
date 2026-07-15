from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

TASS_DIR = RAW_DIR / "tass"
SARCASMO_DIR = RAW_DIR / "sarcasmo"
CONLL2002_DIR = RAW_DIR / "conll2002"
PROSTATA_DIR = RAW_DIR / "prostata"
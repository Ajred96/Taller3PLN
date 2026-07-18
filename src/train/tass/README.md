# TASS — Clasificación de sentimientos (3 clases: N / NEU / P)

Punto 1 del taller · Integrante 2.

## Modelos
- **BETO** (`dccuchile/bert-base-spanish-wwm-cased`)
- **XLNet** (`xlnet-base-cased`)
- **XLM-RoBERTa-large** (`xlm-roberta-large`)

Cada runner recorre **batch sizes 8, 16, 32** con **Early Stopping (paciencia 5)**,
selecciona el mejor por **F1 macro** y evalúa sobre **test** (accuracy, F1 macro, F1 weighted).

## Datos
`data/processed/tass/` trae `train.json` (4802) y `test.json` (2443).
El conjunto de **validación** se genera dentro del código: 15% estratificado del train
(`stratify_by_column`, seed=42) — necesario para Early Stopping.

## Ejecución (Colab / GPU)

```bash
pip install -r requirements.txt

# dry-run local (estructura, sin entrenar)
python src/train/tass/train_beto.py

# entrenamiento real (GPU)
RUN=1 python src/train/tass/train_beto.py
RUN=1 python src/train/tass/train_xlnet.py
RUN=1 python src/train/tass/train_xlmroberta.py
```

> Los scripts hacen `from tass_common import ...`, así que ejecútalos desde el
> directorio `src/train/tass/` o añade esa carpeta al `PYTHONPATH`.

Salidas:
- Modelos: `outputs/models/tass/<modelo>_bs<N>/best/`
- Reportes: `outputs/reports/tass_<modelo>_report.txt`

## Inferencia

```bash
python src/predict/tass/predict_tass.py
MODEL=outputs/models/tass/xlmroberta_bs16/best python src/predict/tass/predict_tass.py
```

## Baseline a superar
BETO sobre TASS ~**64% F1** (dato del taller). Objetivo: superarlo con XLNet y XLM-RoBERTa-large.

## Análisis (Punto 5)
Ver `outputs/reports/tass_analisis_punto5.md`.

# Taller 3 - Procesamiento de Lenguaje Natural (PLN)

## Integrantes

| Integrante                     | Responsabilidad |
|--------------------------------|----------------|
| Anderson Johan Alban Angulo    | Preprocesamiento y preparación de datasets |
| Integrante 2                   | Clasificación de sentimientos (TASS) |
| Cristian Camilo Llanos Alvarez | Detección de sarcasmo |
| Integrante 4                   | NER CoNLL2002 |
| Integrante 5                   | NER Biomédico (Próstata) y evaluación |

---

# Objetivo

Desarrollar y evaluar modelos de Procesamiento de Lenguaje Natural (PLN) para tareas de:

- Clasificación de sentimientos.
- Detección de sarcasmo.
- Reconocimiento de entidades nombradas (NER).
- Extracción de entidades biomédicas.

---

# Estructura del Proyecto

```text
Taller3/
│
├── data/
│   ├── raw/
│   │   ├── tass/
│   │   ├── sarcasmo/
│   │   ├── conll2002/
│   │   └── prostata/
│   │
│   └── processed/
│       ├── tass/
│       ├── sarcasmo/
│       ├── conll2002/
│       ├── prostata/
│       └── prostata_json/
│
├── outputs/
│   ├── reports/
│   ├── figures/
│   └── modelBetoSarcasmo/          
│       ├── beto_bs8_best/
│       ├── beto_bs16_best/
│       └── beto_bs32_best/
│
├── src/
│   ├── config/
│   ├── preprocessing/
│   ├── train/
│   │   └── sarcasmo/
│   │       ├── train_beto_Bs8.py
│   │       └── train_beto_Bs16.py
│   └── predict/
│       └── sarcasmo/
│           ├── predictBeto8Bs8.py
│           ├── predictBetoBs16.py
│           └── predictBetoBs32.py
│
├── requirements.txt
└── README.md
```

---

# Datasets Utilizados

## 1. TASS

**Tarea:** Análisis de sentimientos.

Etiquetas:

- P → Positivo
- N → Negativo
- NEU → Neutro

Archivos:

```text
data/raw/tass/
├── cr-tass.csv
└── cr1-tass.csv
```

---

## 2. Sarcasmo

**Tarea:** Detección de sarcasmo.

Etiquetas:

- 0 → No sarcasmo
- 1 → Sarcasmo

Archivos:

```text
data/raw/sarcasmo/
├── train.csv
└── validation.csv
```

---

## 3. CoNLL2002

**Tarea:** Reconocimiento de entidades nombradas (NER).

Entidades:

- PER
- LOC
- ORG
- MISC

Archivos:

```text
data/raw/conll2002/
├── train.txt
├── valid.txt
└── test.txt
```

---

## 4. Próstata

**Tarea:** Extracción de entidades biomédicas.

Ejemplos de entidades:

- CANCER
- FECHA
- BIOMARCADOR
- GLEASON
- DOSIS
- TRATAMIENTO
- MEDICAMENTO
- TNM
- EDAD
- CIRUGIA

Archivos:

```text
data/raw/prostata/
├── training.bio
├── validation_cleaned.bio
└── testing_cleaned.bio
```

---

# Flujo de Preprocesamiento

El proyecto implementa una etapa de preparación de datos antes del entrenamiento de modelos.

## 1. Exploración de Datasets

Script:

```bash
python -m src.preprocessing.explore_datasets
```

Genera:

```text
outputs/reports/
outputs/figures/
```

Incluye:

- Cantidad de registros.
- Distribución de etiquetas.
- Estadísticas generales.
- Gráficas automáticas.

---

## 2. Validación BIO

Script:

```bash
python -m src.preprocessing.validate_bio
```

Objetivo:

- Detectar etiquetas inválidas.
- Detectar errores BIO.
- Verificar consistencia estructural.

---

## 3. Inspección de Errores

Script:

```bash
python -m src.preprocessing.inspect_bio_errors
```

Permite revisar el contexto de errores detectados durante la validación.

---

## 4. Corrección de Anotaciones Biomédicas

Script:

```bash
python -m src.preprocessing.fix_prostata_annotations
```

Realiza correcciones puntuales sobre inconsistencias detectadas en el dataset clínico.

Los archivos originales nunca se modifican.

---

## 5. Conversión a JSON

Script:

```bash
python -m src.preprocessing.create_splits
```

Genera datasets estandarizados para entrenamiento.

Formato clasificación:

```json
{
  "text": "Ejemplo",
  "label": "P"
}
```

Formato NER:

```json
{
  "tokens": ["San", "Sebastián"],
  "labels": ["B-LOC", "I-LOC"]
}
```

---

## 6. Dataset Cards

Script:

```bash
python -m src.preprocessing.export_dataset_cards
```

Genera documentación automática para cada dataset.

---

# Scripts Disponibles

| Script | Descripción |
|----------|----------|
| explore_datasets.py | Exploración y estadísticas |
| validate_bio.py | Validación de etiquetas BIO |
| inspect_bio_errors.py | Inspección contextual de errores |
| fix_prostata_annotations.py | Corrección de anotaciones |
| create_splits.py | Conversión a JSON |
| export_dataset_cards.py | Generación de dataset cards |

---

# Instalación

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar:

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

# Estado Actual del Proyecto

## Completado

- Estructura base del proyecto.
- Organización de datasets.
- Exploración de datos.
- Reportes automáticos.
- Gráficas de distribución.
- Validación BIO.
- Corrección de inconsistencias.
- Conversión a JSON.
- Dataset Cards.

## Pendiente

- Entrenamiento de modelos.
- Evaluación.
- Comparación de resultados.
- Métricas finales.
- Conclusiones.

---

# Resultados

> Esta sección será completada durante las etapas de entrenamiento y evaluación de modelos.
# Taller 3 - Procesamiento de Lenguaje Natural (PLN)

## Integrantes

| Integrante | Responsabilidad |
|--------------------------------|----------------|
| Anderson Johan Alban Angulo | Preprocesamiento y preparación de datasets |
| Integrante 2 | Clasificación de sentimientos (TASS) |
| Cristian Camilo Llanos Alvarez | Detección de sarcasmo |
| Jhoan Felipe Leon | NER CoNLL2002 |
| Juan Esteban Clavijo García | NER Biomédico (Próstata) y Prompt Engineering |

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
│   │   └── prompt_engineering/
│   │       ├── resultados_prompting.json
│   │       └── reporte_prompting.txt
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
│   │   ├── sarcasmo/
│   │   │   ├── train_beto_Bs8.py
│   │   │   └── train_beto_Bs16.py
│   │   └── ner_llm/
│   │       ├── utils_ner.py
│   │       ├── train_beto_ner.py
│   │       ├── train_xlmroberta_ner.py
│   │       └── main_ner.py
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

**Tarea:** Extracción de entidades biomédicas en textos clínicos de cáncer de próstata.

Entidades:

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

| Script                      | Descripción                                       |
| --------------------------- | ------------------------------------------------- |
| explore_datasets.py         | Exploración y estadísticas                        |
| validate_bio.py             | Validación de etiquetas BIO                       |
| inspect_bio_errors.py       | Inspección contextual de errores                  |
| fix_prostata_annotations.py | Corrección de anotaciones                         |
| create_splits.py            | Conversión a JSON                                 |
| export_dataset_cards.py     | Generación de dataset cards                       |
| utils_ner.py                | Utilidades compartidas para NER con LLMs          |
| train_beto_ner.py           | Fine-tuning de BETO sobre dataset próstata        |
| train_xlmroberta_ner.py     | Fine-tuning de XLM-RoBERTa sobre dataset próstata |
| main_ner.py                 | Pipeline completo NER + tabla comparativa         |

---

# NER Biomédico — Integrante 5

Esta sección documenta el trabajo de reconocimiento de entidades nombradas sobre el dataset clínico de próstata, dividido en dos partes.

---

## Parte A: Fine-tuning de BETO y XLM-RoBERTa

Se entrenaron 6 modelos en total (2 arquitecturas × 3 batch sizes) sobre el dataset de próstata en formato BIO, usando `AutoModelForTokenClassification` con Early Stopping de 3 épocas y hasta 10 épocas de entrenamiento.

### Ejecutar entrenamiento

```bash
# Entrenar ambos modelos
python -m src.train.ner_llm.main_ner

# Entrenar solo BETO
python -m src.train.ner_llm.main_ner --modelo beto

# Entrenar solo XLM-RoBERTa
python -m src.train.ner_llm.main_ner --modelo xlmroberta
```

### Resultados — F1 sobre conjunto de prueba

| Modelo      | Batch size | F1         | Precision  | Recall     |
| ----------- | ---------- | ---------- | ---------- | ---------- |
| BETO        | 8          | 0.9631     | 0.9598     | 0.9666     |
| BETO        | 16         | 0.9644     | 0.9617     | 0.9671     |
| BETO        | 32         | 0.9584     | 0.9534     | 0.9635     |
| XLM-RoBERTa | 8          | 0.9621     | 0.9597     | 0.9645     |
| XLM-RoBERTa | 16         | **0.9681** | **0.9667** | **0.9696** |
| XLM-RoBERTa | 32         | 0.9542     | 0.9484     | 0.9600     |

**Referencia baseline:** bert-base-cased sobre CoNLL2002 → F1 = 0.75

Todos los modelos superaron ampliamente el baseline. XLM-RoBERTa con batch size 16 obtuvo el mejor F1 (0.9681). El batch size 16 resultó óptimo para ambas arquitecturas; batch size 32 redujo el rendimiento en los dos casos.

### Modelos publicados en Hugging Face

| Modelo                | Enlace                                                    |
| --------------------- | --------------------------------------------------------- |
| BETO NER bs=8         | https://huggingface.co/Jnch7/beto-ner-prostata-bs8        |
| BETO NER bs=16        | https://huggingface.co/Jnch7/beto-ner-prostata-bs16       |
| BETO NER bs=32        | https://huggingface.co/Jnch7/beto-ner-prostata-bs32       |
| XLM-RoBERTa NER bs=8  | https://huggingface.co/Jnch7/xlmroberta-ner-prostata-bs8  |
| XLM-RoBERTa NER bs=16 | https://huggingface.co/Jnch7/xlmroberta-ner-prostata-bs16 |
| XLM-RoBERTa NER bs=32 | https://huggingface.co/Jnch7/xlmroberta-ner-prostata-bs32 |

---

## Parte B: Prompt Engineering con LLMs Generativos

Se aplicaron 4 modelos generativos grandes sobre un texto clínico de cáncer de próstata, guiados únicamente mediante instrucciones textuales. No se realizó fine-tuning.

Los resultados están disponibles en:

```text
outputs/reports/prompt_engineering/
├── resultados_prompting.json
└── reporte_prompting.txt
```

### Modelos utilizados

| Modelo                                    | Parámetros | Cuantización |
| ----------------------------------------- | ---------- | ------------ |
| mistralai/Mistral-7B-Instruct-v0.2        | 7B         | 4 bits       |
| google/gemma-2-2b-it                      | 2B         | 4 bits       |
| meta-llama/Llama-3.2-3B-Instruct          | 3B         | 4 bits       |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B       | 4 bits       |

### Entidades extraídas por modelo

| Entidad                   | Mistral                      | Gemma            | Llama                | DeepSeek        |
| ------------------------- | ---------------------------- | ---------------- | -------------------- | --------------- |
| EDAD                      | ✔ 72 años                    | ✔ 72             | ✔ 72 años            | ✔ 72 años       |
| CANCER                    | ✔ adenocarcinoma             | ✔ adenocarcinoma | ✔ cáncer de próstata | ✗ incorrecto    |
| GLEASON                   | ✔ 3+3                        | ✔ 3+3            | ✔ 3+3                | ✔ 3+3           |
| BIOMARCADOR               | ✔ PSA 9,9 ng/dL              | ✔ PSA            | ✔ PSA                | ✔ PSA 9,9 ng/dL |
| CIRUGIA                   | ✗ clasificó como TRATAMIENTO | ✗ vacío          | ✗ vacío              | ✗ vacío         |
| TNM / DOSIS / MEDICAMENTO | ✗                            | ✗                | ✗                    | ✗               |

### Conclusión comparativa

El fine-tuning (Parte A) superó ampliamente al prompt engineering en precisión estructurada: BETO y XLM-RoBERTa lograron F1 de 0.96-0.97 sobre el mismo dominio médico, mientras que los modelos generativos cometieron errores conceptuales, ignoraron entidades como CIRUGIA y no siguieron consistentemente el formato solicitado. Llama y Gemma siguieron mejor las instrucciones de formato; Mistral y DeepSeek respondieron en JSON propio ignorando el formato pedido.

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
- Fine-tuning BETO para NER biomédico (3 batch sizes).
- Fine-tuning XLM-RoBERTa para NER biomédico (3 batch sizes).
- Publicación de 6 modelos en Hugging Face.
- Prompt Engineering con 4 LLMs generativos.
- Comparación fine-tuning vs prompt engineering.

## Pendiente

- Entrenamiento modelos TASS.
- Entrenamiento modelos Sarcasmo.
- NER CoNLL2002 con BiLSTM.
- Métricas finales consolidadas.
- Conclusiones generales del grupo.

---

# Resultados




## NER Biomédico — Próstata

| Modelo | Batch | F1 | Precision | Recall |
| ----------- | ----- | ---------- | ---------- | ---------- |
| BETO | 8 | 0.9631 | 0.9598 | 0.9666 |
| BETO | 16 | 0.9644 | 0.9617 | 0.9671 |
| BETO | 32 | 0.9584 | 0.9534 | 0.9635 |
| XLM-RoBERTa | 8 | 0.9621 | 0.9597 | 0.9645 |
| XLM-RoBERTa | 16 | **0.9681** | **0.9667** | **0.9696** |
| XLM-RoBERTa | 32 | 0.9542 | 0.9484 | 0.9600 |

Modelos disponibles en: https://huggingface.co/JuanC513

## NER CoNLL2002 — BiLSTM + CRF

Se entrenaron dos variantes de BiLSTM+CRF sobre CoNLL2002: con FastText, y con FastText más un canal convolucional de caracteres (CNN). La variante con CNN obtuvo el mejor resultado.

| Modelo | F1 | Precision | Recall |
| ----------------------- | ------ | --------- | ------ |
| Baseline (enunciado) | 0.6430 | - | - |
| BiLSTM+CRF+FastText | 0.6809 | 0.7143 | 0.6505 |
| BiLSTM+CRF+CNN+FastText | 0.7828 | 0.7768 | 0.7890 |

La variante con CNN mejoró +13.98 puntos F1 sobre el baseline del enunciado. Por tipo de entidad, la mayor mejora se dio en MISC (+17.9 pts), seguida de ORG (+10.5 pts) y LOC (+9.0 pts).

# Dataset Card: PROSTATA

## Descripción

Dataset clínico anotado en formato BIO con entidades biomédicas como CANCER, FECHA, BIOMARCADOR, GLEASON, DOSIS, TRATAMIENTO, MEDICAMENTO, TNM, EDAD y CIRUGIA.

## Tarea

- Tipo: NER biomédico
- Objetivo: Extracción de entidades clínicas relacionadas con cáncer de próstata
- Formato procesado: JSON

## Splits disponibles

| Split | Registros | Estado |
|---|---:|---|
| train | 3106 | OK |
| validation | 929 | OK |
| test | 991 | OK |

Total de registros: **5026**

## Distribución global de etiquetas

| Etiqueta | Cantidad |
|---|---:|
| O | 82901 |
| I-CANCER | 3297 |
| I-FECHA | 2253 |
| B-FECHA | 1982 |
| B-CANCER | 1867 |
| I-DOSIS | 1612 |
| B-TRATAMIENTO | 1539 |
| I-GLEASON | 1471 |
| I-BIOMARCADOR | 1330 |
| B-BIOMARCADOR | 1300 |
| B-DOSIS | 1081 |
| I-TRATAMIENTO | 1032 |
| B-GLEASON | 906 |
| B-MEDICAMENTO | 738 |
| B-TNM | 537 |
| I-CIRUGIA | 467 |
| B-EDAD | 393 |
| B-CIRUGIA | 392 |
| I-EDAD | 387 |
| I-MEDICAMENTO | 82 |
| I-TNM | 32 |

## Ejemplos

### train

```json
{
  "tokens": [
    "Paciente",
    "de",
    "72",
    "años",
    ",",
    "con",
    "antecedentes",
    "médicos",
    "de",
    "HTA",
    "."
  ],
  "labels": [
    "O",
    "O",
    "B-EDAD",
    "I-EDAD",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O"
  ]
}
```

### validation

```json
{
  "tokens": [
    "Fecha",
    "13-11-2022",
    "."
  ],
  "labels": [
    "O",
    "B-FECHA",
    "O"
  ]
}
```

### test

```json
{
  "tokens": [
    "Abril",
    "8",
    "de",
    "2022",
    ",",
    "paciente",
    "con",
    "Tumor",
    "maligno",
    "de",
    "la",
    "próstata",
    ",",
    "ambulatorio",
    "en",
    "buena",
    "condición",
    "general",
    "con",
    "limitación",
    "funcional",
    "leve",
    "al",
    "momento",
    "por",
    "férula",
    "en",
    "msi",
    ",",
    "ojos",
    "pupilas",
    "irr",
    "conjuntivas",
    "rosadas",
    ",",
    "cuello",
    "sin",
    "masas",
    "aparentes",
    ",",
    "abdomen-",
    "pelvis",
    "depresible",
    "sin",
    "dolor",
    "ni",
    "masas",
    ",",
    "extremidades",
    "sin",
    "edemas",
    "férula",
    "msi",
    "pulsos",
    "bien",
    "movilidad",
    "normal",
    ",",
    "neurológico",
    "sin",
    "déficit",
    "al",
    "momento",
    "."
  ],
  "labels": [
    "B-FECHA",
    "I-FECHA",
    "I-FECHA",
    "I-FECHA",
    "O",
    "O",
    "O",
    "B-CANCER",
    "I-CANCER",
    "I-CANCER",
    "I-CANCER",
    "I-CANCER",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O",
    "O"
  ]
}
```

## Observaciones de preprocesamiento

- Se conservaron los archivos originales en `data/raw/prostata`.
- Se aplicaron correcciones puntuales sobre anotaciones inconsistentes detectadas durante la validación BIO.
- Los archivos corregidos fueron exportados a `data/processed/prostata` y convertidos posteriormente a JSON en `data/processed/prostata_json`.

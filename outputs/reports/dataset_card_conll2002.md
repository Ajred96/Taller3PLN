# Dataset Card: CONLL2002

## Descripción

Dataset CoNLL2002 en español con entidades PER, LOC, ORG y MISC en formato BIO.

## Tarea

- Tipo: NER
- Objetivo: Reconocimiento de entidades nombradas
- Formato procesado: JSON

## Splits disponibles

| Split | Registros | Estado |
|---|---:|---|
| train | 8323 | OK |
| validation | 1915 | OK |
| test | 1517 | OK |

Total de registros: **11755**

## Distribución global de etiquetas

| Etiqueta | Cantidad |
|---|---:|
| O | 322631 |
| B-ORG | 10490 |
| I-ORG | 7462 |
| B-LOC | 6981 |
| B-PER | 6278 |
| I-PER | 5396 |
| I-MISC | 4423 |
| B-MISC | 2957 |
| I-LOC | 2553 |

## Ejemplos

### train

```json
{
  "tokens": [
    "Melbourne",
    "(",
    "Australia",
    ")",
    ",",
    "25",
    "may",
    "(",
    "EFE",
    ")",
    "."
  ],
  "labels": [
    "B-LOC",
    "O",
    "B-LOC",
    "O",
    "O",
    "O",
    "O",
    "O",
    "B-ORG",
    "O",
    "O"
  ]
}
```

### validation

```json
{
  "tokens": [
    "Sao",
    "Paulo",
    "(",
    "Brasil",
    ")",
    ",",
    "23",
    "may",
    "(",
    "EFECOM",
    ")",
    "."
  ],
  "labels": [
    "B-LOC",
    "I-LOC",
    "O",
    "B-LOC",
    "O",
    "O",
    "O",
    "O",
    "O",
    "B-ORG",
    "O",
    "O"
  ]
}
```

### test

```json
{
  "tokens": [
    "La",
    "Coruña",
    ",",
    "23",
    "may",
    "(",
    "EFECOM",
    ")",
    "."
  ],
  "labels": [
    "B-LOC",
    "I-LOC",
    "O",
    "O",
    "O",
    "O",
    "B-ORG",
    "O",
    "O"
  ]
}
```

## Observaciones de preprocesamiento

- El dataset se conservó sin modificaciones, ya que las anomalías BIO detectadas fueron mínimas y corresponden a diferencias de anotación del corpus.

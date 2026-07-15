# Dataset Card: TASS

## Descripción

Dataset TASS en español con tres clases de sentimiento: negativo, neutro y positivo.

## Tarea

- Tipo: Clasificación multiclase
- Objetivo: Análisis de sentimientos
- Formato procesado: JSON

## Splits disponibles

| Split | Registros | Estado |
|---|---:|---|
| train | 4802 | OK |
| test | 2443 | OK |

Total de registros: **7245**

## Distribución global de etiquetas

| Etiqueta | Cantidad |
|---|---:|
| N | 2836 |
| NEU | 2316 |
| P | 2093 |

## Ejemplos

### train

```json
{
  "text": "No mames este pinche dolor que pedo? ya mejor llévame Diosito.",
  "label": "N"
}
```

### test

```json
{
  "text": "Inteligente respuesta de Putin",
  "label": "P"
}
```

## Observaciones de preprocesamiento

- El dataset fue normalizado a una estructura común con campos `text` y `label`.

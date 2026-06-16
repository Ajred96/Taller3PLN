# Dataset Card: SARCASMO

## Descripción

Dataset en español para clasificación binaria entre sarcasmo y no sarcasmo.

## Tarea

- Tipo: Clasificación binaria
- Objetivo: Detección de sarcasmo
- Formato procesado: JSON

## Splits disponibles

| Split | Registros | Estado |
|---|---:|---|
| train | 15276 | OK |
| validation | 3820 | OK |

Total de registros: **19096**

## Distribución global de etiquetas

| Etiqueta | Cantidad |
|---|---:|
| 0 | 11483 |
| 1 | 7613 |

## Ejemplos

### train

```json
{
  "text": "Así son las grabaciones que comprometen a Fernández Díaz",
  "label": "1"
}
```

### validation

```json
{
  "text": "Roba el martell d’un jutge del Tribunal Suprem i indulta tots els presos independentistes",
  "label": "1"
}
```

## Observaciones de preprocesamiento

- El dataset fue normalizado a una estructura común con campos `text` y `label`.

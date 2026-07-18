# Análisis Punto 5 — TASS vs Sarcasmo

> Rellenar las tablas de TASS tras ejecutar los runners en GPU
> (`outputs/reports/tass_*_report.txt`). El lado Sarcasmo ya está medido.

## Tabla resumen (test)

| Dataset  | Modelo             | Batch | Accuracy | F1 (macro/clase) |
|----------|--------------------|-------|----------|------------------|
| TASS     | BETO               | 8/16/32 | _pend_ | _pend_ |
| TASS     | XLNet              | 8/16/32 | _pend_ | _pend_ |
| TASS     | XLM-RoBERTa-large  | 8/16/32 | _pend_ | _pend_ |
| Sarcasmo | BETO (BS=32) ✅    | 32    | 90.96%   | 88.16% |
| Sarcasmo | XLM-RoBERTa-large  | 16    | 63.08%   | 12.31% |

## Preguntas del taller

### 1. ¿Por qué los rendimientos sobre TASS son menores que sobre Sarcasmo?
- **Número de clases:** TASS es 3 clases (N/NEU/P) vs Sarcasmo 2 clases. Más clases
  reparten la probabilidad y el azar baja de 50% a ~33%, subiendo la dificultad.
- **Frontera semántica difusa:** la clase NEU (neutro) se solapa con N y P; el sarcasmo
  binario tiene señales léxicas/pragmáticas más marcadas.
- **Balance:** TASS train N=1885 / NEU=1523 / P=1394 (relativamente balanceado); el reto
  es la ambigüedad, no el desbalance.

### 2. ¿Qué papel juega el número de clases en ambos datasets?
- 2 clases (Sarcasmo) → tarea binaria, F1 binario directo, baseline aleatorio 50%.
- 3 clases (TASS) → se reporta **F1 macro** (promedio no ponderado) y **weighted**;
  baseline aleatorio ~33%. La métrica macro penaliza el mal desempeño en la clase minoritaria (P).

### 3. Comparar XLNet y XLM-RoBERTa-large frente a BETO (sobre TASS)
- **BETO:** monolingüe español nativo → representación refinada, converge rápido (referencia).
- **XLNet (`xlnet-base-cased`):** no es nativo español; su modelado permutacional puede
  ayudar en contexto, pero la falta de preentrenamiento en español limita el techo.
- **XLM-RoBERTa-large:** multilingüe gigante; con la mayoría de capas congeladas tiende al
  **underfitting** (como se vio en Sarcasmo). Requiere más épocas / descongelar más capas
  para "despertar" en español.

_Completar con las cifras reales una vez entrenados los tres modelos._

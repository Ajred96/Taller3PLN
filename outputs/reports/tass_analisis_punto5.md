# Análisis Punto 5 — TASS vs Sarcasmo

> TASS medido (BETO, XLNet, XLM-RoBERTa-large). Sarcasmo medido. Detalle por época: `tass_informe_completo.md`.

## Tabla resumen (test)

| Dataset  | Modelo             | Batch | Accuracy | F1 macro |
|----------|--------------------|-------|----------|----------|
| TASS     | BETO               | 8     | 64.88%   | 64.18%   |
| TASS     | BETO               | 16    | 66.56%   | 66.10%   |
| TASS     | BETO ✅ (mejor)    | 32    | 66.88%   | 66.81%   |
| TASS     | XLNet ✅ (mejor)   | 8     | 53.34%   | 53.35%   |
| TASS     | XLNet              | 16    | 53.21%   | 51.82%   |
| TASS     | XLNet              | 32    | 53.34%   | 52.89%   |
| TASS     | XLM-RoBERTa-large ✅ (mejor) | 16 | 67.38% | **67.30%** |
| TASS     | XLM-RoBERTa-large  | 8     | 67.54%   | 67.17%   |
| TASS     | XLM-RoBERTa-large  | 32    | 67.42%   | 66.45%   |
| Sarcasmo | BETO (BS=32) ✅    | 32    | 90.96%   | 88.16% (binario) |
| Sarcasmo | XLM-RoBERTa-large  | 16    | 63.08%   | 12.31% (binario) |

**Ganador TASS: XLM-RoBERTa-large bs16 (F1 macro 67.30%)**, seguido de BETO bs32 (66.81%).
Los 3 salvo XLNet superan el baseline del taller (64% f1).
HF: [beto](https://huggingface.co/gustavoa6791/beto-tass-sentiment-univalle-pln) ·
[xlnet](https://huggingface.co/gustavoa6791/xlnet-tass-sentiment-univalle-pln) ·
[xlmroberta](https://huggingface.co/gustavoa6791/xlmroberta-tass-sentiment-univalle-pln)

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
- **XLM-RoBERTa-large:** ganador (bs16, F1 macro 67.30%). Aun con el cuerpo congelado (solo
  6 capas activas), su capacidad multilingüe rinde bien en TASS. Contraste notable con Sarcasmo,
  donde el mismo modelo colapsó (F1 12%): el factor decisivo es la tarea/dataset, no el modelo.
- **BETO:** monolingüe español nativo (bs32, 66.81%). A ~0.5pt del ganador con 5× menos
  parámetros → mejor costo/beneficio, opción recomendada en producción.
- **XLNet (`xlnet-base-cased`):** no nativo español; se estanca ~53% (apenas sobre azar).
  La falta de preentrenamiento en español limita el techo.

### 4. Efecto del batch size
- **BETO:** batch grande mejor (bs32 > bs16 > bs8).
- **XLM-R:** óptimo bs16; bs32 baja el F1 macro.
- **XLNet:** batch grande no ayuda; no aprende bien en ningún caso.

_Completar con las cifras reales una vez entrenados los tres modelos._

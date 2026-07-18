# Informe TASS — Clasificación de Sentimientos (Punto 1)

**Integrante 2** · Dataset TASS · 3 clases: N (negativo), NEU (neutro), P (positivo)
**Última actualización:** 2026-07-18

---

## Fase I — Preprocesamiento

- Dataset: `data/processed/tass/` — train 4802, test 2443.
- Validación generada en código: **15% estratificado** del train (seed=42) → train 4081 / val 721.
- Distribución train: N=1885, NEU=1523, P=1394.
- Tokenización: `AutoTokenizer` por modelo, truncation, `max_length=128`.
- Padding dinámico: `DataCollatorWithPadding`.
- Etiquetas → id: `{N:0, NEU:1, P:2}` (columna `ClassLabel` para split estratificado).

## Fase II — Entrenamiento (diseño experimental)

| Hiperparámetro | Valor |
|---|---|
| Modelos | BETO, XLNet, XLM-RoBERTa-large |
| Batch Size | 8, 16, 32 |
| Épocas | máx 10 (XLMR 8) |
| Optimizador | AdamW (default HF), lr 1e-5 (BETO/XLMR), 2e-5 (XLNet) |
| Early Stopping | paciencia 5 (monitor F1 macro) |
| Weight decay | 0.01 |
| Selección | mejor F1 macro (val) |
| Métrica reporte | Accuracy, F1 macro, F1 weighted (test) |

Baseline del taller a superar: **64% F1** (BETO base sobre TASS).

## Fase III — Resultados sobre TEST

### Tabla resumen (mejor por batch)

| Modelo | Batch | Épocas | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| **BETO** | 8 | 6/10 | 64.88% | 64.18% | 64.51% |
| **BETO** | 16 | 8/10 | 66.56% | 66.10% | 66.45% |
| **BETO** ✅ | 32 | 10/10 | **66.88%** | **66.81%** | **66.99%** |
| XLNet ✅ | 8 | 9/10 | 53.34% | 53.35% | 53.65% |
| XLNet | 16 | 10/10 | 53.21% | 51.82% | 52.17% |
| XLNet | 32 | 10/10 | 53.34% | 52.89% | 53.11% |
| XLM-RoBERTa-large ✅ | 16 | 8/8 | 67.38% | **67.30%** | 67.45% |
| XLM-RoBERTa-large | 8 | 8/8 | 67.54% | 67.17% | 67.37% |
| XLM-RoBERTa-large | 32 | 8/8 | 67.42% | 66.45% | 66.68% |

**Ganador global: XLM-RoBERTa-large bs16 (F1 macro 67.30%).** Supera baseline 64% y a BETO (66.81%).
Ranking: XLM-R bs16 (67.30%) > BETO bs32 (66.81%) > XLNet bs8 (53.35%).

### Evolución por época — BETO

**bs8** (early stop época 6; mejor val época 1):
| Época | Train Loss | Val Loss | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| 1 | 0.7267 | 0.7298 | 0.6963 | 0.6920 | 0.6952 |
| 2 | 0.6070 | 0.7936 | 0.6713 | 0.6633 | 0.6674 |
| 3 | 0.3663 | 1.0249 | 0.6685 | 0.6564 | 0.6616 |
| 4 | 0.2980 | 1.4828 | 0.6699 | 0.6717 | 0.6740 |
| 5 | 0.1692 | 1.8710 | 0.6657 | 0.6657 | 0.6694 |
| 6 | 0.1656 | 2.0967 | 0.6588 | 0.6588 | 0.6637 |

**bs16** (early stop época 8; mejor val época 3):
| Época | Train Loss | Val Loss | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| 1 | 0.7895 | 0.7511 | 0.6768 | 0.6685 | 0.6733 |
| 2 | 0.6356 | 0.7418 | 0.6852 | 0.6836 | 0.6861 |
| 3 | 0.4483 | 0.8229 | 0.6893 | 0.6884 | 0.6901 |
| 4 | 0.3079 | 0.9814 | 0.6782 | 0.6805 | 0.6827 |
| 5 | 0.2121 | 1.2662 | 0.6699 | 0.6732 | 0.6752 |
| 6 | 0.1663 | 1.4999 | 0.6616 | 0.6611 | 0.6639 |
| 7 | 0.1376 | 1.6824 | 0.6685 | 0.6696 | 0.6723 |
| 8 | 0.0899 | 1.7955 | 0.6796 | 0.6812 | 0.6833 |

**bs32** (10 épocas; mejor val época 2):
| Época | Train Loss | Val Loss | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| 1 | 0.9548 | 0.7689 | 0.6644 | 0.6486 | 0.6534 |
| 2 | 0.6959 | 0.7356 | 0.6865 | 0.6831 | 0.6877 |
| 3 | 0.6052 | 0.7649 | 0.6796 | 0.6777 | 0.6802 |
| 4 | 0.4172 | 0.8483 | 0.6852 | 0.6838 | 0.6856 |
| 5 | 0.3310 | 0.9402 | 0.6865 | 0.6864 | 0.6887 |
| 6 | 0.2568 | 1.0236 | 0.6810 | 0.6791 | 0.6813 |
| 7 | 0.2240 | 1.0955 | 0.6699 | 0.6688 | 0.6723 |
| 8 | 0.1437 | 1.1669 | 0.6755 | 0.6751 | 0.6776 |
| 9 | 0.1283 | 1.2171 | 0.6713 | 0.6697 | 0.6717 |
| 10 | 0.1109 | 1.2406 | 0.6768 | 0.6770 | 0.6790 |

### Evolución por época — XLNet

**bs8** (early stop época 9):
| Época | Train Loss | Val Loss | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| 1 | 1.0797 | 1.0211 | 0.4993 | 0.4547 | 0.4702 |
| 2 | 0.9689 | 0.9846 | 0.5284 | 0.5056 | 0.5165 |
| 3 | 0.9078 | 1.0415 | 0.5257 | 0.5004 | 0.5018 |
| 4 | 0.7629 | 0.9819 | 0.5839 | 0.5821 | 0.5858 |
| 5 | 0.5801 | 1.0164 | 0.5756 | 0.5776 | 0.5807 |
| 6 | 0.4521 | 1.2936 | 0.5409 | 0.5438 | 0.5450 |
| 7 | 0.3979 | 1.6120 | 0.5798 | 0.5786 | 0.5817 |
| 8 | 0.3298 | 1.7598 | 0.5645 | 0.5673 | 0.5702 |
| 9 | 0.3118 | 2.1768 | 0.5728 | 0.5639 | 0.5678 |

**bs16** (10 épocas):
| Época | Train Loss | Val Loss | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| 1 | 1.1012 | 1.1113 | 0.3925 | 0.1879 | 0.2213 |
| 2 | 1.0713 | 1.0651 | 0.4619 | 0.3608 | 0.3813 |
| 3 | 0.9977 | 0.9809 | 0.5090 | 0.4455 | 0.4526 |
| 4 | 0.8704 | 0.9774 | 0.5021 | 0.5052 | 0.4957 |
| 5 | 0.8011 | 0.9660 | 0.5756 | 0.5593 | 0.5651 |
| 6 | 0.6513 | 1.0415 | 0.5825 | 0.5708 | 0.5744 |
| 7 | 0.5461 | 1.1250 | 0.5714 | 0.5703 | 0.5714 |
| 8 | 0.4623 | 1.2134 | 0.5659 | 0.5645 | 0.5655 |
| 9 | 0.4091 | 1.3285 | 0.5645 | 0.5597 | 0.5618 |
| 10 | 0.3353 | 1.3820 | 0.5548 | 0.5529 | 0.5548 |

**bs32** (10 épocas):
| Época | Train Loss | Val Loss | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| 1 | 1.1017 | 1.0902 | 0.3925 | 0.1879 | 0.2213 |
| 2 | 1.0972 | 1.0741 | 0.4522 | 0.3595 | 0.3789 |
| 3 | 1.0493 | 1.0102 | 0.4716 | 0.4194 | 0.4242 |
| 4 | 0.9167 | 0.9173 | 0.5548 | 0.5287 | 0.5349 |
| 5 | 0.8556 | 0.9132 | 0.5742 | 0.5715 | 0.5710 |
| 6 | 0.7653 | 0.9776 | 0.5853 | 0.5453 | 0.5506 |
| 7 | 0.7169 | 1.0354 | 0.5839 | 0.5793 | 0.5810 |
| 8 | 0.5647 | 1.0994 | 0.5589 | 0.5597 | 0.5593 |
| 9 | 0.5266 | 1.2093 | 0.5562 | 0.5534 | 0.5527 |
| 10 | 0.5087 | 1.2153 | 0.5617 | 0.5577 | 0.5589 |

### Evolución por época — XLM-RoBERTa-large (fp16, cuerpo congelado + últimas 6 capas)

**bs8** (8 épocas; mejor val época 5):
| Época | Train Loss | Val Loss | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| 1 | 1.6097 | 0.7819 | 0.6644 | 0.6513 | 0.6557 |
| 2 | 1.5204 | 0.7830 | 0.6685 | 0.6568 | 0.6620 |
| 3 | 1.3859 | 0.7167 | 0.6727 | 0.6556 | 0.6604 |
| 4 | 1.3846 | 0.7504 | 0.6796 | 0.6777 | 0.6793 |
| 5 | 1.1246 | 0.7612 | 0.6893 | 0.6872 | 0.6896 |
| 6 | 1.1986 | 0.7927 | 0.6852 | 0.6764 | 0.6797 |
| 7 | 1.0944 | 0.8046 | 0.6796 | 0.6773 | 0.6795 |
| 8 | 1.0046 | 0.8281 | 0.6782 | 0.6738 | 0.6763 |

**bs16** (8 épocas; mejor val época 5):
| Época | Train Loss | Val Loss | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| 1 | 4.3394 | 0.8191 | 0.6255 | 0.6183 | 0.6223 |
| 2 | 3.1228 | 0.7567 | 0.6824 | 0.6809 | 0.6835 |
| 3 | 2.9131 | 0.7275 | 0.6810 | 0.6685 | 0.6723 |
| 4 | 2.7113 | 0.7132 | 0.6824 | 0.6762 | 0.6791 |
| 5 | 2.6692 | 0.7205 | 0.6935 | 0.6923 | 0.6947 |
| 6 | 2.4443 | 0.7306 | 0.6893 | 0.6769 | 0.6811 |
| 7 | 2.3741 | 0.7416 | 0.6921 | 0.6856 | 0.6882 |
| 8 | 2.2275 | 0.7401 | 0.6963 | 0.6894 | 0.6922 |

**bs32** (8 épocas; mejor val época 8):
| Época | Train Loss | Val Loss | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|---|---|
| 1 | 8.8589 | 1.0793 | 0.3925 | 0.1879 | 0.2213 |
| 2 | 8.2315 | 0.7463 | 0.6630 | 0.6625 | 0.6644 |
| 3 | 6.1679 | 0.7168 | 0.6879 | 0.6797 | 0.6822 |
| 4 | 5.7110 | 0.7306 | 0.6741 | 0.6543 | 0.6588 |
| 5 | 5.5737 | 0.7227 | 0.6907 | 0.6886 | 0.6902 |
| 6 | 5.3019 | 0.7118 | 0.6865 | 0.6789 | 0.6814 |
| 7 | 5.1572 | 0.7147 | 0.6838 | 0.6814 | 0.6827 |
| 8 | 5.0295 | 0.7143 | 0.6949 | 0.6891 | 0.6911 |

### Métricas por clase (classification_report)

_Pendiente: correr `classification_report` desde los modelos en HF (precision/recall/F1 para N, NEU, P). Script en `src/train/tass/README.md` / celda Colab entregada._

## Modelos publicados en Hugging Face

- BETO: https://huggingface.co/gustavoa6791/beto-tass-sentiment-univalle-pln
- XLNet: https://huggingface.co/gustavoa6791/xlnet-tass-sentiment-univalle-pln
- XLM-RoBERTa-large: https://huggingface.co/gustavoa6791/xlmroberta-tass-sentiment-univalle-pln

---

## Punto 5 — Análisis

### ¿Por qué TASS rinde menos que Sarcasmo?
- **Nº de clases:** TASS 3 (N/NEU/P) vs Sarcasmo 2. Azar baja de 50% a ~33%; frontera más difícil.
- **Ambigüedad:** la clase NEU se solapa con N y P; el sarcasmo binario tiene señales léxicas/pragmáticas más marcadas.
- Resultado: mejor TASS (BETO 66.8% F1 macro) ≪ mejor Sarcasmo (BETO 88.2% F1).

### ¿Qué papel juega el número de clases?
- 2 clases → F1 binario, baseline 50%.
- 3 clases → se reporta F1 **macro** (promedio no ponderado, penaliza clase minoritaria P) y **weighted**; baseline ~33%.

### Comparación XLNet y XLM-RoBERTa-large frente a BETO (TASS)
- **XLM-RoBERTa-large** (multilingüe, 560M): **ganador**, bs16 F1 macro 67.30%. Aun con el cuerpo
  congelado (solo 6 capas), su capacidad multilingüe rinde bien en TASS. Contraste con Sarcasmo,
  donde el mismo modelo colapsó (F1 12%): la diferencia es el dataset/tarea, no el modelo.
- **BETO** (monolingüe español, 110M): segundo, bs32 F1 66.81%. A ~0.5pt del ganador con 5× menos
  parámetros → altísima eficiencia. Converge rápido.
- **XLNet** (`xlnet-base-cased`, no nativo español): estancado ~53% (apenas sobre azar). Empeora con
  batch grande; falta de preentrenamiento en español = techo bajo.

### Efecto del Batch Size
- **BETO:** batch grande → mejor (bs32 > bs16 > bs8). Gradiente más estable, menos overfitting (bs8 dispara val_loss desde época 3).
- **XLM-R:** óptimo en bs16 (67.30%); bs32 baja el F1 macro (66.45%). bs8 y bs16 muy parejos.
- **XLNet:** batch grande NO ayuda (bs8 ≥ bs32); el modelo no aprende bien en ningún caso.

### Conclusión
Para TASS el mejor es **XLM-RoBERTa-large (bs16, F1 macro 67.30%)**, seguido muy de cerca por
**BETO (bs32, 66.81%)**. Los 3 salvo XLNet superan el baseline 64%. La lección clave: el mismo
XLM-R que fracasó en Sarcasmo (2 clases, señal binaria) triunfa en TASS (3 clases) — el rendimiento
depende de la tarea, no solo del modelo. Aun así BETO ofrece el mejor costo/beneficio (5× más
pequeño, casi el mismo F1), por lo que en producción sería la opción recomendada.

# Análisis de sentimiento y resúmenes · Amazon Fine Food Reviews

Proyecto final del curso de Deep Learning .
La idea es armar un sistema que haga dos cosas sobre reseñas de productos de Amazon: clasificar el sentimiento (negative / neutral / positive) y generar un resumen corto.

Lo construimos por etapas, probando arquitecturas cada vez más complejas para medir qué aporta cada una.

---

## Lo que hay acá

| Archivo | Contenido |
|---|---|
| `1-eda.ipynb` | EDA, limpieza y split estratificado 70/15/15 |
| `2-mlp.ipynb` | MLP + TF-IDF (baseline) |
| `3-1d-cnn-.ipynb` | 1D-CNN con embeddings aprendidos |
| `4-transformers-bert.ipynb` | RoBERTa zero-shot |
| `5-componente-generativo-resumenes.ipynb` | T5-small zero-shot para summarization |
| `6a-fine-tuning.ipynb` | Fine-tuning parcial de T5-small |
| `6b-despliegue-gradio.ipynb` | Demo web con Gradio |
| `7-comparativa-final.ipynb` | Consolidación de métricas de todo el proyecto |
| `Informe_Proyecto_Final_DL.pdf` | Documento técnico (16 páginas) |
| `Presentacion_Proyecto_Final_DL.pptx` | Presentación para la defensa |

---

## Dataset

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) publicado por Stanford SNAP. 568 454 reseñas recolectadas entre 1999 y 2012. Lo mapeamos a tres clases a partir del score: 1-2 → negative, 3 → neutral, 4-5 → positive.

El dataset es fuertemente desbalanceado (78% positive), así que reportamos F1 macro además de accuracy para no escondernos detrás de un número inflado.

---

## Cómo reproducir

Los notebooks están pensados para correr en **Kaggle con GPU T4**. No en local sin GPU — NB4 (inferencia RoBERTa sobre 568k reseñas) tarda unas 3 horas, y NB6a (fine-tuning T5) otro tanto.

### Pasos

1. Crea un dataset en Kaggle con `Reviews.csv` (lo encuentras en el link de arriba). Nuestro slug fue `reviews`.
2. Sube los notebooks a Kaggle en orden. Respeta los nombres de archivo — los notebooks posteriores referencian a los anteriores por slug.
3. En cada notebook: **Settings → Accelerator → GPU T4** + **Add data** con los inputs que correspondan:
   - NB1: dataset `reviews`.
   - NB2/3/4: el notebook `1-eda`.
   - NB5: dataset `reviews`.
   - NB6a: dataset `reviews`.
   - NB6b: notebook `6a-fine-tuning`.
   - NB7: notebooks `2-mlp-copy3`, `3-1d-cnn-copy-2`, `4-integraci-n-...`, `5-componente-...`, `6a-fine-tuning`.
4. Ejecuta en orden (NB1 → NB2 → ... → NB7). Después de cada uno, **Save Version → Quick Save** para que los CSVs queden disponibles como input del siguiente.

Si usas otro username en Kaggle, hay que ajustar los paths que apuntan a `/kaggle/input/notebooks/rodrigolopez29/...` en los notebooks posteriores.

### Tiempos aproximados en T4

- NB1, NB2, NB5, NB6b, NB7: minutos.
- NB3: ~5 min.
- NB4: ~3 horas (inferencia RoBERTa sobre todos los splits).
- NB6a: ~75-90 min (fine-tuning T5 1 epoch).

---

## Resultados

**Clasificación (TEST, 85 268 reseñas):**

| Métrica | MLP | 1D-CNN | RoBERTa zero-shot |
|---|---|---|---|
| Accuracy | 0.8804 | **0.8954** | 0.8344 |
| F1 macro | **0.7408** | 0.7355 | 0.6166 |
| Precision macro | 0.7197 | **0.7540** | 0.6157 |
| Recall macro | **0.7665** | 0.7230 | 0.6182 |

no hay un ganador absoluto entre MLP y CNN. La CNN gana accuracy y precision; el MLP gana recall y F1 macro (mejor equilibrio entre clases). El Transformer zero-shot queda por debajo — lo atribuimos a *domain shift* (el modelo fue preentrenado sobre tweets, no reseñas de comida).

**Generativo (ROUGE, 500 muestras del TEST):**

| Métrica | T5 zero-shot | T5 fine-tuned parcial | Δ |
|---|---|---|---|
| ROUGE-1 | 0.0849 | 0.1359 | +60% |
| ROUGE-2 | 0.0232 | 0.0469 | +102% |
| ROUGE-L | 0.0783 | 0.1346 | +72% |

Con un fine-tuning parcial limitado (solo ~41% de los parámetros entrenables, 1 sola epoch) la mejora es grande. La lección — que confirma lo que vimos con RoBERTa — es que los modelos preentrenados necesitan adaptación al dominio para rendir.

---

## Notas sobre limitaciones

- La clase **neutral** sigue siendo la más difícil en todas las etapas (F1 ≈ 0.45-0.50). Estrategias como oversampling con SMOTE o data augmentation con VAE quedan como trabajo futuro.
- El fine-tuning se limitó a 1 epoch por la cuota de GPU gratuita. Con 3-5 epochs es esperable ganar más.
- ROUGE mide solapamiento léxico, no semántico. Un resumen con sinónimos distintos al ground truth es penalizado aunque sea correcto. BERTScore daría una imagen más completa.
- El sistema funciona sobre reseñas en inglés. Trasladarlo a otro idioma requiere re-entrenar.

---

## Autores

Equipo Don Bosco 2026:

- Alfredo Argueta Interiano — AI252944
- Giovanni Alexander Escobar — EM252920
- Ivo Luis Orellana Girón — OG252913
- Marlon Alexander Palacios Díaz — PD252876
- José Rodrigo López Torres — LT170438

---



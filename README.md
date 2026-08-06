# audio-intensity-lab

Subproyecto de la segunda etapa, orientado al análisis de audio mediante DSP y aprendizaje supervisado.

Clasifica pistas musicales en categorías de energía y ritmo a partir de features extraídas con `librosa` (onset density, onset strength, tempo con plegado de octava, centroide y planitud espectral, ZCR, ratio percusivo vía HPSS y rango de loudness EBU R128), agregadas por segmento y evaluadas contra un conjunto de 119 canciones etiquetadas manualmente.

## Estado

En desarrollo desde 2026-06-28. La PoC inicial (fórmula lineal ponderada con umbrales calibrados) está completa y documentada. Actualmente en fase de benchmark comparativo entre las técnicas residuales de esa PoC —clasificadores supervisados sobre las features DSP (árbol de decisión, Random Forest) y clasificación zero-shot con CLAP (`laion/clap-htsat-unfused`)— con el fin de justificar cuantitativamente la migración a redes prototípicas (few-shot) sobre embeddings de MERT, conservando el pipeline DSP como extractor de la señal de tendencia temporal.

**Objetivo de la migración:** eliminar la dependencia de calibración global por género y permitir que las categorías sean definidas por el usuario a partir de un número mínimo de ejemplos, en lugar de estar fijadas en código.

## Benchmark

Se aplica paralelismo a nivel de archivo de audio mediante workers para distribuir el procesamiento.

Dentro de la extracción por segmento, la separación armónico-percusiva (`librosa.effects.hpss`, que alimenta la feature `perc_ratio`) concentra prácticamente todo el costo: sobre el dataset probado de 207 canciones / 1676 segmentos, HPSS acumula ~28 de los ~30 minutos totales. Es una operación de filtrado por mediana sobre la matriz STFT completa, puramente CPU-bound y sin uso de GPU.

La evaluación del flujo por etapa, variando la cantidad de workers en paralelo, reveló el siguiente problema crítico. La muestra se conformó mediante una selección aleatoria de audios, escalando progresivamente la cantidad de segmentos por canción desde 3 hasta 18, con incrementos unitarios y un último salto de 14 a 18 segmentos.


| Stage      | Total Execution Time (s) | Avg. HPSS Time (s) | Avg. Margin (s) | Overhead Ratio (%) | Total Execution (s) | Time per Audio |
| ---------- | ------------------------ | ------------------ | --------------- | ------------------ | ------------------- | -------------- |
| workers_16 | 1285.92                  | 45.66              | 34.71           | 76.04              | 119.90              | 10.72          |
| workers_3  | 260.57                   | 12.04              | 4.25            | 35.30              | 104.00              | 2.51           |
| workers_1  | 205.31                   | 10.39              | 2.44            | 23.51              | 205.31              | 1.00           |


3 workers es el óptimo, y 16 workers es peor que 3. La prueba de que es contención y no otra cosa está en la columna `avg_hpss_time_s`: la *misma* operación de HPSS pasa de 10.4s a 45.7s por tarea — el trabajo no cambió, cambió cuánto tiene que pelear cada proceso por el CPU. Con 16 procesos, la máquina termina haciendo **6.26× el trabajo total** (factor de degradación de eficiencia) del baseline para entregar un resultado *peor* que con 3.

El `avg_margin_s` cuenta la misma historia desde otro ángulo: el overhead pasa de 23% del tiempo de HPSS a 76%. Con 16 workers, tres cuartas partes del esfuerzo se va en coordinación y competencia, no en procesar audio.

# Optimización y eliminación de HPSS del pipeline de extracción

La característica `perc_ratio`, que representa el **ratio de energía percusiva obtenido mediante separación armónico-percusiva (HPSS)**, era calculada utilizando `librosa.effects.hpss` dentro del archivo `dsp/extract_features.py`.

Esta métrica formaba parte del cálculo del **eje de ritmo**, junto con las características normalizadas `onset_n` y `tempo_n`:

```python
score_ritmo = 0.45 * onset_n + 0.35 * perc_n + 0.20 * tempo_n
```

- `onset_n`: intensidad de los ataques (onsets), normalizada.
- `perc_n`: energía percusiva normalizada, derivada de `perc_ratio`.
- `tempo_n`: tempo normalizado.

Tras el análisis previo sobre el dataset inicial, se confirmó la hipótesis inicial: el principal inconveniente de `perc_ratio` **no era la calidad de la información que aportaba**, sino **el costo computacional** asociado al proceso de separación armónico-percusiva (HPSS) que consumía **el 93 % del tiempo de extracción**. Como cada iteración de calibración exige re-extraer el dataset completo, ese costo determinaba directamente cuántos experimentos podían correrse por día. Reducirlo era el mayor cuello de botella del ciclo de trabajo.

## 2. Hipótesis: reducir el tamaño del kernel

`librosa.decompose.hpss` aplica filtrado por mediana sobre la matriz STFT. El parámetro `kernel_size` (por defecto **31**) controla el largo de ese filtro y es el principal driver del costo computacional.

Se planteó reemplazar:

```python
# Actual — waveform in, waveform out (STFT + decompose + 2× ISTFT)
yh, yp = librosa.effects.hpss(seg)
perc_ratio = np.sum(yp**2) / (np.sum(yh**2) + np.sum(yp**2))
```

por:

```python
# Propuesto — sobre el espectrograma, kernel reducido
S = np.abs(librosa.stft(seg, n_fft=2048))
H, P = librosa.decompose.hpss(S, kernel_size=9)
perc_ratio = np.sum(P**2) / (np.sum(H**2) + np.sum(P**2))
```



## 3. Validación sobre dataset

Umbral de aceptación: Spearman ≥ 0.98, desvío medio ≤ 2 %, desvío máximo ≤ 10 %.

### 3.1 Resultados


| kernel | Spearman | Desvío medio | Desvío máx. | Segmentos con desvío > 10 % | Sesgo   |
| ------ | -------- | ------------ | ----------- | --------------------------- | ------- |
| 17     | 0.918    | 45.5 %       | 103.6 %     | 151/159 (95 %)              | +44.5 % |
| 11     | 0.870    | 55.4 %       | 153.7 %     | 151/159 (95 %)              | +55.0 % |
| 9      | 0.861    | 57.3 %       | 189.6 %     | 147/159 (93 %)              | +56.9 % |


**Ninguna variante pasó el umbral.** Incluso el kernel más conservador (17) produjo un desvío medio de 45.5 %.

### 3.4 Anomalía detectada: causa residual además del kernel

La relación kernel → desvío es monótona y consistente, lo que confirma que el kernel es *un* driver. Sin embargo, al extrapolar linealmente hasta k=31 el desvío predicho es **+24 %**, cuando debería ser ~0 % (mismo kernel en ambos lados de la comparación).

```
desvío ≈ -1.52 kernel + 71.4  →  predicción en k=31: +24.4 %
```

Ese residuo indica una **segunda diferencia** entre las corridas comparadas, además del kernel. 

## 4. Cambio de enfoque

Ante el fracaso de la optimización, se reformuló la pregunta: en lugar de *cómo calcular* `perc_ratio` *más barato*, **si** `perc_ratio` **vale lo que cuesta**.

Al ampliar el dataset y balancear las clases minoritarias (128 canciones, mínimo 15 por clase), la importancia de `perc_n_mediana` en el Random Forest cayó.


| Feature          | Importancia (128 canciones) |
| ---------------- | --------------------------- |
| `flat_n_mediana` | **0.174** (#1)              |
| `zcr_n_mediana`  | 0.170 (#2)                  |
| `cent_n_mediana` | 0.135 (#3)                  |
| `perc_n_mediana` | 0.106 (#4)                  |


Las tres features de timbre suman **0.479** de la importancia total — casi la mitad — mientras que `perc_ratio` había perdido peso relativo.

### 4.1 Ablation test

Se eliminó `perc_n_mediana` de la lista `FEATURES` y se repitió la validación cruzada (5-fold estratificado, `balanced_accuracy`, `class_weight="balanced"`). Sin re-extracción: mismos datos, una columna menos.


| Modelo            | Con `perc_ratio` | Sin `perc_ratio` | Δ        |
| ----------------- | ---------------- | ---------------- | -------- |
| Árbol de decisión | 39.4 % ± 6.6     | 41.8 % ± 8.5     | **+2.4** |
| **Random Forest** | **51.5 % ± 5.0** | **48.5 % ± 3.1** | **−3.0** |


Intervalos (media ± 2 · error estándar, k=5 folds):

- Con `perc_ratio`: **[47.0 %, 56.0 %]**
- Sin `perc_ratio`: **[45.7 %, 51.3 %]**

**Los intervalos se solapan** → la diferencia de 3.0 puntos no es distinguible del ruido de muestreo.

## 5. Decisión

**Se elimina HPSS y la feature** `perc_ratio` **del pipeline de extracción.**


| Dimensión              | Impacto                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------- |
| Costo eliminado        | ~28 min de 30 min por corrida (**93 %**)                                              |
| Costo en accuracy      | −3.0 puntos, **dentro del ruido** (intervalos solapados)                              |
| Estabilidad del modelo | Desviación **mejora**: ±5.0 → ±3.1                                                    |
| Árbol de decisión      | **Mejora** +2.4 puntos al eliminarla                                                  |
| Complejidad eliminada  | `PERC_LO`/`PERC_HI`, debate de `kernel_size`, dependencia de `effects` vs `decompose` |


El eje de ritmo pasa a calcularse sin el componente percusivo:

```python
# Después — pesos a redistribuir/recalibrar
score_ritmo = 0.69 * onset_n + 0.31 * tempo_n
```



## 6. Resultado y estado posterior


| Métrica                                   | Antes                       | Después      |
| ----------------------------------------- | --------------------------- | ------------ |
| Tiempo de extracción (207 canciones)      | ~30 min                     | ~2 min       |
| Random Forest (balanced accuracy, 5-fold) | 51.5 % ± 5.0                | 48.5 % ± 3.1 |
| Features en el modelo                     | 13                          | 12           |
| Constantes de calibración                 | incluye `PERC_LO`/`PERC_HI` | eliminadas   |


El ciclo de iteración pasa de ~30 min a ~2 min, habilitando experimentación rápida sobre segmentación, `SR` y recalibraciones que antes costaban media hora por prueba.
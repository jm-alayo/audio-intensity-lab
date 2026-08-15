# audio-intensity-lab

Análisis y benchmark de clasificación musical mediante técnicas de DSP y ML supervisado.

PoC incial realizada bajo 4 técnicas implementadas: formula lineal ponderada con umbrales calibrados, árbol de decisión, Random Forest y clasificación zero-shot con CLAP. 

Sin alcance actual del ~70-80% de balanced accuracy, evidencia que motiva la migración a redes prototípicas (few-shot) sobre embeddings de MERT, ademas de eliminar la calibración global por género y permitir que el usuario defina categorías con pocos ejemplos, reemplazando las fijadas en código. 

Repositorio complementario: ++**[async-music-fetch-engine](https://github.com/jm-alayo/async-music-fetch-engine)**++

## Benchmark

Se utilizó un dataset compuesto por 96 canciones para el todo el reporte, distribuidas en un total de 5 clases:

- **alta-agresiva:** 16 canciones
- **alta-ritmica:** 27 canciones
- **baja-contemplativa:** 9 canciones **(clase minoritaria)**
- **baja-ritmica:** 21 canciones
- **incrementable-decreciente:** 23 canciones

Dentro de la extracción por segmento `features\extract.py`, la separación armónico-percusiva (`librosa.effects.hpss`, que alimenta la feature `perc_ratio`, procesamiento estrictamente CPU-bound) concentra prácticamente todo el costo: sobre el dataset real de 96 canciones / 890 segmentos, HPSS acumula el 83–87 % del tiempo de extracción por segmento (`workers=1`).

La evaluación se realizó sobre **28 de las 96 canciones**, estratificadas por `seg_N` (`_features_2_rock_english_segmin25.csv`). Se seleccionaron `RANDOM_SELECTION=2` canciones aleatorias para cada una de las 15 cantidades de segmentos presentes (4–20), o una si el grupo tenía una sola, usando `RANDOM_SEED=42` para reproducibilidad. Selección registrada en `evidence/phase_a_workers/sample_selection.csv`.

- `HPSS/margen prom. por canción`: tiempo de CPU acumulado de todos los segmentos de cada canción.
- `Reloj de pared total`: tiempo real de las 28 canciones con esa cantidad de workers.
- `Tiempo real por canción`: `reloj de pared total ÷ 28 canciones`.
- `Paralelismo efectivo`: `tiempo de CPU acumulado ÷ reloj de pared`; aceleración efectiva obtenida mediante el procesamiento en paralelo (RTF invertido).
- Antes de medir se hace un *warm-up*: se cargan las 28 canciones una vez, sin cronometrar, para que las tres configuraciones arranquen con el mismo estado de caché de disco (ver observación más abajo).


| Etapa      | Trabajo CPU acumulado (s) | HPSS prom./canción (s) | Margen prom./canción (s) | Overhead (%) | Reloj de pared total (s) | Paralelismo efectivo (s) | Tiempo real por canción (s) |
| ---------- | ------------------------- | ---------------------- | ------------------------ | ------------ | ------------------------ | ------------------------ | --------------------------- |
| workers_16 | 1105.37                   | 29.24                  | 10.24                    | 35.02        | 103.48                   | 10.68                    | **3.70**                    |
| workers_3  | 467.38                    | 12.96                  | 3.73                     | 28.82        | 178.89                   | 2.61                     | **6.39**                    |
| workers_1  | 354.52                    | 10.97                  | 1.69                     | 15.37        | 379.62                   | 0.93                     | **13.56**                   |


Análisis: **16 workers es el más rápido por canción** (3.70 s), seguido de 3 (6.39 s) y 1 (13.56 s) — ningún salto perjudicial.

La contención aumenta con la cantidad de workers: el trabajo de HPSS alcanza **29.24 s de CPU acumulada**. El volumen de trabajo no cambia, pero aumenta la competencia por los recursos de CPU. Con 16 procesos, la máquina realiza **3.12× el trabajo total**, pero logra un mejor tiempo por canción: **3.66× más rápido que con 1 worker y 1.73× que con 3**.

El margen, correspondiente al tiempo no asociado a HPSS, representa un **35.02 % del tiempo de HPSS**, reflejando una mayor competencia por CPU a medida que aumentan los workers. Sin embargo, en este hardware, el overhead **no llega a superar la ganancia del paralelismo**.

> **Observación (control de sesgo de caché):** la tabla usa warm-up. Sin warm-up, los tiempos fueron **395.83 s** (`workers_1`, ~4 % más), **171.51 s** (`workers_3`, ~4 % menos) y **110.59 s** (`workers_16`, ~6 % más), respecto a los valores con warm-up. Estas variaciones corresponden al ruido de ejecución y no muestran un sesgo direccional: **16 workers sigue siendo la configuración más rápida con o sin warm-up**.

> **Advertencia (96 canciones):** con HPSS presente (tarea pesada), más workers gana y la contención de CPU no llega a comerse la ganancia del paralelismo. Sin HPSS (pipeline actual, tarea liviana), `N_WORKERS=3` **es el más rápido y 16 el más lento**, por el overhead de gestión de procesos e I/O sobre tareas cortas. `N_WORKERS=3` sigue siendo el default óptimo por su equilibrio en tareas livianas, no por saturación de CPU.
>
> Comparación con 28 canciones con y sin HPSS: `research\hpss_replication_96\evidence\phase_a_workers\comparison_with_vs_without.log`
>
> *Evidencia:* `evidence/phase_a_workers/times_with_hpss.csv`, `evidence/phase_a_workers/summary_with_hpss.csv`.



### Tiempo real de producción (sin HPSS)

`perc_ratio` ya no se calcula en el pipeline actual. Ejecutando `features/extract.py` sin HPSS sobre las **96 canciones completas**, con el detalle por canción guardado en `evidence/validation_full_scale/times_without_hpss_96.csv`


| Etapa | Trabajo CPU acumulado (s) | Reloj de pared total (s) | Paralelismo efectivo (s) | Duración promedio de procesamiento (s) |
| ----- | ------------------------- | ------------------------ | ------------------------ | -------------------------------------- |
| 1     | 195.79                    | 214.8 (~4 min)           | 0.91                     | 2.24                                   |
| 3     | 748.90                    | 261.1 (~4 min)           | 2.87                     | 2.72                                   |
| 16    | 4698.33                   | 309.0 (~5 min)           | 15.20                    | 3.22                                   |


> **Observación (variación entre corridas):** `workers_3` y `workers_16` empeoraron minimamente sin cambios de código, debido a variación de carga del sistema. Con ambas muestras, `N_WORKERS=1` nunca fue el peor y fue el más rápido en la última corrida; `N_WORKERS=3` sigue siendo un default razonable, pero **no hay un ganador estable entre 1 y 3** para la tarea liviana actual.

Para validar la extrapolación por muestra (28→96), se restauró HPSS real temporalmente (kernel 31, sin variantes ni Spearman) y se corrió la extracción completa de las **96 canciones**, `N_WORKERS=16` (luego revertido: sin HPSS, `N_WORKERS=3`; quedan solo los CSVs de las 2 corridas en `data/features/`).


| Corrida (con HPSS, `N_WORKERS=16`) | CPU acumulada (s) | Reloj de pared (s) | Paralelismo (×) |
| ---------------------------------- | ----------------- | ------------------ | --------------- |
| Sin warm-up                        | 8276.42           | 545.9 (9.10 min)   | 15.16           |
| Con warm-up                        | 7383.68           | 484.1 (8.07 min)   | 15.25           |


*Evidencia:* `evidence/validation_full_scale/times_with_hpss_restored_96.csv`, `evidence/validation_full_scale/extrapolation_stability.csv`.

> **Observación (la extrapolación por muestra es inestable con HPSS):** la muestra de 28 da 3.95 s/segmento → escalado a 890 segmentos = 3519 s de estimación ingenua. Corrigiendo por el aumento real observado por canción (+52.68 % promedio, consistente en las 28 filas) da 5373 s — pero lo real fue **7384-8276 s (+37 a +54 % sobre esa estimación ya corregida)**. Causa: con HPSS los 16 workers se sostienen saturados ~8-9 min en el lote completo vs. un par de minutos en la muestra, y esa carga sostenida degrada el rendimiento por canción de forma no lineal (throttling/contención) — algo que una muestra corta no captura. La tabla "sin HPSS" de arriba, en cambio, sí extrapola de forma estable (tarea liviana, nunca satura la CPU el tiempo suficiente).



# Optimización y eliminación de HPSS del pipeline de extracción

`perc_ratio`, el **ratio de energía percusiva obtenido mediante HPSS**, se calculaba con `librosa.effects.hpss` en `dsp/extract_features.py`. Esta métrica integraba el **eje de ritmo** junto con `onset_n` y `tempo_n`:

```python
score_ritmo = 0.45 * onset_n + 0.35 * perc_n + 0.20 * tempo_n
```

- `onset_n`: intensidad de los ataques (onsets), normalizada.
- `perc_n`: energía percusiva normalizada, derivada de `perc_ratio`.
- `tempo_n`: tempo normalizado.

El problema de `perc_ratio` no era la calidad de la señal, sino el costo de HPSS: **82-86 %** del tiempo de extracción sobre el dataset real de 96 canciones (`workers=3`/`workers=1`, `evidence/phase_a_workers/times_with_hpss.csv`) — y, como muestra "Tiempo real de producción" en el Benchmark más arriba, un costo **inestable al escalar** (la muestra de 28 canciones subestimaba el costo real a 96 en 37-54 %, incluso ya corregida). Costo alto e impredecible, sumado a que cada iteración de calibración exige re-extraer el dataset completo.

## Reducción del kernel (evaluación opcional)

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

Umbral de aceptación: Spearman ≥ 0.98, desvío medio ≤ 2 %, desvío máximo ≤ 10 %.

Resultados sobre las 96 canciones / 890 segmentos


| kernel | Spearman | Desvío medio | Desvío máx. | Segmentos con desvío > 10 % | Sesgo   |
| ------ | -------- | ------------ | ----------- | --------------------------- | ------- |
| 17     | 0.959    | 18.8 %       | 167.3 %     | 546/890 (61 %)              | +16.9 % |
| 11     | 0.912    | 32.2 %       | 280.9 %     | 676/890 (76 %)              | +30.2 % |
| 9      | 0.888    | 37.5 %       | 353.0 %     | 693/890 (78 %)              | +35.3 % |


**Ninguna variante pasó el umbral.** Incluso el kernel más conservador (17) produjo un desvío medio de 18.8 %, ~9× por encima del umbral aceptable (2 %).

## Validación de metodología: cálculo en una sola pasada

`perc_ratio_k31`, `perc_ratio_k17`, `perc_ratio_k11` y `perc_ratio_k9` se calculan en la misma pasada, sobre el mismo segmento cargado una sola vez, evitando comparar entre corridas o versiones de pipeline distintas — así cualquier diferencia medida es atribuible únicamente al kernel. Con esa metodología, el sesgo con signo por kernel (`evidence/phase_b_kernel_validation/perc_ratio_per_segment.csv`) es +16.93 % (k=17), +30.23 % (k=11) y +35.28 % (k=9); el ajuste lineal de esos 3 puntos da **desvío ≈ -2.276·kernel + 55.55 → -15.0 % en k=31** — cercano a 0 % como corresponde (mismo kernel a ambos lados de la comparación en ese punto), dentro de la imprecisión esperable al extrapolar en línea recta desde k≤17 hasta k=31.

**Confirmado con una segunda fuente independiente:** el mismo cálculo se corrió también dentro de `features/extract.py` real (no el script de investigación), en paralelo con `N_WORKERS=16` sobre las 96 canciones completas — mismos 890 segmentos, resultado idéntico número por número (`evidence/phase_b_kernel_validation/perc_ratio_per_segment_from_extract.csv`). No es una coincidencia de un solo script: el pipeline de producción corriendo en paralelo reproduce exactamente el mismo Spearman/desvío/sesgo por kernel que el script de investigación.

## Reevaluación de `perc_ratio`

Ante el fracaso de la optimización, se reformuló la pregunta: en lugar de *cómo calcular* `perc_ratio` *más barato*, **si** `perc_ratio` **vale lo que cuesta**.

Recalculando `perc_ratio` real (kernel 31) para las 96 canciones del dataset actual y reincorporándola como feature (`perc_n_mediana`, con `PERC_LO`/`PERC_HI` derivados por percentiles p5/p95 del propio dataset — no a mano, ver `evidence/phase_c_ablation/perc_ratio_bounds.csv`), la importancia en el Random Forest queda:


| Feature          | Importancia (96 canciones) |
| ---------------- | -------------------------- |
| `cent_n_mediana` | **0.154** (#1)             |
| `perc_n_mediana` | 0.153 (#2)                 |
| `flat_n_mediana` | 0.141 (#3)                 |
| `zcr_n_mediana`  | 0.139 (#4)                 |


Las tres features de timbre (`cent`, `flat`, `zcr`) suman **0.434** de la importancia total — un poco menos de la mitad.

## Prueba de ablación: impacto de `perc_ratio` en accuracy

Se agregó `perc_n_mediana` a la lista `FEATURES` y se repitió la validación cruzada (5-fold estratificado, `balanced_accuracy`, `class_weight="balanced"`), sobre las mismas 96 canciones etiquetadas, con y sin la columna.


| Modelo            | Con `perc_ratio`  | Sin `perc_ratio`  | Δ         |
| ----------------- | ----------------- | ----------------- | --------- |
| Árbol de decisión | 47.3 % ± 15.9     | 37.2 % ± 13.1     | **-10.1** |
| **Random Forest** | **52.5 % ± 13.6** | **48.9 % ± 13.3** | **-3.6**  |


Intervalos (media ± 2 · error estándar, k=5 folds):

- Árbol de decisión — con: **[15.5 %, 79.0 %]** · sin: **[11.1 %, 63.3 %]**
- Random Forest — con: **[25.3 %, 79.6 %]** · sin: **[22.3 %, 75.6 %]**

**Los intervalos se solapan** en ambos modelos: la diferencia no es estadísticamente distinguible del ruido de muestreo (folds de ~19 canciones sobre 96).

## Decisión

**Se elimina HPSS y la feature** `perc_ratio` **del pipeline de extracción.**


| Dimensión             | Impacto                                                                               |
| --------------------- | ------------------------------------------------------------------------------------- |
| Costo eliminado       | 82-87 % del tiempo de extracción por segmento (`workers=1`/`workers=3`)               |
| Costo en accuracy     | Random Forest: −3.6 puntos, dentro del ruido (intervalos solapados)                   |
| Árbol de decisión     | Empeora −10.1 puntos al eliminarla                                                    |
| Complejidad eliminada | `PERC_LO`/`PERC_HI`, debate de `kernel_size`, dependencia de `effects` vs `decompose` |


El eje de ritmo pasa a calcularse sin el componente percusivo:

```python
score_ritmo = 0.69 * onset_n + 0.31 * tempo_n
```

> **Advertencia:** ambos modelos empeoran al remover `perc_ratio` y la feature quedó en el top 2 de importancia — la justificación estadística (intervalos solapados) sostiene la decisión, pero es débil; ameritaría revalidar con un dataset más grande. Del lado del costo la decisión es más sólida: con HPSS, una muestra de 28 canciones subestima el costo real a escala de 96 en 37-54 %, incluso corregida por su propio factor de error — ningún benchmark chico predice bien el costo real en producción, mientras que sin HPSS la extrapolación por muestra sí es estable.



## Resultado y estado posterior

El pipeline de producción corre sin HPSS: extracción de las 96 canciones en ~3.4 minutos (`N_WORKERS=3`), Random Forest en 48.9 % ± 13.3 de balanced accuracy (5-fold), 12 features en el modelo, sin constantes de calibración de `perc_ratio`.

## Benchmark de DSP

El árbol de decisión da 37.2 % ± 13.1 de balanced accuracy, 11.7 puntos por debajo del Random Forest. La regla lineal (49.7 %) y el acuerdo final del router de confianza (74.0 %) se reportan por completitud del pipeline, pero se miden sobre el conjunto de entrenamiento completo, no con validación cruzada — no son comparables contra el 48.9 % del Random Forest. La desviación amplia en ambos modelos con CV (±13.3, ±13.1) es consistente con el tamaño de muestra: `baja-contemplativa` (n=9) reparte solo ~2 ejemplos por fold de los 5.

Importancia de features en el modelo de producción (sin `perc_ratio`): `zcr_n_mediana` (0.179), `cent_n_mediana` (0.177) y `flat_n_mediana` (0.174) concentran 0.530 de la importancia total — más de la mitad del poder de decisión del modelo, repartido casi por igual entre las tres.

Matriz de confusión (predicho × humano) y desempeño por categoría:

```
                           alta-agresiva  alta-ritmica  baja-contemplativa  baja-ritmica  incrementable-decreciente
predicted \ human
alta-agresiva                         12             4                   0             1                          2
alta-ritmica                           2            19                   0             2                          1
baja-contemplativa                     1             1                   9             3                          2
baja-ritmica                           1             0                   0            13                          0
incrementable-decreciente              0             3                   0             2                         18
```


| Categoría                   | n   | Recall  | Precisión | F1     |
| --------------------------- | --- | ------- | --------- | ------ |
| `alta-agresiva`             | 16  | 75.0 %  | 63.2 %    | 68.6 % |
| `alta-ritmica`              | 27  | 70.4 %  | 79.2 %    | 74.5 % |
| `baja-contemplativa`        | 9   | 100.0 % | 56.2 %    | 72.0 % |
| `baja-ritmica`              | 21  | 61.9 %  | 92.9 %    | 74.3 % |
| `incrementable-decreciente` | 23  | 78.3 %  | 78.3 %    | 78.3 % |


`baja-contemplativa` tiene 100 % de recall pero solo 56.2 % de precisión — el modelo la usa como categoría "default" con más frecuencia de la que corresponde (16 predicciones sobre 9 casos reales). `baja-ritmica` es el patrón inverso: precisión alta (92.9 %) pero recall bajo (61.9 %).

Del router de confianza: 40.6 % de las canciones caen en `confiable`, 25.0 % en `ambigua`, 34.4 % en `revisar` — más de un tercio del dataset no alcanza el umbral de confianza mínimo del árbol y queda para desempate o revisión manual.

**Limitación conocida:** "10cc – I'm Not in Love" (balada suave, sin percusión marcada) obtuvo un `score_energia` más alto que "Marilyn Manson – Sweet Dreams (Are Made of This)" (rock industrial distorsionado) — 0.325 vs 0.257. No es un error de cálculo: `flatness`/`zcr` miden las docenas de capas vocales superpuestas de 10cc como ruido de banda ancha, indistinguible de la distorsión real de Manson. Como estas tres features de timbre concentran el 53 % de la importancia del modelo, esto es una limitación estructural del enfoque DSP — uno de los argumentos centrales para migrar a embeddings pre-entrenados (MERT) en la siguiente fase.
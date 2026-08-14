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

La evaluación se realizó sobre **28 de las 96 canciones**, estratificadas por `seg_N` (`_features_2_rock_english_segmin25.csv`). Se seleccionaron `RANDOM_SELECTION=2` canciones aleatorias para cada una de las 15 cantidades de segmentos presentes (4–20), o una si el grupo tenía una sola, usando `RANDOM_SEED=42` para reproducibilidad. Selección registrada en `evidence/worker_sample_selection_96.csv`.

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

> **Advertencia (96 canciones):** Esta sección invierte la conclusión del documento original sobre el benchmark de workers. La causa no es el hardware sino la carga de trabajo:
>
> - **Con HPSS (tarea pesada):** Provoca contención de CPU.
> - **Sin HPSS (pipeline actual / tarea liviana):** `N_WORKERS=3` **es el más rápido y 16 el más lento** debido al *overhead* de gestión de procesos e I/O sobre tareas cortas.
>
> `N_WORKERS=3` sigue siendo óptimo por su equilibrio en tareas livianas, no por saturación de CPU.
>
> *Evidencia:* `evidence/benchmark_worker_times_96.csv`, `evidence/worker_summary_96.csv`.

### Tiempo real de producción (sin HPSS)

El benchmark reincorpora HPSS para compararlo con el documento original, aunque `perc_ratio` ya no se calcula en el pipeline actual. Ejecutando `features/extract.py` sin HPSS sobre las **96 canciones completas**, con el detalle por canción guardado en `evidence/benchmark_worker_times_no_hpss.csv`


| Etapa | Trabajo CPU acumulado (s) | Reloj de pared total (s) | Paralelismo efectivo (s) | Duración promedio de procesamiento (s) |
| ----- | ------------------------- | ------------------------ | ------------------------ | -------------------------------------- |
| 1     | 195.79                    | 214.8 (~4 min)           | 0.91                     | 2.04                                   |
| 3     | 748.90                    | 261.1 (~4 min)           | 2.87                     | 7.80                                   |
| 16    | 4698.33                   | 309.0 (~5 min)           | 15.20                    | 48.94                                  |


> **Observación (variación entre corridas):** esta tabla sustituye una corrida anterior donde `workers_3` fue el más rápido. `workers_1` se mantuvo estable (~215–221 s), mientras `workers_3` y `workers_16` empeoraron sin cambios de código, debido a variación de carga del sistema. Con ambas muestras, `N_WORKERS=1` nunca fue el peor y fue el más rápido en la última corrida; `N_WORKERS=3` sigue siendo un default razonable, pero **no hay un ganador estable entre 1 y 3** para la tarea liviana actual.

---

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

> **Observación (dataset real, 96 canciones):** midiendo HPSS real sin contención (`workers=1`) sobre la muestra actual, la fracción de tiempo que ocupa HPSS es **86.4 %** del costo por segmento (11.41 s de HPSS contra 1.79 s del resto de las features); a `N_WORKERS=3` (config actual de producción) baja a 82.0 % por el paralelismo. Mismo orden de magnitud que el 93 % original — la conclusión de que HPSS domina el costo de extracción se sostiene. Evidencia: `evidence/benchmark_worker_times_96.csv`.



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

Sobre las 96 canciones reales / 890 segmentos (todo el dataset actual, no una submuestra):


| kernel | Spearman | Desvío medio | Desvío máx. | Segmentos con desvío > 10 % | Sesgo   |
| ------ | -------- | ------------ | ----------- | --------------------------- | ------- |
| 17     | 0.959    | 18.8 %       | 167.3 %     | 546/890 (61 %)              | +16.9 % |
| 11     | 0.912    | 32.2 %       | 280.9 %     | 676/890 (76 %)              | +30.2 % |
| 9      | 0.888    | 37.5 %       | 353.0 %     | 693/890 (78 %)              | +35.3 % |


**Ninguna variante pasó el umbral.** Incluso el kernel más conservador (17) produjo un desvío medio de 18.8 %.

> **Observación (dataset real, 96 canciones):** el veredicto es el mismo (ninguna variante aplicable), pero el desvío medio en kernel=17 es 18.8 %, muy por debajo del 45.5 % medido en el dataset original — sigue estando ~9× por encima del umbral aceptable (2 %), así que no cambia la decisión, pero la brecha real es menos extrema de lo que sugería la corrida anterior.



### 3.4 Anomalía: no se reproduce con una corrida limpia

En la corrida original, extrapolar linealmente la relación kernel → desvío hasta k=31 predecía un desvío de +24.4 % cuando debía dar ~0 % (mismo kernel a ambos lados de la comparación), lo cual apuntaba a una **segunda diferencia** entre las corridas comparadas además del kernel — sospecha registrada en `research/hpss_cost_analysis.py`, que compara dos CSVs de esquema distinto (`filename` vs `music_name`).

Acá `perc_ratio_k31`, `perc_ratio_k17`, `perc_ratio_k11` y `perc_ratio_k9` se calculan en la misma pasada, sobre el mismo segmento cargado una sola vez — sin comparar corridas ni versiones de pipeline distintas. Con esa metodología limpia, la extrapolación lineal a k=31 da **desvío ≈ -2.31·kernel + 58.03 → -13.7 % en k=31**: lejos del +24.4 % anómalo original, y del orden de magnitud correcto (cercano a 0 %, dada la imprecisión de extrapolar en línea recta desde k≤17 hasta k=31). Confirma la sospecha original: la anomalía era un artefacto de comparar corridas con una segunda diferencia metodológica de por medio, no una propiedad real de la relación kernel-desvío.

**Confirmado con una segunda fuente independiente:** el mismo cálculo se corrió también dentro de `features/extract.py` real (no el script de investigación), en paralelo con `N_WORKERS=16` sobre las 96 canciones completas — mismos 890 segmentos, resultado idéntico número por número (`evidence/perc_ratio_per_segment_from_extract.csv`). No es una coincidencia de un solo script: el pipeline de producción corriendo en paralelo reproduce exactamente el mismo Spearman/desvío/sesgo por kernel que el script de investigación.

## 4. Cambio de enfoque

Ante el fracaso de la optimización, se reformuló la pregunta: en lugar de *cómo calcular* `perc_ratio` *más barato*, **si** `perc_ratio` **vale lo que cuesta**.

Recalculando `perc_ratio` real (kernel 31) para las 96 canciones del dataset actual y reincorporándola como feature (`perc_n_mediana`, con `PERC_LO`/`PERC_HI` derivados por percentiles p5/p95 del propio dataset — no a mano, ver `evidence/perc_ratio_bounds.csv`), la importancia en el Random Forest queda:


| Feature          | Importancia (96 canciones) |
| ---------------- | -------------------------- |
| `cent_n_mediana` | **0.154** (#1)             |
| `perc_n_mediana` | 0.153 (#2)                 |
| `flat_n_mediana` | 0.141 (#3)                 |
| `zcr_n_mediana`  | 0.139 (#4)                 |


Las tres features de timbre (`cent`, `flat`, `zcr`) suman **0.434** de la importancia total — un poco menos de la mitad.

> **Observación (dataset real, 96 canciones) — cambia el sentido de esta sección:** en el dataset original, ampliar y balancear el dataset hacía **caer** el peso de `perc_n_mediana` a #4 (0.106), por detrás de las tres features de timbre. Acá pasa lo contrario: `perc_n_mediana` (0.153) queda prácticamente empatada con la feature más importante (0.154, #1), muy por encima de su lugar original. La idea de que "`perc_ratio` había perdido peso relativo" no se sostiene con el dataset real de 96 canciones — ahí es de las features más informativas del modelo.



### 4.1 Ablation test

Se agregó `perc_n_mediana` a la lista `FEATURES` y se repitió la validación cruzada (5-fold estratificado, `balanced_accuracy`, `class_weight="balanced"`), sobre las mismas 96 canciones etiquetadas, con y sin la columna.


| Modelo            | Con `perc_ratio`  | Sin `perc_ratio`  | Δ         |
| ----------------- | ----------------- | ----------------- | --------- |
| Árbol de decisión | 47.3 % ± 15.9     | 37.2 % ± 13.1     | **-10.1** |
| **Random Forest** | **52.5 % ± 13.6** | **48.9 % ± 13.3** | **-3.6**  |


Intervalos (media ± 2 · error estándar, k=5 folds):

- Árbol de decisión — con: **[15.5 %, 79.0 %]** · sin: **[11.1 %, 63.3 %]**
- Random Forest — con: **[25.3 %, 79.6 %]** · sin: **[22.3 %, 75.6 %]**

**Los intervalos se solapan** en ambos modelos → igual que en el dataset original, la diferencia no es estadísticamente distinguible del ruido de muestreo (con folds de ~19 canciones sobre 96, la varianza es incluso mayor que antes).

> **Observación (dataset real, 96 canciones) — cambia el sentido de esta sección:** estadísticamente la conclusión es la misma ("no distinguible del ruido"), pero la *dirección* cambió en el árbol de decisión: en el dataset original, sacar `perc_ratio` **mejoraba** el árbol +2.4 puntos; acá lo **empeora** -10.1 puntos, la mayor caída de las dos comparaciones. El Random Forest mantiene el mismo signo que el dataset original (remover perdiendo `perc_ratio` cuesta accuracy: -3.0 antes, -3.6 ahora), consistente en dirección aunque con intervalos más anchos por tener menos canciones.



## 5. Decisión

**Se elimina HPSS y la feature** `perc_ratio` **del pipeline de extracción.**


| Dimensión              | Impacto                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| Costo eliminado        | ~83–87 % del tiempo agregado de extracción por segmento (medido, `workers=1`/`workers=3`) |
| Costo en accuracy      | RF: −3.6 puntos, **dentro del ruido** (intervalos solapados)                              |
| Estabilidad del modelo | RF std: 13.6 % → 13.3 % — cambio marginal, no la mejora clara del dataset original        |
| Árbol de decisión      | **Empeora** −10.1 puntos al eliminarla                                                    |
| Complejidad eliminada  | `PERC_LO`/`PERC_HI`, debate de `kernel_size`, dependencia de `effects` vs `decompose`     |


El eje de ritmo pasa a calcularse sin el componente percusivo:

```python
# Después — pesos a redistribuir/recalibrar
score_ritmo = 0.69 * onset_n + 0.31 * tempo_n
```

> **Advertencia (dataset real, 96 canciones):** la justificación estadística de esta decisión (intervalos solapados) se mantiene, pero es más débil que en el dataset original. Ahí, el costo en accuracy era negativo solo en Random Forest y el Árbol de decisión mejoraba al remover — un panorama mixto que favorecía la eliminación. Acá, **ambos modelos empeoran al remover** `perc_ratio`, la caída en el árbol de decisión es 3-4 veces mayor, y la feature quedó empatada en el primer puesto de importancia (no en el cuarto). La decisión de eliminar HPSS sigue siendo defendible por costo computacional y porque los intervalos siguen solapados, pero si en algún momento se reconsidera el trade-off costo/accuracy, este es el dataset donde `perc_ratio` mostró más valor — ameritaría revalidar con un dataset más grande antes de descartarla definitivamente.



## 6. Resultado y estado posterior


| Métrica                                        | Antes (con HPSS)                                        | Después (sin HPSS, actual)                       |
| ---------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------ |
| Tiempo de extracción (96 canciones, 3 workers) | ~9.8 min (real, extrapolado de muestra de 28 canciones) | ~3.4 min (real, extrapolado de la misma muestra) |
| Random Forest (balanced accuracy, 5-fold)      | 52.5 % ± 13.6                                           | 48.9 % ± 13.3                                    |
| Features en el modelo                          | 13                                                      | 12                                               |
| Constantes de calibración                      | incluiría `PERC_LO`/`PERC_HI`                           | eliminadas                                       |


El ciclo de iteración pasa de ~9.8 min a ~3.4 min sobre las 96 canciones actuales a la configuración de producción (`N_WORKERS=3`).

> **Observación (dataset real, 96 canciones):** la reducción real (~65 %, 2.9×) es menor a la reportada originalmente (~93 %, 15×, sobre 207 canciones). Motivo principal: la cifra original comparaba contra una corrida efectivamente no paralela sobre un corpus mayor, mientras que acá "antes" y "después" se miden ambos a `N_WORKERS=3`, donde parte del costo de HPSS ya se absorbe en paralelo. Ver también la sección "Benchmark" más arriba — en este hardware, 16 workers resultó más rápido que 3, lo opuesto a la contención documentada en el dataset original.


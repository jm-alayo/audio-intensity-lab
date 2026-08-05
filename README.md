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

### Tabla del reporte


| Stage      | Total Execution Time (s) | Avg. HPSS Time (s) | Avg. Margin (s) | Overhead Ratio (%) | Total Execution (s) | Time per Audio |
| ---------- | ------------------------ | ------------------ | --------------- | ------------------ | ------------------- | -------------- |
| workers_16 | 1285.92                  | 45.66              | 34.71           | 76.04              | 119.90              | 10.72          |
| workers_3  | 260.57                   | 12.04              | 4.25            | 35.30              | 104.00              | 2.51           |
| workers_1  | 205.31                   | 10.39              | 2.44            | 23.51              | 205.31              | 1.00           |


3 workers es el óptimo, y 16 workers es peor que 3. La prueba de que es contención y no otra cosa está en la columna `avg_hpss_time_s`: la *misma* operación de HPSS pasa de 10.4s a 45.7s por tarea — el trabajo no cambió, cambió cuánto tiene que pelear cada proceso por el CPU. Con 16 procesos, la máquina termina haciendo **6.26× el trabajo total** (factor de degradación de eficiencia) del baseline para entregar un resultado *peor* que con 3.

El `avg_margin_s` cuenta la misma historia desde otro ángulo: el overhead pasa de 23% del tiempo de HPSS a 76%. Con 16 workers, tres cuartas partes del esfuerzo se va en coordinación y competencia, no en procesar audio.
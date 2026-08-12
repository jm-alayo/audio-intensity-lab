# Arquitectura — audio-intensity-lab (music-tagger-benchmark)

> Refactor aplicado 2026-08-11 (rama `devalayo`). Reemplaza la estructura anterior (`dsp/` + `dsp/llm-clap/`) descrita hasta ahora en el blueprint del proyecto.
> Historia previa al refactor: `devlog-blueprints/AG_NOTES_MATHIAS/A_2026/datalab-music-mood/MDs_workspace/TAGGER_MUSIC_GENESIS.md`.

## Resumen ejecutivo

El proyecto clasifica canciones en 5 categorías de energía/ritmo (`alta-agresiva`, `alta-ritmica`, `baja-contemplativa`, `baja-ritmica`, `incrementable-decreciente`) a partir de features acústicas extraídas con `librosa`. La arquitectura tiene tres capas con responsabilidad única:

- **`features/`** — extracción de features por canción (DSP puro, sin nada de modelos).
- **`ml/`** — todo el machine learning del proyecto: dos clasificadores (Random Forest/árbol y una regla lineal calibrada automáticamente), la capa de confianza que decide qué canciones necesitan revisión humana, y la evaluación (cross-validation, learning curve, matriz de confusión).
- **`shared/`** — rutas del proyecto y utilidades genéricas de I/O/estadística, sin lógica de dominio.

**Route B (CLAP zero-shot) fue eliminada del proyecto** — dejó de compararse contra el enfoque DSP+ML. La resolución de casos ambiguos ya no depende de un modelo externo: se resuelve por desempate entre los dos modelos internos (Random Forest vs regla lineal).

El pipeline de descarga/normalización de MP3 (`async-music-fetch-engine/`) es un proyecto hermano fuera de este repo — `music-tagger-benchmark/` solo consume los archivos ya descargados desde `PLAYLIST_BASE`.

## Diagrama de flujo

```
data/audio/_canciones_rock_english.xlsx (hoja "data": music_name, categorico, album)
                    │
                    ▼
        shared/utils.py: load_tracks()
                    │  filtra categorico != "sin_categoria" AND album == "album-rock"
                    ▼
        features/extract.py  ── lee mp3 reales desde PLAYLIST_BASE
        (multiprocessing.Pool, librosa: onset/tempo/centroide/flatness/zcr/RMS, LRA EBU R128)
        constantes de extracción (SR, ONSET_GATE_PCT, bounds de normalización) en features/config.py
                    │  escribe
                    ▼
data/features/_features_{id}_rock_english_segmin25.csv   (features por segmento)
data/features/_summary_{id}_rock_english_segmin25.csv    (agregado por canción)
data/features/variables/_config_{id}_....csv             (snapshot de configuración de esa corrida)
                    │
                    ▼
        shared/utils.py: prepare_summary()  ── merge summary + etiqueta humana por music_name
                    │
                    ▼
                ml/train.py  (CLI de orquestación)
                    │
        ┌───────────┼────────────────────┐
        ▼           ▼                    ▼
  TreeClassifier  LinearRuleClassifier    │
  (ml/models.py)  calibrado por           │
        │         ModelCalibrator         │
        │         (ml/calibration.py)     │
        └───────────┬────────────────────┘
                    ▼
          ConfidenceRouter (ml/confidence.py)
          confiable / ambigua / revisar via predict_proba;
          "ambigua" se resuelve por desempate contra LinearRuleClassifier
                    │
                    ▼
        ml/artifacts/*.joblib (modelos entrenados)
        ml/runs/confidence_routing_{id}.csv (clasificación final + matriz de confusión)
```

## Módulos

### `shared/` — infraestructura común
- **`settings.py`** — único lugar con rutas del proyecto: `PLAYLIST_BASE`, `TRACKS_XLSX`, `FEATURES_DIR`, `ML_ARTIFACTS_DIR`, `ML_RUNS_DIR`, `TIME_LOG_CSV`. Todo lo demás importa rutas desde acá.
- **`utils.py`** — funciones puras reusadas en todo el proyecto: `filename_key`, `load_tracks` (carga y filtra el xlsx de etiquetas), `find_latest_summary`/`find_current_id`/`next_free_id` (resolución de IDs de corrida), `slope`/`extremes_delta`/`aggregate_mean`/`aggregate_median`/`normalize`/`percentile` (estadística), `prepare_summary` (merge summary↔etiquetas por `music_name`), `derive_threshold_bounds` (bounds de calibración derivados de percentiles reales — sigue sin haber ningún número hardcodeado acá).

Sin dependencias dentro del proyecto (solo `pandas`/`numpy`). Todo `features/`, `ml/` y `research/` dependen de `shared/`.

### `features/` — extracción de features (ex `dsp/`)
- **`extract.py`** — extrae features por segmento desde los mp3 reales (onset density/strength, tempo con plegado de octava, centroide y planitud espectral, ZCR, RMS, LRA EBU R128) y las agrega en `score_energia`/`score_ritmo`/`score_noise` por canción. Corre en paralelo (`multiprocessing.Pool`). Escribe `_features_{id}`/`_summary_{id}`/`_config_{id}` en `data/features/`. Expone `FEATURES`, la lista de columnas que consume `ml/`.
- **`config.py`** — todas las constantes de extracción (`SR`, `TRIM_DB`, `ONSET_GATE_PCT`, los 8 pares `*_LO`/`*_HI` de normalización, la configuración de LRA). Los bounds de normalización son tuplas versionadas (`ONSET_LO = (0.5, 2.3124)`) — índice `[0]` es el default original, `[-1]` es el valor calibrado vigente; ese historial reemplaza la necesidad de un comentario de "por qué cambió este número".

### `ml/` — todo el machine learning, consolidado
- **`models.py`** — `BaseClassifier` (interfaz común: `fit`/`predict`/`save`/`load`), `TreeClassifier` (envuelve `DecisionTreeClassifier`/`RandomForestClassifier` de sklearn) y `LinearRuleClassifier` (la regla de 2 ejes energía/ritmo sobre las features normalizadas — benchmark documentado, no el enfoque principal). `CATEGORIES` vive acá, es la única lista de categorías del proyecto.
- **`calibration.py`** — `ModelCalibrator`: calibra un `LinearRuleClassifier` con `differential_evolution` maximizando balanced accuracy. **Los 12 parámetros (7 pesos + 5 umbrales) son completamente libres** — no hay ningún peso fijado a mano. Los bounds de los pesos son `(0.01, 1.0)`; los de los umbrales se derivan de percentiles reales del dataset (`derive_threshold_bounds`).
- **`confidence.py`** — `ConfidenceRouter`: clasifica cada canción como `confiable`/`ambigua`/`revisar` según el margen `predict_proba` top1-top2 de un `TreeClassifier` ya entrenado. Los umbrales (`confident_pct`, `ambiguous_gap`) se derivan de percentiles de la distribución real de probabilidades del propio dataset (`fit_thresholds()`), no son constantes fijas. Para las filas `ambigua`, si se le pasa un `LinearRuleClassifier`, resuelve por desempate: si concuerda con el top1 del árbol lo confirma, si concuerda con el top2 vota por él, si no concuerda con ninguno queda sin resolver.
- **`evaluation.py`** — `Evaluator`: `cross_validate` (K-fold adaptativo según el mínimo de ejemplos por clase), `learning_curve`, `accuracy_by_category`, `confusion_report` (accuracy + matriz de confusión sugerida-vs-humano).
- **`train.py`** — CLI único de orquestación:
  ```
  python -m ml.train --model random_forest --summary-id 12
  python -m ml.train --model linear_rule
  python -m ml.train --model all --learning-curve
  ```
  Entrena el/los modelo(s) pedidos, los guarda en `ml/artifacts/*.joblib`, corre `ConfidenceRouter` sobre el Random Forest (con desempate del modelo lineal si también se entrenó) y escribe `ml/runs/confidence_routing_{id}.csv` con la matriz de confusión final.
- **`ml/artifacts/`** — modelos entrenados (`.joblib`, gitignored, regenerables con `ml/train.py`).
- **`ml/runs/`** — salidas de clasificación por corrida (gitignored, regenerables).

Verificado de punta a punta contra datos reales (`_summary_12`, 81 canciones etiquetadas tras el merge): Random Forest 42.4% ± 16.5 balanced accuracy (cross-validada), regla lineal calibrada 47.0% (train), 72.8% de acuerdo final con desempate (in-sample, mismo patrón que usaba el proyecto históricamente para la capa de confianza).

### `research/` — todo lo que no es el pipeline activo, agrupado con identidad (ex `dsp/obsolet/` + `dsp/test/` + `dsp/files/`)
Reemplaza la carpeta "obsolet" (nombre que no distinguía qué seguía vivo). Nada de lo que había se eliminó — ver `research/README.md` para el detalle de qué es cada script y su estado (activo / histórico / roto a propósito):
- `range_calibration.py` — activo, audita `features/config.py` contra datos reales.
- `hpss_cost_analysis.py`, `execution_time_analysis.py`, `zcr_diagnostics.py` — scripts de investigación puntual (el primero documenta por qué se eliminó HPSS del pipeline).
- `legacy_cross_validate.py` — histórico, roto a propósito (reemplazado por `ml.evaluation.Evaluator`, se conserva como referencia de metodología).
- `legacy_labels/` — el CSV de 119 canciones curado a mano que originó el proyecto, ya no usado por ningún script activo.
- `legacy_runs/`, `legacy_models/` — salidas y modelos de corridas anteriores a este refactor.
- `graphics/` — visualización exploratoria (`matplotlib`).

### Otras carpetas de nivel raíz
- **`data/audio/`** — `_canciones_rock_english.xlsx`, única fuente de verdad de la etiqueta humana.
- **`data/features/`** — salida versionada de `features/extract.py`.
- **`benchmarks/`** — resultado parcial de un intento de etiquetado de género vía Gemini (histórico, sin código asociado en el repo).
- **`fewshot/`** — vacía, dirección futura declarada en `README.md` (embeddings MERT).
- **`definitions/CONCEPTOS-SONIDOS.md`** — glosario DSP del proyecto.

## Qué cambió respecto a la estructura anterior

- **Route B (CLAP) eliminada por completo** — `dsp/llm-clap/` y la sección correspondiente de `requirements.txt` ya no existen. `ConfidenceRouter` resuelve casos ambiguos con el propio `LinearRuleClassifier`, no con un modelo externo.
- **Todo el ML consolidado en `ml/`** — antes repartido entre `classify.py`, `train_tree.py`, `classify_arbol.py`, `clasificador_escenarios.py`, `curva_aprendizaje.py` y `obsolet/calibrate_weights.py`. Ahora son 5 archivos con una interfaz común (`BaseClassifier`) y un solo punto de entrada (`ml/train.py`).
- **Calibración 100% automática** — los pesos que antes estaban fijados a mano (`W_E_CENTROID_FIXED`, `W_E_DYN_FIXED`, `W_R_TEMPO_FIXED`) ahora son parámetros libres más de `differential_evolution`.
- **Dos bugs reales corregidos como efecto del refactor**: el `TypeError` de `train_tree.py`/`curva_aprendizaje.py` (llamaban `prepare_summary()` con un argumento de menos) y el `KeyError: 'filename'` de `classify.py`/`classify_arbol.py` (el summary real usa `music_name`, no `filename`) — verificado corriendo `ml/train.py` contra `_summary_12` sin errores.
- **`dsp/` → `features/` + `ml/` + `research/`** — separa extracción de features, machine learning, e investigación histórica en tres carpetas con una responsabilidad cada una.

## Pendientes

- Reproducir el 73.9% histórico de Random Forest (2026-07-17, dataset/esquema viejo) contra el dataset actual — el número verificado ahora (42.4% cross-validado sobre 81 canciones) no es directamente comparable.
- Confirmar si `SR=44100` y `ONSET_GATE_PCT=60` (`features/config.py`) siguen siendo los valores queridos — el cambio original no quedó documentado en ningún commit.
- `research/hpss_cost_analysis.py` no corre contra datos nuevos (compara dos CSVs de esquema distinto, `filename` vs `music_name`) — no se corrigió porque investiga una feature (`perc_ratio`) que ya no se calcula.
- Decidir el futuro de `main` vs `devalayo` en git (merge pendiente).

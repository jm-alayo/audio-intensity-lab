# Guía de uso — analyze-mp3

Pipeline de clasificación de intensidad para MP3s de Spotify. Toma una carpeta de canciones descargadas con spotdl y las clasifica en tres categorías según su energía y comportamiento a lo largo del tiempo.

---

## Estructura del proyecto

```
analyze-mp3/
├── extract_features.py   ← DSP pesado: genera raw_features.csv
├── calibrate.py          ← Muestra distribución de scores (para elegir umbrales)
├── classify.py           ← Clasifica usando umbrales: genera classified.csv
├── requirements.txt
└── out/
    ├── rock_english_raw_features.csv    ← generado por extract_features.py
    └── rock_english_classified.csv     ← generado por classify.py
```

Los CSV de salida se crean automáticamente en `out/` al ejecutar los scripts.

---

## Requisitos previos

### 1. Dependencias Python

```bash
pip install -r requirements.txt
```

### 2. ffmpeg (necesario para librosa en Windows)

```bash
choco install ffmpeg
```

O descárgalo manualmente desde [ffmpeg.org](https://ffmpeg.org/download.html) y agrégalo al PATH.

---

## Las tres categorías

| Categoría | Criterio | Uso sugerido en Cyberpunk |
|-----------|----------|---------------------------|
| `baja` | Score promedio bajo | Estación ambiental / de fondo |
| `alta` | Score promedio alto, sin crecimiento marcado | Estación de acción |
| `incrementable` | Energía que crece entre el primer y último segmento | Estación de build-up |

Los umbrales exactos se calibran con tus propias canciones usando `calibrate.py`.

---

## Flujo completo

### Paso 1 — Extraer features

```bash
# Procesar toda la playlist (el input es solo el nombre de la carpeta)
python extract_features.py rock_english

# Con límite de canciones (útil para pruebas rápidas)
python extract_features.py rock_english --limit 20

# Con más segmentos por canción (más detalle de la pendiente)
python extract_features.py rock_english --segments 4

# Batch size más pequeño (conexión lenta o máquina limitada)
python extract_features.py rock_english --batch 10
```

**Output:** `out/rock_english_raw_features.csv`

El proceso es **resumible**: si se interrumpe, la siguiente ejecución salta las canciones ya procesadas. Verás al inicio cuántas ya están en el CSV y cuántas quedan.

---

### Paso 2 — Calibrar umbrales

```bash
python calibrate.py rock_english
```

**Output:** Muestra en consola los histogramas y percentiles de `score_promedio` y `pendiente`, y sugiere umbrales basados en P33/P66/P75 de tu distribución real.

Ejemplo de output:

```
SCORE_PROMEDIO — estadísticas (n=423):
  media=0.4821  std=0.0934  min=0.2105  max=0.7438
  Percentiles:
    P10 = 0.3412
    P25 = 0.4102
    P33 = 0.4387   ← score_low sugerido
    P50 = 0.4801
    P66 = 0.5312   ← score_high sugerido
    P75 = 0.5691
    P90 = 0.6244

SUGERENCIA DE UMBRALES:
  score_low   = 0.4387
  score_high  = 0.5312
  slope_thresh= 0.1241

  python classify.py rock_english \
    --score-low 0.4387 \
    --score-high 0.5312 \
    --slope-thresh 0.1241
```

Ajusta los umbrales a tu criterio si la distribución sugerida no refleja tu expectativa musical.

---

### Paso 3 — Clasificar

```bash
# Con umbrales por defecto (0.35 / 0.55 / 0.15)
python classify.py rock_english

# Con umbrales calibrados en el paso anterior
python classify.py rock_english \
  --score-low 0.44 \
  --score-high 0.53 \
  --slope-thresh 0.12
```

**Output:** `out/rock_english_classified.csv` con columnas:

| Columna | Descripción |
|---------|-------------|
| `filename` | Nombre del archivo MP3 |
| `filepath` | Ruta completa |
| `duration_s` | Duración en segundos |
| `score_promedio` | Score de intensidad promedio (0–1) |
| `pendiente` | Diferencia de score entre último y primer segmento |
| `categoria_sugerida` | Categoría asignada por el sistema |
| `categoria_humano` | **Vacía** — para tu validación manual |

---

### Paso 4 — Validación humana (opcional pero recomendada)

1. Abre `out/rock_english_classified.csv` en Excel o cualquier editor
2. Escucha ~40–50 canciones distribuidas entre las 3 categorías
3. Rellena la columna `categoria_humano` con tu juicio (`baja`, `alta`, o `incrementable`)
4. Guarda y ejecuta:

```bash
python classify.py rock_english --validate
```

**Output:**

```
Validación sobre 47 canciones etiquetadas manualmente:
  Acuerdo sistema vs humano: 39/47 = 83.0%

  Matriz de confusión (sistema → humano):
  sugerida/humano   baja             alta             incrementable
  baja              14               2                0
  alta              3                17               1
  incrementable     0                1                9
```

Si el acuerdo es bajo en alguna categoría, ajusta el umbral correspondiente y vuelve a ejecutar `classify.py` (no necesitas re-extraer features).

---

## Referencia de parámetros

### extract_features.py

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `playlist` | — | Nombre de carpeta en `out-playlist/` |
| `--segments` / `-s` | `3` | Segmentos por canción. Con 4–5 la pendiente es más confiable. |
| `--limit` / `-l` | todas | Número máximo de canciones |
| `--batch` / `-b` | `20` | Canciones por batch (para pausar entre bloques) |
| `--output` / `-o` | `out/{playlist}_raw_features.csv` | Ruta del CSV de salida |

### calibrate.py

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `playlist` | — | Nombre de la playlist |
| `--input` / `-i` | `out/{playlist}_raw_features.csv` | CSV de features |
| `--bins` / `-b` | `20` | Bins del histograma ASCII |

### classify.py

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `playlist` | — | Nombre de la playlist |
| `--score-low` | `0.35` | Por debajo → `baja` |
| `--score-high` | `0.55` | Por encima (y no incrementable) → `alta` |
| `--slope-thresh` | `0.15` | Pendiente mínima para `incrementable` |
| `--input` / `-i` | `out/{playlist}_raw_features.csv` | CSV de entrada |
| `--output` / `-o` | `out/{playlist}_classified.csv` | CSV de salida |
| `--validate` | — | Calcula % de acuerdo si `categoria_humano` está rellena |

---

## Cómo funciona el score de intensidad

Cada canción se divide en N segmentos iguales. Por cada segmento se extraen:

| Feature | Rango normalizado | Peso |
|---------|------------------|------|
| RMS energy (amplitud) | 0–0.30 → 0–1 | 40% |
| Onset density (onsets/seg) | 0–8.0 → 0–1 | 35% |
| Tempo (BPM) | 60–200 → 0–1 | 25% |

```
score_segmento = 0.40 * rms_norm + 0.35 * onset_norm + 0.25 * tempo_norm
score_promedio = promedio(score_seg1, score_seg2, ..., score_segN)
pendiente      = score_segN - score_seg1
```

**Preprocesamiento antes de medir:**
1. Recorte de silencio (`librosa.effects.trim`, top_db=30)
2. Normalización de loudness a −23 LUFS (EBU R128, `pyloudnorm`)

Estos pasos son necesarios para que el RMS no quede contaminado por silencio inicial/final ni por diferencias de masterización entre canciones.

---

## Integración con Cyberpunk (Fase 7)

Una vez que `classified.csv` esté validado, puede usarse para distribuir los MP3s a carpetas de estaciones de radio de RadioExt:

```python
import csv, shutil
from pathlib import Path

STATION_MAP = {
    "baja":          Path("C:/...cyberpunk/mods/radioExt/stations/ambient"),
    "alta":          Path("C:/...cyberpunk/mods/radioExt/stations/action"),
    "incrementable": Path("C:/...cyberpunk/mods/radioExt/stations/buildup"),
}

with open("out/rock_english_classified.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cat  = row["categoria_sugerida"]
        dest = STATION_MAP.get(cat)
        if dest:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(row["filepath"], dest / row["filename"])
```

---

## Notas de rendimiento

- `extract_features.py` es el único script lento (DSP sobre audio). Con 400 canciones espera ~15–40 minutos según el hardware.
- `calibrate.py` y `classify.py` leen solo el CSV y son instantáneos.
- Puedes reejecutar `classify.py` con distintos umbrales cuantas veces quieras sin tocar el audio.

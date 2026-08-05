import time
import pyloudnorm as pyln
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1. SEÑAL Y FRECUENCIA (Carga de datos)
PROJECT_ROOT = Path(__file__).parent.parent
PLAYLIST_BASE = PROJECT_ROOT / "spotdl-poc" / "out-playlist" / "rock_english"
SR = 22050
TARGET_LUFS = -23.0    # normalización EBU R128
TRIM_DB=45

#path_audio = PLAYLIST_BASE / "Radiohead - Exit Music (For A Film).mp3"
path_audio = PLAYLIST_BASE / "Milky Chance - Stolen Dance.mp3"
#path_audio = PLAYLIST_BASE / "Vundabar - Alien Blues.mp3"

y, sr = librosa.load(path_audio, duration=30, sr=SR)  # Analizamos los primeros 30s para el ejemplo 

# --- PROCESAMIENTO DE LA SUBFASE 1 ---
start_time = time.time()

# --- NORMALIZACIÓN LUFS ---
meter = pyln.Meter(sr)
lufs  = meter.integrated_loudness(y)

# validar trim
y, _ = librosa.effects.trim(y, top_db=TRIM_DB)

y = pyln.normalize.loudness(y, lufs, TARGET_LUFS)

y = np.clip(y, -1.0, 1.0)

# ENVOLVENTE DE FUERZA ONSET
oenv = librosa.onset.onset_strength(y=y, sr=sr)

# CURVAS DE IMPACTO Y PICOS (Heurística adaptativa por defecto de Librosa)
onset_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr)

# FUERZA MEDIA (El reemplazo del RMS)
onset_strength_mean = float(oenv.mean())

# DENSIDAD ONSET (Eventos por segundo en esta ventana de 30s)
duracion_s = len(y) / sr
onset_density = len(onset_frames) / duracion_s

print(f"Métricas calculadas en {time.time() - start_time:.4f} segundos:")
print(f" - Fuerza Media (Onset Strength Mean): {onset_strength_mean:.4f}")
print(f" - Densidad de Onsets Filtrada: {onset_density:.2f} onsets/segundo")


## ENTENDIENDO VARIABLES HASTA ESTE PUNTO 
## y representa la señal del dominio del tiempo, en otras palabras, es la onda de sonido física capturada digitalmente
## oenv representa la fuerza del impacto o el envolvente de fuerza en cada instante de tiempo, es decir, olas de impactos que miden la tasa de cambio de la energía
## onset_frames representa los instantes (cuadros) de tiempo en los que se recoge la ubicación de los picos de la señal mediante la heurística de la mediana de librosa
## TODAS ESTAS VARIABLES RETORNAN UN ARRAY DE NUMEROS!

# --- GENERACIÓN DE LOS GRÁFICOS VISUALES ---
plt.figure(figsize=(14, 10))

# Gráfico 1: SEÑAL (Forma de onda en el tiempo)
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.6, color="blue")
plt.title("1. Señal de Audio (Dominio del Tiempo)", fontsize=12, fontweight="bold")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo (Segundos)")

# Gráfico 2: ENVOLVENTE DE FUERZA, PICOS Y DENSIDAD
# Convertimos los frames de los onsets a tiempo (segundos) para el gráfico
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
times_oenv = librosa.times_like(oenv, sr=sr)

plt.subplot(3, 1, 2)
plt.plot(
    times_oenv,
    oenv,
    label="Envolvente de Fuerza Onset (oenv)",
    color="purple",
    linewidth=1.5,
)
plt.axhline(
    y=onset_strength_mean,
    color="r",
    linestyle="--",
    label=f"Fuerza Media ({onset_strength_mean:.2f})",
)
plt.vlines(
    onset_times,
    ymin=0,
    ymax=oenv.max(),
    color="orange",
    linestyle=":",
    alpha=0.7,
    label=f"Picos Detectados (Densidad: {onset_density:.2f} n/s)",
)
plt.title(
    "2. Envolvente de Fuerza, Picos Seleccionados y Densidad",
    fontsize=12,
    fontweight="bold",
)
plt.ylabel("Fuerza del Impacto")
plt.xlabel("Tiempo (Segundos)")
plt.legend(loc="upper right")

# Gráfico 3: FRECUENCIA (Espectrograma para entender los LUFS e Impactos)
plt.subplot(3, 1, 3)
stft = librosa.stft(y)
stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
librosa.display.specshow(stft_db, sr=sr, x_axis="time", y_axis="log", cmap="magma")
plt.title(
    "3. Frecuencias en el Tiempo (Espectrograma)", fontsize=12, fontweight="bold"
)
plt.colorbar(format="%+2.0f dB")
plt.xlabel("Tiempo (Segundos)")

plt.tight_layout()
plt.show()
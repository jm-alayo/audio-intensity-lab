import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pyloudnorm as pyln
import time

# 1. SEÑAL Y FRECUENCIA (Carga de datos)
PROJECT_ROOT = Path(__file__).parent.parent
PLAYLIST_BASE = PROJECT_ROOT / "spotdl-poc" / "out-playlist" / "rock_english"
SR = 22050
TARGET_LUFS = -23.0    # normalización EBU R128
TRIM_DB=45

path_audio = PLAYLIST_BASE / "Radiohead - Exit Music (For A Film).mp3"
#path_audio = PLAYLIST_BASE / "Milky Chance - Stolen Dance.mp3"
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

# 2. Calcular las métricas frame por frame (sin el .mean() para poder graficarlas)
frames = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
flatness = librosa.feature.spectral_flatness(y=y)[0]
zcr = librosa.feature.zero_crossing_rate(y=y)[0]

# Crear un vector de tiempo para el eje X que coincida con los frames
times = librosa.times_like(frames, sr=sr)

# 3. Configurar la gráfica con 3 subtramas (Subplots)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# --- Subtrama 1: Centroide Espectral sobre el Espectrograma ---
stft = np.abs(librosa.stft(y))
stft_db = librosa.amplitude_to_db(stft, ref=np.max)
librosa.display.specshow(stft_db, y_axis='linear', x_axis='time', sr=sr, ax=axes[0], cmap='magma')
line_centroid, = axes[0].plot(times, frames, color='cyan', linewidth=2, label='Centroide Espectral')
axes[0].set_title('Centroide Espectral (Brillo/Gravedad de Frecuencia)')
axes[0].legend(loc='upper right')

# --- Subtrama 2: Planitud Espectral ---
line_flatness, = axes[1].plot(times, flatness, color='crimson', linewidth=1.5, label='Planitud (Flatness)')
axes[1].set_title('Planitud Espectral (0.0 = Tonal Puro | 1.0 = Ruido/Distorsión)')
axes[1].set_ylabel('Escala 0 a 1')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='upper right')

# --- Subtrama 3: Tasa de Cruces por Cero (ZCR) ---
line_zcr, = axes[2].plot(times, zcr, color='limegreen', linewidth=1.5, label='ZCR')
axes[2].set_title('Tasa de Cruces por Cero (Frecuencias altas / Rugosidad)')
axes[2].set_xlabel('Tiempo (segundos)')
axes[2].set_ylabel('Tasa de cruce')
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].legend(loc='upper right')

axes[0].tick_params(labelbottom=True)
axes[1].tick_params(labelbottom=True)

plt.tight_layout()
plt.show()
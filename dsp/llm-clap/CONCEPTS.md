# ── Rangos de normalización por feature (clip a [0, 1]) ────────────────────────
# Calibrados con percentiles p5/p95 del dataset real; recalcular en calibrate.py
# cuando se procesen más playlists.
#
# NOTA: el RMS medio deja de usarse como señal de intensidad — tras normalizar
# a -23 LUFS, converge en todas las canciones (86.7% de los segmentos saturaban
# norm_rms=1.0, el "modulador maestro" era una constante). Se reemplaza por
# features que sí sobreviven a la normalización LUFS: fuerza de ataque, brillo
# espectral, ratio percusivo (HPSS) y dinámica intra-segmento.
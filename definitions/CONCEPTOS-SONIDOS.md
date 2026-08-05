**TERMINOLOGÍAS**

* Envolvente de Fuerza Onset
* Densidad Onset
* Señal y Frecuencia
* Curvas de Impacto/Fuerza y Picos (Reemplaza RMS)
* Heurística basada en la mediana de ruido de fondo
* LUFs
* Procesamiento en bits
* Voltaje Digital
* Aceleración  de fuerzas







#### **SEÑAL DE AUDIO (DOMINIO DEL TIEMPO)**

Un **archivo de audio digital** (como un MP3, una vez decodificado) representa la amplitud de cada muestra mediante valores normalizados en un rango de -1.0 a 1.0. Si una señal supera estos límites, se produce ***clipping digital*** o **distorsión por recorte**, ya que el sistema no puede representar amplitudes mayores que el máximo permitido.



Los valores comprendidos dentro de este rango no corresponden a una unidad física específica, sino que son **números de punto flotante normalizados que indican la amplitud relativa de la señal** con respecto al máximo nivel que el formato puede representar. Por ejemplo, un valor de 0.75 significa que la muestra alcanza el 75 % de la amplitud máxima disponible, mientras que un valor de -0.75 representa la misma amplitud, pero con polaridad opuesta.



#### **ENVOLVENTE DE FUERZA (PICOS Y DENSIDAD)**



* **Los Pisos Espectrales** representan el nivel de ruido constante o residual de la canción. Matemáticamente, es el valor mínimo al que regresa la línea morada entre golpe y golpe. En pasajes acústicos o silencios, el piso está pegado a cero en fuerza de impacto; en bloques pesados de rock con distorsión continua, el piso se eleva drásticamente porque la energía nunca deja de sonar, volviéndose el punto de partida para los nuevos sonidos.



* **La Fuerza Media** *(oenv.mean())* es conceptualmente el promedio matemático simple del segmento evaluado: la suma de la altura de todos los puntos de la envolvente dividida entre el total de muestras en el tiempo. Es una representación gráfica e informativa para nosotros que mide la agresividad global, pero no influye en las decisiones del algoritmo ni se calcula mediante la heurística de la mediana.



* **La Heurística de la Mediana** es el verdadero cerebro adaptativo que toma las decisiones. En lugar de usar la línea roja global, analiza ventanas de milisegundos locales a los lados de cada curva. Si una pequeña "montañita" morada sobresale de su entorno inmediato (su mediana local), el algoritmo la aprueba, permitiendo detectar impactos musicales sutiles incluso si están por debajo de la fuerza media de la canción.



* **La Fuerza de Impacto** mide la tasa de cambio o aceleración espectral entre milisegundos contiguos, no el volumen absoluto. Su escala en el eje Y puede incrementarse a necesidad más allá de 10 si ocurre un estallido violento desde el silencio; sin embargo, en bloques pesados, los picos se contienen (no suben tanto) porque la resta matemática ocurre sobre un piso que ya estaba lleno de ruido constante.



* **Los Picos Detectados** son los máximos locales aprobados por la heurística adaptativa (las líneas naranjas). Al calcular la tasa de cuántos de estos picos ocurren por cada segundo evaluado, el sistema genera matemáticamente la Densidad Onset Filtrada (que es, literalmente, tu tasa de densidad de golpes), la métrica final para medir el ritmo de la canción.



* **La línea morada (Envolvente de Fuerza)** es la materia prima cruda que fabrica Librosa transformando el audio original. No tiene nada que ver con el RMS ni con tus scores anteriores. Cada punto en esta línea representa una tasa de cambio espectral: mide milisegundo a milisegundo cuánta energía musical nueva apareció en comparación con el instante anterior.



* **Los scores informativos (Pisos y Fuerza Media)** se extraen analizando la altura de esa línea morada. El "Piso" es el nivel más bajo al que cae la línea entre golpes (elevado en canciones ruidosas y en cero en silencios). La "Fuerza Media" es simplemente el promedio matemático global de toda la curva morada, sirviendo como un indicador visual de agresividad sin interferir en el algoritmo.



#### **ANALISIS DE GRAFICOS**

Para Librosa, la variable *stft\_db* contiene la misma matriz exacta de números con los mismos decibelios en ambos casos. Las funciones matemáticas de análisis timbrico (*spectral\_centroid*, *flatness, zcr*) ni siquiera miran cómo vas a pintar el gráfico; ellas procesan los datos crudos en el backend de forma idéntica. La diferencia es 100% de la capa de presentación (***librosa.display.specshow*** y Matplotlib):

* ***y\_axis='linear':*** Es la mejor para humanos inspeccionando el Centroide Espectral, porque tu ojo puede ver de forma natural y proporcional por qué la línea sube hasta los 6000Hz o se queda en los 2000Hz.
* ***y\_axis='log':*** Es la mejor para humanos analizando armonías o notas musicales, porque el oído humano percibe el tono de forma logarítmica (la diferencia entre un bajo de 60Hz y uno de 120Hz se siente gigante al oído, pero en la escala lineal son apenas unos milímetros de espacio).



#### **CONCEPTO DEL PERCENTIL**

El Percentil 60 *(np.percentile(onset\_strength, 60))* actúa como un termómetro estadístico que analiza el 100% de los datos de la línea morada (ordenados ascendentemente) y devuelve un único número flotante (**el gate**) ubicado en el punto exacto del eje Y donde el 60% de los momentos de la canción tienen una fuerza menor y el 40% restante una fuerza mayor; de por sí esta función no modifica nada, sino que eres tú en tu propio código quien usa ese número devuelto como una aduana matemática para descartar los picos inferiores (ruido o detalles suaves) y quedarse exclusivamente con el 40% de los impactos superiores más fuertes para calcular una tasa de densidad de golpes limpia y real.



#### **CONCEPTO DEL PLEGADO DE OCTAVA**

El Plegado de Octava (fold\_tempo) es una función estabilizadora que fuerza cualquier BPM detectado a entrar en un rango seguro (como 70 a 140 BPM) mediante multiplicaciones o divisiones por 2; esto sirve porque el BPM no mide solo tambores, sino el pulso coordinado de toda la banda (la cuadrícula invisible donde guitarras o sintetizadores generan micro-picos de energía), provocando que Librosa sufra de "miopía matemática" y confunda una balada lenta de 60 BPM con una de 120 BPM al procesar los rasgueos intermedios como si fueran el pulso principal. Al normalizar este dato en el backend, evitas anomalías extremas y permites que tu clasificador cruce de forma justa un BPM coherente con tu Densidad de Golpes para diferenciar con precisión canciones rápidas de baladas lentas en tu CSV.



#### **CENTROIDES ESPECTRALES**

El Centroide Espectral (la línea blanca) es el punto de equilibrio del brillo de la música; funciona como un resumen ejecutivo que comprime el desmadre de frecuencias y energía del espectrograma en un solo punto numérico por instante, permitiendo que tu sistema —al aplicar el .mean()— identifique al grano si un segmento es oscuro y pesado (graves) o chillón y sibilante (agudos) sin procesar miles de píxeles.



El valor promedio del centroide le dice al modelo en qué parte del espectro se concentra la fuerza de la música. El sistema buscará estos tres perfiles automáticos:



* **Centroide Muy Bajo (Ej: < 1500 Hz)**: Le indica al sistema que la sección está dominada por frecuencias graves (bajos, bombos pesados). Al sistema le interesa esto para identificar géneros oscuros, pesados, o secciones tranquilas/introvertidas (como la intro de Exit Music, Jazz, Hip-Hop o Reggae).
* **Centroide Medio (Ej: 1500 a 3000 Hz)**: Es el rango estándar de la música con guitarras eléctricas, voces claras y cajas de batería bien balanceadas (como el cuerpo principal de Alien Blues o canciones de Pop/Rock tradicionales).
* **Centroide Muy Alto (Ej: > 3500 Hz)**: Indica que la energía se volcó por completo hacia los agudos (platillos incesantes, chillidos, siseos, sintetizadores estridentes). Al sistema le interesa para detectar subidones de música electrónica (EDM), Metal extremo (Black/Thrash) o secciones de alta tensión.



#### **TASA DE CRUCES POR CERO** *zero\_crossing\_rate*

Mide la rugosidad física de la onda de audio contando cuántas veces la señal eléctrica pasa de positivo a negativo (cruza la línea del cero) en un segundo. Funciona detectando la velocidad del sonido: las ondas suaves y lentas de un bajo o una guitarra acústica limpia casi no cruzan el eje (valores mínimos como en Exit Music), mientras que las texturas ásperas de la distorsión High-Gain, los golpes metálicos de los platillos o el siseo del aire al susurrar hacen que la onda oscile como loca de arriba a abajo (valores altos como la sierra eléctrica verde de Alien Blues), dándole a tu clasificador el dato clave para identificar la agresividad y el ritmo de los impactos percusivos en cada segmento.



Señal de Audio (Dominio del Tiempo): ¡Exacto! No procesa nada aún; es el lienzo en bruto. Captura los cambios de presión del aire atrapados digitalmente en una onda que sube y baja.



Envolvente de Fuerza, Picos y Densidad: ¡Casi! No mide qué tan fuerte es el sonido en general, sino el impacto del golpe inicial (el onset). Mide la tasa de cambio para detectar cuándo y con qué energía arranca una nota o un tamborazo, permitiendo calcular la densidad de golpes por segundo.



Frecuencias en el Tiempo (Espectrograma): ¡Exacto! Representa la energía (color/dB) distribuida en la agudez (Hz) a lo largo del tiempo. Al ponerlo en log, estiras visualmente los graves para ver mejor las notas musicales.



Centroide Espectral: ¡Ajuste clave! No mide el promedio de la energía total; mide el punto de equilibrio del brillo (el centro de gravedad). Te dice si la energía que hay se inclina hacia el lado de los graves (oscuro) o de los agudos (brillante).



Planitud Espectral: ¡Exacto! Mide qué tanta bulla o ruido caótico hay en el sonido. 0.0 es una nota musical pura y limpia, y los valores que rozan el 0.2 ya indican distorsión pesada o platillos constantes.



Tasa de Cruces por Cero (ZCR): ¡Corrección importante! No mide intensidad (volumen). Mide la rugosidad física o velocidad de la onda. Cuenta cuántas veces oscila la señal cruzando el cero; un valor alto significa texturas ásperas como distorsión eléctrica, siseos de voz o platillos metálicos, sin importar si suenan fuertes o despacio.


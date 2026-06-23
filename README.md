# Proyecto Audio Fourier

Procesamiento de audio en Python aplicando un filtro pasa bajas a archivos WAV usando la transformada de Fourier (DFT/FFT).

🌐 **App en línea:** [Filtro de Fourier - Procesamiento de Audio](https://python-numerico-proyect.fly.dev/)

## Descripción del proyecto

El proceso de filtrado realiza lo siguiente:

1. Carga un archivo WAV.
2. Convierte el audio a una señal de punto flotante.
3. Calcula la transformada de Fourier (DFT/FFT) de la señal.
4. Aplica un filtro pasa bajas conservando solo las componentes de frecuencia por debajo de un valor de corte.
5. Calcula la transformada inversa (IDFT/IFFT) para reconstruir el audio filtrado.
6. Normaliza y guarda el audio resultante en formato WAV.

## 🎵 `yupiter/` — Notebook de procesamiento

Esta es la pieza central del proyecto: `yupiter/audio_processing_notebook.ipynb`, un notebook de Jupyter que implementa y explica paso a paso el filtro pasa bajas por Transformada Discreta de Fourier (DFT), incluyendo gráficas del espectro antes/después y reproducción del audio resultante.

### Contenido de la carpeta

- `audio_processing_notebook.ipynb` — Notebook principal con todo el desarrollo.
- `processed/` — Carpeta de entrada: aquí se colocan los WAV que se quieren procesar. Incluye un audio de ejemplo (`648112__deleted_user_12449013__vocals_animal1_key-fmin_bpm124_jennaevans.wav`).
- `resultado_template.txt` — Plantilla HTML usada dentro del notebook para mostrar visualmente las métricas del resultado (archivo de entrada/salida, frecuencia de muestreo, duración, energía original y filtrada).

### Estructura del notebook

- **Celda 1 — Definición de funciones:**
  - `lowpass_filter(data, fs, cutoff_freq=4000)`: recibe la señal cruda y devuelve la señal filtrada junto con los datos intermedios del espectro. Internamente: normaliza → `fft()` → aplica máscara con `fftfreq` → `ifft()` → desnormaliza.
  - Función de graficado para visualizar el espectro y la señal original vs. filtrada.
  - Función de procesamiento completo del archivo WAV.
- **Celda 2 — Ejecución y resultados:**
  1. Detecta automáticamente el WAV más reciente en `processed/`.
  2. Aplica el filtro con la frecuencia de corte elegida.
  3. Calcula la reducción de energía como porcentaje.
  4. Genera las gráficas y muestra el panel de métricas (usando `resultado_template.txt`).
  5. Reproduce el audio filtrado con el widget `IPython.display.Audio`.

### Cómo usarlo

1. Colócate en la carpeta del notebook:

   ```powershell
   cd yupiter
   ```

2. Asegúrate de tener instaladas las dependencias necesarias:

   ```powershell
   pip install numpy scipy matplotlib jupyter
   ```

3. Coloca el archivo `.wav` que quieras procesar dentro de `processed/` (ya viene uno de ejemplo).
4. Abre el notebook:

   ```powershell
   jupyter notebook audio_processing_notebook.ipynb
   ```

5. Ejecuta las celdas en orden. El resultado filtrado se guarda automáticamente como `uploads/filtered_<nombre>.wav` dentro de `yupiter/`.

### Notas técnicas

- Si el audio es estéreo, el notebook toma solo el **canal izquierdo** para simplificar el procesamiento. Para procesar ambos canales sería necesario llamar a `lowpass_filter` dos veces.
- El archivo de entrada debe ser un WAV legible por `scipy.io.wavfile`.

## Recomendaciones generales

- Usa archivos WAV mono o estéreo; si son estéreo, solo se procesa el primer canal.
- El filtro pasa bajas elimina las frecuencias por encima del valor de corte seleccionado.
- Para audio de gran duración, el proceso puede tardar más.

## Ejemplo de prueba

1. Colocar un archivo `ejemplo.wav` en `yupiter/processed/`.
2. Ajustar `cutoff_freq` a `4000` Hz.
3. Generar y obtener `audio_filtrado.wav`.

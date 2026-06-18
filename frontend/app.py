import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fft import fft, ifft, fftfreq
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import uuid
import io
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB límite

# Configuración de Cloudinary (usa variables de entorno)
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
)

@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/procesar', methods=['POST'])
def procesar_audio():
    try:
        # 1. Recibir el archivo
        if 'audio' not in request.files:
            return jsonify({'error': 'No se envió ningún archivo'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'Archivo vacío'}), 400
        
        # 2. Parámetros del filtro
        cutoff_freq = float(request.form.get('cutoff_freq', 4000))
        
        # 3. Leer el archivo directo en memoria (sin disco)
        input_buffer = io.BytesIO(file.read())

        # 4. Cargar y procesar audio
        fs, data = wavfile.read(input_buffer)
        
        # Si es estéreo, tomar un canal
        if len(data.shape) > 1:
            data = data[:, 0]
        
        # Normalizar si es int16
        is_int16 = (data.dtype == np.int16)
        data = data.astype(np.float32)
        
        if is_int16:
            data = data / 32768.0
        
        N = len(data)
        
        # 5. Transformada de Fourier
        X = fft(data)
        freqs = fftfreq(N, 1/fs)
        
        # 6. Filtro pasa bajas
        filter_mask = np.abs(freqs) <= cutoff_freq
        X_filtered = X * filter_mask
        
        # 7. Transformada inversa
        data_filtered = np.real(ifft(X_filtered))
        
        # 8. Calcular métricas antes de normalizar el volumen del archivo de salida
        original_energy = np.sum(data**2)
        filtered_energy = np.sum(data_filtered**2)
        
        # 9. Normalizar para el archivo de salida (ajuste de volumen)
        max_val = np.max(np.abs(data_filtered))
        if max_val > 0:
            data_filtered_out = data_filtered / max_val * 0.99
        else:
            data_filtered_out = data_filtered
        
        # 10. Convertir a int16
        data_output = (data_filtered_out * 32767).astype(np.int16)
        
        # 11. WAV filtrado → buffer → Cloudinary (sin disco)
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, fs, data_output)
        wav_buffer.seek(0)

        wav_upload = cloudinary.uploader.upload(
            wav_buffer,
            resource_type='raw',
            folder='NUMERICO/audio_filtrado',
            public_id=str(uuid.uuid4()),
            format='wav'
        )
        wav_url = wav_upload['secure_url']
        wav_public_id = wav_upload['public_id']

        # 12. Generar las gráficas (Audio original, Espectro original, Audio filtrado, Espectro filtrado)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Visualizacion del Procesamiento de Audio con Filtro Pasa Bajas (FFT)', fontsize=14, fontweight='bold')
        
        # Eje de tiempo
        t = np.arange(N) / fs
        
        # Solo mostrar la mitad positiva de las frecuencias del espectro FFT
        half_N = N // 2
        positive_freqs = freqs[:half_N]
        positive_X = np.abs(X[:half_N])
        positive_X_filtered = np.abs(X_filtered[:half_N])
        
        # 12.1. Audio Original (Tiempo)
        axs[0, 0].plot(t, data, color='#1f77b4', alpha=0.7)
        axs[0, 0].set_title('Audio Original (Tiempo)', fontsize=10, fontweight='bold')
        axs[0, 0].set_xlabel('Tiempo (s)', fontsize=8)
        axs[0, 0].set_ylabel('Amplitud', fontsize=8)
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)
        
        # 12.2. Espectro FFT Original (Frecuencia)
        axs[0, 1].plot(positive_freqs, positive_X, color='#ff7f0e', alpha=0.7)
        axs[0, 1].set_title('Espectro FFT Original (Frecuencia)', fontsize=10, fontweight='bold')
        axs[0, 1].set_xlabel('Frecuencia (Hz)', fontsize=8)
        axs[0, 1].set_ylabel('Magnitud (Escala Log)', fontsize=8)
        axs[0, 1].set_yscale('log')
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)
        
        # 12.3. Audio Filtrado (Tiempo)
        axs[1, 0].plot(t, data_filtered, color='#2ca02c', alpha=0.7)
        axs[1, 0].set_title('Audio Filtrado (Tiempo)', fontsize=10, fontweight='bold')
        axs[1, 0].set_xlabel('Tiempo (s)', fontsize=8)
        axs[1, 0].set_ylabel('Amplitud', fontsize=8)
        axs[1, 0].grid(True, linestyle='--', alpha=0.6)
        
        # 12.4. Espectro FFT Filtrado (Frecuencia)
        axs[1, 1].plot(positive_freqs, positive_X_filtered, color='#d62728', alpha=0.7)
        axs[1, 1].set_title('Espectro FFT Filtrado (Frecuencia)', fontsize=10, fontweight='bold')
        axs[1, 1].set_xlabel('Frecuencia (Hz)', fontsize=8)
        axs[1, 1].set_ylabel('Magnitud (Escala Log)', fontsize=8)
        axs[1, 1].set_yscale('log')
        axs[1, 1].axvline(x=cutoff_freq, color='black', linestyle='--', alpha=0.8, label=f'Corte: {int(cutoff_freq)} Hz')
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # 12.5. Gráfico → buffer → Cloudinary (sin disco)
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png', dpi=150)
        plt.close()
        plot_buffer.seek(0)

        plot_upload = cloudinary.uploader.upload(
            plot_buffer,
            resource_type='image',
            folder='NUMERICO/audio_plots',
            public_id=str(uuid.uuid4())
        )
        plot_url = plot_upload['secure_url']

        return jsonify({
            'success': True,
            'wav_url': wav_url,
            'wav_public_id': wav_public_id,
            'plot_url': plot_url,
            'fs': int(fs),
            'duration': round(N/fs, 2),
            'original_energy': float(original_energy),
            'filtered_energy': float(filtered_energy),
            'cutoff_freq': cutoff_freq
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Servidor iniciado en http://localhost:5000")
    print("Subi un audio WAV y se aplicara filtro pasa bajas")
    app.run(debug=True, port=5000)

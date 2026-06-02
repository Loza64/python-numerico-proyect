import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fft import fft, ifft, fftfreq
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid

app = Flask(__name__)
CORS(app)

# Crear carpetas necesarias
os.makedirs('uploads', exist_ok=True)
os.makedirs('processed', exist_ok=True)

# Configuración
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB límite

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
        
        # 3. Guardar archivo temporal
        input_filename = f"uploads/{uuid.uuid4()}.wav"
        file.save(input_filename)
        
        # 4. Cargar y procesar audio
        fs, data = wavfile.read(input_filename)
        
        # Si es estéreo, tomar un canal
        if len(data.shape) > 1:
            data = data[:, 0]
        
        data = data.astype(np.float32)
        
        # Normalizar si es int16
        if data.dtype == np.int16:
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
        
        # 8. Normalizar
        max_val = np.max(np.abs(data_filtered))
        if max_val > 0:
            data_filtered = data_filtered / max_val * 0.99
        
        # 9. Convertir a int16
        data_output = (data_filtered * 32767).astype(np.int16)
        
        # 10. Guardar resultado
        output_filename = f"processed/{uuid.uuid4()}.wav"
        wavfile.write(output_filename, fs, data_output)
        
        # 11. Calcular métricas
        original_energy = np.sum(data**2)
        filtered_energy = np.sum(data_filtered**2)
        
        # 12. Limpiar archivo original
        os.remove(input_filename)
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'fs': int(fs),
            'duration': round(N/fs, 2),
            'original_energy': float(original_energy),
            'filtered_energy': float(filtered_energy),
            'cutoff_freq': cutoff_freq
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/descargar/<filename>')
def descargar(filename):
    return send_file(f"processed/{filename}", as_attachment=True, download_name="audio_filtrado.wav")

if __name__ == '__main__':
    print("🚀 Servidor iniciado en http://localhost:5000")
    print("📌 Subí un audio WAV y se aplicará filtro pasa bajas")
    app.run(debug=True, port=5000)
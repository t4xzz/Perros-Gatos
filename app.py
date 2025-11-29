from flask import Flask, request, render_template, jsonify
import cv2 as cv
import numpy as np
from tensorflow import keras
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Variable global para el modelo
modelo = None

def cargar_modelo():
    """Carga el modelo CNN una sola vez al iniciar la aplicación"""
    global modelo
    
    # Obtener la ruta absoluta del directorio donde está app.py
    # Esto arregla los errores de "File not found" en la nube
    directorio_base = os.path.dirname(os.path.abspath(__file__))
    
    try:
        def cargar_pesos(ruta_bin):
            # Verificación de depuración para Git LFS
            if os.path.exists(ruta_bin):
                tamano = os.path.getsize(ruta_bin)
                print(f"--> Cargando: {os.path.basename(ruta_bin)} ({tamano} bytes)")
            
            with open(ruta_bin, 'rb') as f:
                return np.frombuffer(f.read(), dtype=np.float32)
        
        # Definir arquitectura del modelo
        modelo = keras.Sequential([
            keras.layers.InputLayer(input_shape=(100, 100, 1)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        print("Iniciando carga de pesos...")
        
        # Construir rutas "blindadas" a los archivos
        ruta_bin_1 = os.path.join(directorio_base, 'group1-shard1of2.bin')
        ruta_bin_2 = os.path.join(directorio_base, 'group1-shard2of2.bin')

        pesos1 = cargar_pesos(ruta_bin_1)
        pesos2 = cargar_pesos(ruta_bin_2)
        todos_pesos = np.concatenate([pesos1, pesos2])
        
        # Asignar pesos a cada capa
        idx = 0
        for capa in modelo.layers:
            pesos_capa = capa.get_weights()
            if len(pesos_capa) > 0:
                nuevos_pesos = []
                for peso in pesos_capa:
                    tamano = np.prod(peso.shape)
                    datos = todos_pesos[idx:idx+tamano].reshape(peso.shape)
                    nuevos_pesos.append(datos)
                    idx += tamano
                capa.set_weights(nuevos_pesos)
        
        print(f"✅ Modelo cargado exitosamente. Parámetros: {modelo.count_params():,}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al cargar el modelo: {repr(e)}")
        # Importante: No detenemos la app, pero dejamos el modelo como None
        return False

# --- IMPORTANTE: CARGAR EL MODELO AL INICIO ---
# Esto asegura que Gunicorn (Render) cargue el modelo antes de recibir visitas
print("Inicializando aplicación...")
cargar_modelo()
# ----------------------------------------------

def procesar_imagen(imagen_bytes):
    """Procesa la imagen y realiza la predicción"""
    # Protección por si el modelo falló al cargar
    if modelo is None:
        return None, "El modelo no está disponible (Error de carga en el servidor)"

    try:
        # Convertir bytes a numpy array
        nparr = np.frombuffer(imagen_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            return None, "No se pudo decodificar la imagen"
        
        # Preprocesar imagen
        imgRedimensionada = cv.resize(frame, (100, 100))
        imagenGrises = cv.cvtColor(imgRedimensionada, cv.COLOR_BGR2GRAY)
        imagenNormalizada = imagenGrises / 255.0
        imagenProcesada = imagenNormalizada.reshape(1, 100, 100, 1)
        
        # Realizar predicción
        prediccion = modelo.predict(imagenProcesada, verbose=0)[0][0]
        
        # Determinar resultado
        if prediccion <= 0.5:
            resultado = "GATO"
            color = (255, 165, 0)  # Naranja en BGR
        else:
            resultado = "PERRO"
            color = (0, 255, 0)  # Verde en BGR
        
        confianza = abs(prediccion - 0.5) * 200
        
        # Dibujar resultado en la imagen
        cv.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv.putText(frame, resultado, (20, 70), cv.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        cv.putText(frame, f"Confianza: {confianza:.1f}%", (20, 110), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convertir imagen procesada a base64 para mostrar en web
        _, buffer = cv.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'resultado': resultado,
            # CORRECCIÓN JSON: Convertir a float nativo de Python
            'confianza': round(float(confianza), 2),
            'score': round(float(prediccion), 4),
            'imagen': img_base64
        }, None
        
    except Exception as e:
        return None, f"Error al procesar: {str(e)}"

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/clasificar', methods=['POST'])
def clasificar():
    """Endpoint para clasificar imágenes"""
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    file = request.files['imagen']
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Formato no válido. Use PNG, JPG o JPEG'}), 400
    
    try:
        imagen_bytes = file.read()
        resultado, error = procesar_imagen(imagen_bytes)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check para Cloud Run / Render"""
    return jsonify({
        'status': 'ok', 
        'modelo_cargado': modelo is not None
    })

if __name__ == '__main__':
    print("=" * 70)
    print("INICIANDO SERVIDOR FLASK (MODO LOCAL)")
    print("=" * 70)
    # Nota: cargar_modelo() ya se ejecutó arriba al importar el script
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
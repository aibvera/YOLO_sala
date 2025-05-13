'''
Buenas.
ABV 2024.

Script para la supervisión.

IMPORTANTE:
- Debe haber una carpeta saved para el guardado del contenido multimedia.
- Debe haber una BD llamada db.db para el almacenado de registros.
- Se debe actualizar la ruta del folder en el archivo run.bat.
'''

import cv2
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import torch
import os
import time

# Prueba de funcionamiento de cuda con pytorch:
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} está disponible.")
else:
    print("No hay una GPU disponible. Se usará la CPU")

# URL RTSP de la cámara (reemplaza con la URL correcta)
rtsp_url = "rtsp://abuenov:ingdesonido@192.168.1.40:554/stream1"
# /stream1 para visualización dentro de la misma red.
# /stream2 para visualización remota.

# Iniciar la captura de video desde la cámara RTSP
cap = cv2.VideoCapture(rtsp_url)

# Verifica si la conexión fue exitosa
if not cap.isOpened():
    print("No se puede conectar a la cámara")
    exit()

# Importar modelo YOLO:
model = YOLO('yolov8n.pt')

# Análisis de cuadros del video:
c = 0
print('Iniciando bucle de detección')
try:
    while True:

        # Lectura del frame:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error al recibir el frame de la cámara")
            break
        t = time.time() - start
        if t > 1_000:
            dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('Lectura de cuadros muy lenta /', dt_string)

        # Detección de bertas:
        c += 1
        if c % 30 != 0:
            continue
        c = 0
        results = model.predict(frame, classes=[0,16], show=False, verbose=False)
        frame = results[0].plot()

        # Procesar cajas si se han detectado:
        if results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu()

            # Loop entre cajas detectadas:
            for box in boxes:

                # Selección de caja desde imagen:
                x, y, w, h = box
                x = int(x)
                y = int(y)
                w = int(w/2)
                h = int(h/2)
                detected = frame[y-h:y+h,x-w:x+w,:]

                # Guardar imagen de la caja:
                date_today = datetime.now().strftime('%Y-%m-%d')
                time_now = datetime.now().strftime('%H-%M-%S')
                save_dir = os.path.join('saved', date_today)
                os.makedirs(save_dir, exist_ok=True)  # Crea todos los directorios necesarios
                file_path = os.path.join(save_dir, f'{time_now}.png')
                cv2.imwrite(file_path, detected)
                print('Se detecto y guardó correctamente una imagen a las', time_now)

except KeyboardInterrupt:
    print("Interrupción manual recibida. Cerrando...")

except Exception as e:
    # Obtener la fecha y hora actual
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Nombre del archivo de error
    error_filename = f"Error_{current_time}.txt"

    # Guardar el error en un archivo de texto
    with open(error_filename, "w") as f:
        f.write(f"Ocurrió un error inesperado a las {current_time}:\n")
        f.write(str(e))
        f.write('\n')

    print(f"Se ha guardado el error en {error_filename}")

# Liberar la captura y cerrar las ventanas
cap.release()

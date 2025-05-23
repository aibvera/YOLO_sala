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

# Función de escritura en BD:
def write(sql_str: str):
    with sqlite3.connect('db.db') as conn:
        cursor = conn.cursor()
        cursor.execute(sql_str)
        conn.commit()

# Función de lectura de BD:
def query(sql_str: str):
    with sqlite3.connect('db.db') as conn:
        cursor = conn.cursor()
        cursor.execute(sql_str)
        return cursor.fetchall()

# Crear campos en la BD, de ser nueva:
with sqlite3.connect('db.db') as conn:
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Registers (
        DateTime TEXT,
        Track_Id INTEGER,
        Path     TEXT
    );
    ''')
    conn.commit()

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

# Encontrar el último id detectado:
last_id = query('SELECT MAX(Track_Id) FROM Registers;')[0][0]
if last_id is None:
    last_id = 0

# Análisis de cuadros del video:
detected_ids = []
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

        # Tracking persistente (consistente entre cuadros) de las cajas:
        c += 1
        if c % 15 != 0:
            continue
        c = 0
        results = model.track(frame, persist=True, classes=0, show=False, verbose=False)
        frame = results[0].plot()

        # Procesar cajas si se han detectado:
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Loop entre cajas detectadas:
            for box, track_id in zip(boxes, track_ids):
                if track_id not in detected_ids:  # Para no sobreescribir la primera

                    # Selección de caja desde imagen:
                    x, y, w, h = box
                    x = int(x)
                    y = int(y)
                    w = int(w/2)
                    h = int(h/2)
                    detected = frame[y-h:y+h,x-w:x+w,:]

                    # Guardar imagen de la caja:
                    d_string = datetime.now().strftime("%Y-%m-%d")
                    p = f'saved/{d_string}'
                    if not os.path.isdir(p):
                        os.mkdir(p)
                    new_id = last_id + track_id
                    file_path = f'./saved/{d_string}/Id_{new_id}.png'
                    cv2.imwrite(file_path, detected)
                    detected_ids.append(track_id)

                    # Escritura del registro en BD:
                    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    query = f'''INSERT INTO Registers (DateTime, Track_Id, Path) VALUES ('{dt_string}', {str(new_id)}, '{file_path}');'''
                    write(query)
                    print('Se detecto y guardó correctamente al id', new_id)

except KeyboardInterrupt:
    print("Interrupción manual recibida. Cerrando...")

except Exception as e:
    # Obtener la fecha y hora actual
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Nombre del archivo de error
    error_filename = f"Error_{current_time}.txt"

    # Guardar el error en un archivo de texto
    with open(error_filename, "w") as f:
        f.write(f"Ocurrió un error inesperado a las {current_time}:\n")
        f.write(str(e))

    print(f"Se ha guardado el error en {error_filename}")

# Liberar la captura y cerrar las ventanas
cap.release()

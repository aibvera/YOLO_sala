'''
Buenas.
ABV 2023.

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
import multiprocessing
import psutil

# Prueba de funcionamiento de cuda con pytorch:
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} está disponible.")
else:
    print("No hay una GPU disponible. Se usará la CPU")

# Limitar hilos por variables de entorno
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENMP"] = "false"

# Limitar hilos en OpenCV y PyTorch:
cv2.setNumThreads(1)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Obtener el ID del proceso actual
p = psutil.Process(os.getpid())
# Limitar la afinidad del proceso a un solo núcleo (el núcleo 0)
p.cpu_affinity([0])
print(f"Afinidad de CPU establecida para el proceso: {p.cpu_affinity()}")

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

# Crear una ventana nombrada para mostrar el video:
# cv2.namedWindow('Cámara de Seguridad', cv2.WINDOW_NORMAL)

# Encontrar el último id detectado:
last_id = query('SELECT MAX(Track_Id) FROM Registers;')[0][0]
if last_id is None:
    last_id = 0

# Análisis de cuadros del video:
c = 0
detected_ids = []
# Información sobre núcleos disponibles y configurados
print("Núcleos disponibles:", multiprocessing.cpu_count())
print("Núcleos usados por PyTorch:", torch.get_num_threads())
try:
    while True:

        # Lectura del frame:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error al recibir el frame de la cámara")
            break
        t = time.time() - start
        if t > 5_000:
            dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('Lectura de cuadros muy lenta /', dt_string)

        # Contador de N cuadros por segundo:
        if c > 0 and c <= 4:
            c += 1
            continue
        else:
            c = 0

        # Tracking persistente (consistente entre cuadros) de las cajas:
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

        # Mostrar el frame capturado en la ventana estable:
        # cv2.imshow('Camara de Seguridad', frame)

        # Presionar 'q' para salir del bucle:
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Actualizar contador:
        c += 1

except KeyboardInterrupt:
    print("Interrupción manual recibida. Cerrando...")

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

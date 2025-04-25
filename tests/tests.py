import cv2
from ultralytics import YOLO
import psutil
import torch
import time
import os

# URL RTSP de la cámara (reemplaza con la URL correcta)
rtsp_url = "rtsp://abuenov:ingdesonido@192.168.1.40:554/stream1"
# /stream1 para visualización dentro de la misma red.
# /stream2 para visualización remota.


def test_camera_conn():

    # Intenta abrir la conexión
    cap = cv2.VideoCapture(rtsp_url)

    # Evaluar
    if not cap.isOpened():
        print("❌ No se pudo conectar con la cámara.")
    else:
        print("✅ Conexión exitosa con la cámara.")
        ret, frame = cap.read()
        if ret:
            print("✅ Frame recibido correctamente.")
            # Puedes guardar una imagen si quieres verla después
            cv2.imwrite("test_frame.jpg", frame)
            print("🖼️ Imagen guardada como frame.jpg")
        else:
            print("⚠️ No se pudo leer un frame.")
        cap.release()


def test_yolo(threads: int):

    # Obtener el ID del proceso actual y limitar núcleos
    if threads > 8:
        return print('No usar más de 8 nucleos.')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    p = psutil.Process(os.getpid())
    p.cpu_affinity([i for i in range(threads)])

    # Lectura de imagen
    model = YOLO("yolov8n.pt")
    p = os.path.join(os.path.dirname(__file__), 'data/person.jpg')
    frame = cv2.imread(p)

    # Solo detección
    start = time.time()
    results = model(frame, classes=0, verbose=False)
    print("⏱️ Detección:", round(time.time() - start, 2), "s")

    # Detección + tracking
    start = time.time()
    results = model.track(frame, classes=0, persist=True, verbose=False)
    print("⏱️ Tracking:", round(time.time() - start, 2), "s")

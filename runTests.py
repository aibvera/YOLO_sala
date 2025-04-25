from tests import test_camera_conn, test_yolo

def main():

    print('######################## Pruebas ########################')
    print()

    # Prueba de conexión a cámara
    print('Conexión a cámara')
    test_camera_conn()
    print()

    # Prueba de tiempo de yolo
    print('Detección y tracking (YOLO)')
    n = 8
    print('Número de hilos:', n)
    test_yolo(n)
    print()

    print('##################### Fin de pruebas #####################')

if __name__ == '__main__':
    main()

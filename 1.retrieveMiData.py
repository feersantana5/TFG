#!/usr/bin/python3

import argparse
import roypy
import time
import queue
from api.sample_camera_info import print_camera_info
from roypy_sample_utils import CameraOpener, add_camera_opener_options
from roypy_platform_utils import PlatformHelper
import numpy as np
import matplotlib.pyplot as plt
import mpldatacursor


class MyListener(roypy.IDepthDataListener):
    # inicilizo contador de frame
    frame = 0

    # constructor
    def _init_(self, q, directorio, clase_objeto, objeto, n_captura):
        super(MyListener, self).__init__()
        self.queue = q
        self.directorio = directorio
        self.claseObjeto = clase_objeto
        self.objeto = objeto
        self.nCaptura = n_captura

    def onNewData(self, data):
        # lista por cada eje y lista tridimensional
        zvalues = []
        yvalues = []
        xvalues = []
        xyzvalues = []
        # bucle for que recorre todos los puntos
        for i in range(data.getNumPoints()):
            # añade a la lista los puntos del eje correspondiente
            zvalues.append(data.getZ(i))
            yvalues.append(data.getY(i))
            xvalues.append(data.getX(i))
            # lista tridimensional
            xyzvalues.append([data.getX(i), -data.getY(i), -data.getZ(i)])
        # array que convierte la lista de valores en array vectorizado
        zarray = np.asarray(zvalues)
        yarray = np.asarray(yvalues)
        xarray = np.asarray(xvalues)
        xyzarray = np.asarray(xyzvalues)
        # array reformateado sin modificar sus datos
        # el numero de elementos por fila es desconocido y por columna segun el ancho de la profundida de imagen
        p = zarray.reshape(-1, data.width)

        # guarda el array en el directorio indicado en formato .txt
        np.savetxt(f"./{self.directorio}/{self.claseObjeto}/{self.objeto}/captura{self.nCaptura}/MatricesX/{self.objeto}X{self.nCaptura}{self.frame}.txt", xarray)
        np.savetxt(f"./{self.directorio}/{self.claseObjeto}/{self.objeto}/captura{self.nCaptura}/MatricesY/{self.objeto}Y{self.nCaptura}{self.frame}.txt", yarray)
        np.savetxt(f"./{self.directorio}/{self.claseObjeto}/{self.objeto}/captura{self.nCaptura}/MatricesZ/{self.objeto}Z{self.nCaptura}{self.frame}.txt", zarray)
        np.savetxt(f"./{self.directorio}/{self.claseObjeto}/{self.objeto}/captura{self.nCaptura}/MatricesXYZ/{self.objeto}XYZ{self.nCaptura}{self.frame}.txt", xyzarray)

        # añade a la cola el array p
        self.queue.put(p)
        # incrementa la variable frame
        self.frame += 1

    def paint(self, data):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData."""

        # crea la imagen de profundidad con los datos crudos
        fig, ax = plt.subplots()
        ax.imshow(data, interpolation='none')
        mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
        plt.imshow(data)
        # guarda la imagen en el directorio indicado
        plt.savefig(f"./{self.directorio}/{self.claseObjeto}/{self.objeto}/captura{self.nCaptura}/ImagenesProfundidad(2D)/{self.objeto}C{self.nCaptura}F{self.frame-1}.png", bbox_inches='tight')
        # pausa necesaria para asegurar algunas imagenes
        plt.pause(0.05)


def main():
    # Wrapper que inicializa llamadas necesarias por windows para operar camaras UVC (USB video class)
    PlatformHelper()
    # valores por defecto
    seconds = 5
    mode = "MODE_9_5FPS_2000"
    directorio = "MiData"
    clase_objeto = "mesa"
    objeto = "MesaB"
    n_captura = 1
    # añade a la cámara la interfaz para actuar con los args dados por consola y activa y encience la cámara con
    # la configuracion deseada
    parser = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser)
    parser.add_argument("-s", "--seconds", type=int, default=seconds, help="duracion de la captura")
    parser.add_argument("-m", "--mode", type=str, default=mode, help="modo de capturacion: "
                                                                                   "MODE_9_5FPS_2000, "
                                                                                   "MODE_9_10FPS_1000, "
                                                                                   "MODE_9_15FPS_700, "
                                                                                   "MODE_9_25FPS_450, "
                                                                                   "MODE_5_35FPS_600, "
                                                                                   "MODE_5_45FPS_500")
    parser.add_argument("-d", "--directorio", type=str, default=directorio, help="ruta del directorio a guardar los resultados")
    parser.add_argument("-c", "--clase_objeto", type=str, default=clase_objeto, help="clase del objeto a capturar")
    parser.add_argument("-o", "--objeto", type=str, default=objeto, help="nombre del objeto a capturar")
    parser.add_argument("-n", "--n_captura", type=int, default=n_captura, help="numero de captura")
    options = parser.parse_args()
    opener = CameraOpener(options)
    cam = opener.open_camera()
    # Seleccion de modo de captura deseado
    cam.setUseCase(options.mode)

    # presenta por consola la informacion escogida en la camara
    print_camera_info(cam)
    print("isConnected", cam.isConnected())
    print("getFrameRate", cam.getFrameRate())

    # usamos la cola para sincronizar el callback con el hilo principal, el dibujo debe de ir en el hilo principal
    # inicializa la cola, asocia a la camara un listener y comienca a captar datos
    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)
    cam.startCapture()
    # crea un bucle que corra un tiempo (por defecto 5 segundos)
    process_event_queue(q, l, options.seconds)
    cam.stopCapture()

def process_event_queue(q, painter, seconds):
    # crea un bucle que correra la cantiad de tiempo indicada
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            # obtiene item de la cola
            # se bloquea hasta obtener un item o haber pasado 5 segundos
            item = q.get(True, 5)
        except queue.Empty:
            # cuando se pasa el tiempo
            break
        else:
            # crea imagen
            painter.paint(item)


if __name__ == "__main__":
    main()

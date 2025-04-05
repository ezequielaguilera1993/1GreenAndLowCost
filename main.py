from utils import *  # Importás la clase base con utilidades

FAST_RUN = True
FAST_RUN_SPEED = 0.3


class Geometria(CrearFiguras, Tools):  # Heredás de la clase utilitaria
    def play(self, *animations, run_time=1, **kwargs):
        if FAST_RUN:
            run_time = FAST_RUN_SPEED
        return super().play(*animations, run_time=run_time, **kwargs)

    def construct(self):
        self.crear_ejes_con_camara(x_range=[-10, 10, 1], y_range=[-7, 7, 1])

        t1 = self.crear_figura_coloreada(([0, 0,"le"], [1, 0], [ 1, 1,"le"]), etiquetar_vertices=False)

        self.wait(10, frozen_frame=True)



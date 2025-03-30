from utils import *# Importás la clase base con utilidades

FAST_RUN = True

class TrianguloConIncentro(CrearFiguras):  # Heredás de la clase utilitaria
    def play(self, *animations, run_time=1, **kwargs):
        if FAST_RUN:
            run_time = 0.3
        return super().play(*animations, run_time=run_time, **kwargs)

    def construct(self):
        self.crear_ejes_con_camara(x_range=[-10, 10, 1], y_range=[-7, 7, 1])
        puntos = [("A", 0, 0),
                  ("B", 5, 0),]
        p=self.crear_figura(puntos)


        p1=np.array([0,1,0])
        self.crear_figura([("D",0,1)])

        self.dibujar_linea_perpendicular(p,0.3)

        p2 = np.array([1, 0, 0])
        # Dibujar el círculo usando los dos puntos
        self.dibujar_circulo_dos_puntos(p1, p2, escala=1.0, color=BLUE, stroke_width=2)

        self.wait(20)

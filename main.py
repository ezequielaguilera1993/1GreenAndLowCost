from utils import *# Importás la clase base con utilidades

FAST_RUN = True

class TrianguloConIncentro(CrearFiguras, Tools):  # Heredás de la clase utilitaria
    def play(self, *animations, run_time=1, **kwargs):
        if FAST_RUN:
            run_time = 0.3
        return super().play(*animations, run_time=run_time, **kwargs)

    def construct(self):
        self.crear_ejes_con_camara(x_range=[-10, 10, 1], y_range=[-7, 7, 1])

        puntos = [("A", -6, 0),
                  ("B", 5, 0),
                  ]
        self.crear_figura(puntos)


        # Dibujar el círculo usando los dos puntos
        circuloCentral=self.dibujar_circulo_auxiliar([0,2],c=[1,0])
        #
        print("circulo: ",circuloCentral.get_l)

        segmentoIzquierdo=self.crear_figura_auxiliar([circuloCentral.get_l, (0, 2)])


        circuloAuxiliar=self.dibujar_circulo_auxiliar(r=segmentoIzquierdo.get_tamaño_lado("AB"),c=circuloCentral.get_r)

        puntos = self.interseccion_figuras(circuloCentral, circuloAuxiliar)
        for p in puntos:
            self.crear_punto(p, color=GREEN)


        interseccionDeArriba=self.seleccionar_punto(12, puntos)
        self.crear_punto(interseccionDeArriba, color=YELLOW)

        self.crear_figura_auxiliar([interseccionDeArriba,circuloAuxiliar.get_c])

        paralela= self.crear_figura_auxiliar([(-6,2), (6,2)])

        self.wait(10, frozen_frame=True)


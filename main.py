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
        circulo=self.dibujar_circulo([0,2],c=[1,0])
        #
        print("circulo: ",circulo.get_l)

        segmentoIzquierdo=self.crear_figura_auxiliar([(3, 2), (0, 2)])
        #
        # circulo2=self.dibujar_circulo(r=segmentoIzquierdo.get_tamaño_lado("AB"),c=circulo.get_r)
        #
        # puntos = self.interseccion_figuras(circulo, circulo2)
        # for p in puntos:
        #     self.crear_punto(p, color=GREEN)



        # self.crear_punto(self.seleccionar_punto(12, puntos), color=YELLOW)

        # Agregar un texto a la escena
        # texto = Text("Vamos a jugar al balon cesto y me das mi smartphone?")
        # texto.to_edge(DOWN)  # Posiciona el texto en la parte inferior
        # self.play(Write(texto))

        self.wait(10, frozen_frame=True)


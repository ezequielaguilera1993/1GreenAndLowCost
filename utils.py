# utils.py

from manim import *
import math
import numpy as np


class Scene(Scene):
    def crear_ejes_con_camara(self, x_range, y_range):
        ejes = NumberPlane(
            x_range=x_range,
            y_range=y_range,
            background_line_style={
                "stroke_opacity": 0.4,
                "stroke_color": LIGHT_GRAY,
                "stroke_width": 1
            },
            axis_config={
                "stroke_color": LIGHT_GRAY,
                "stroke_width": 1,
            }
        )
        plane_width = ejes.width
        plane_height = ejes.height
        aspect_ratio = config.frame_width / config.frame_height
        if plane_width / plane_height > aspect_ratio:
            self.camera.frame_width = plane_width
            self.camera.frame_height = plane_width / aspect_ratio
        else:
            self.camera.frame_height = plane_height
            self.camera.frame_width = plane_height * aspect_ratio
        self.play(Create(ejes), run_time=0.5)

class CrearFiguras(Scene):

    def dibujar_circulo_dos_puntos(
            self,
            p1: np.ndarray,
            p2: np.ndarray,
            escala: float = 1.0,
            color=BLUE,
            stroke_width: float = 1.0
    ):
        """
        Dibuja un círculo a partir de dos puntos, considerando dichos puntos como extremos del diámetro.

        Parámetros:
        -----------
        p1 : np.ndarray
            Primer punto (coordenadas [x, y, z]) que define el diámetro.
        p2 : np.ndarray
            Segundo punto (coordenadas [x, y, z]) que define el diámetro.
        escala : float, opcional
            Factor para modificar el tamaño del círculo. Por defecto, 1.0, de modo que
            el radio es la mitad de la distancia entre los puntos.
        color : Manim color
            Color del círculo. Por defecto, BLUE.
        stroke_width : float, opcional
            Grosor del contorno del círculo. Por defecto, 1.0.

        Retorna:
        --------
        circulo : Circle
            Objeto Circle dibujado.
        """
        # Calcular el centro (punto medio) del diámetro
        centro = (p1 + p2) / 2
        # Calcular la distancia entre los dos puntos
        distancia = np.linalg.norm(p2 - p1)
        # Calcular el radio (la mitad de la distancia, ajustado por escala)
        radio = (distancia / 2) * escala

        # Crear y posicionar el círculo
        circulo = Circle(radius=radio, color=color, stroke_width=stroke_width).move_to(centro)
        self.play(Create(circulo))
        return circulo

    def dibujar_linea_perpendicular(
            self,
            figura: Polygon,
            t: float,
            escala: float = 2.0,  # Valor predeterminado ajustado para que la línea sea del mismo largo que la figura
            color=YELLOW,
            stroke_width: float = 0.5,
            dash_length: float = 0.1,
            dashed_ratio: float = 0.7
    ):
        """
        Dibuja una línea perpendicular al segmento definido por un objeto Polygon de dos puntos,
        en el punto determinado por el parámetro t.

        Parámetros:
        -----------
        figura : Polygon
            Objeto Polygon con exactamente dos vértices.
        t : float
            Parámetro entre 0 y 1 que indica el punto en el segmento donde se dibuja la perpendicular.
            0 indica el punto "más a la izquierda o más arriba" y 1 el "más a la derecha o abajo" (según el orden).
        escala : float, opcional
            Factor para alargar o acortar la perpendicular. Por defecto, 2.0 para que la perpendicular
            tenga la misma longitud que el segmento.
        color : Manim color
            Color de la línea. Por defecto, YELLOW.
        stroke_width : float, opcional
            Grosor de la línea. Por defecto, 0.5.
        dash_length : float, opcional
            Longitud de cada segmento punteado. Por defecto, 0.1.
        dashed_ratio : float, opcional
            Proporción de línea/vacío en la línea punteada. Por defecto, 0.7.

        Retorna:
        --------
        line_perp : DashedLine
            La línea perpendicular dibujada.
        """
        # Obtenemos los vértices del objeto Polygon
        vertices = figura.get_vertices()
        if len(vertices) != 2:
            raise ValueError("La figura debe tener exactamente 2 puntos para calcular una perpendicular.")

        # Asumimos que los dos vértices son p1 y p2
        p1, p2 = vertices[0], vertices[1]
        # Reordenar los puntos para que "0" sea el que esté más a la izquierda,
        # o en caso de igualdad en x, el que esté más arriba.
        if (p1[0] > p2[0]) or (np.isclose(p1[0], p2[0]) and p1[1] < p2[1]):
            p1, p2 = p2, p1

        # Punto en el segmento según t: t=0 → p1, t=1 → p2, t=0.5 → punto medio.
        punto_en_segmento = p1 + t * (p2 - p1)
        vector = p2 - p1

        # Calcular el vector perpendicular (rotación de 90° en 2D)
        perpendicular = np.array([-vector[1], vector[0], 0])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        # Escalar la longitud de la perpendicular según la longitud del segmento y el parámetro escala.
        perpendicular *= np.linalg.norm(vector) * escala / 4

        extremo1 = punto_en_segmento + perpendicular
        extremo2 = punto_en_segmento - perpendicular

        line_perp = DashedLine(
            extremo1,
            extremo2,
            color=color,
            stroke_width=stroke_width,
            dash_length=dash_length,
            dashed_ratio=dashed_ratio
        )
        self.play(Create(line_perp))
        return line_perp


    def dibujar_mediatriz(self, figura: Polygon, escala: float = 1.0, color=YELLOW):
        """
        Dibuja la mediatriz (la perpendicular en el punto medio) de un segmento definido por un objeto Polygon de dos puntos.

        Parámetros:
        -----------
        figura : Polygon
            Objeto Polygon con exactamente dos vértices.
        escala : float, opcional
            Factor para alargar o acortar la mediatriz. Por defecto, 1.0.
        color : Manim color
            Color de la línea de la mediatriz. Por defecto, YELLOW.

        Retorna:
        --------
        mediatriz : DashedLine
            La línea punteada dibujada como mediatriz.
        """
        # Utilizamos la función anterior con t=0.5 para obtener el punto medio.
        return self.dibujar_linea_perpendicular(figura, t=0.5, escala=escala, color=color)

    def crear_figura(self, puntos_etiquetados, color=WHITE, config=None):
        if config is None:
            config = {}
        mostrar_vertices = config.get("mostrar_vertices", True)
        mostrar_lados = config.get("mostrar_lados", True)

        vertices = []
        etiquetas = []
        for tag, x, y in puntos_etiquetados:
            vertices.append(np.array([x, y, 0]))
            etiquetas.append(tag)

        figura = Polygon(*vertices, color=color)
        self.play(Create(figura))

        if mostrar_vertices:
            self.etiquetar_vertices(vertices, etiquetas)

        if mostrar_lados:
            self.etiquetar_lados(vertices)
        return figura

    def etiquetar_vertices(self, vertices, etiquetas, color=WHITE, offset=0.3):
        puntos = VGroup()
        centroid = np.mean(vertices, axis=0)
        for p, etiqueta in zip(vertices, etiquetas):
            dot = Dot(p, color=color)
            direccion = p - centroid
            if np.linalg.norm(direccion) != 0:
                direccion = direccion / np.linalg.norm(direccion)
            else:
                direccion = UP
            label = MathTex(etiqueta).scale(0.6).move_to(p + direccion * offset)
            puntos.add(VGroup(dot, label))
        self.play(*[FadeIn(p) for p in puntos])

    def etiquetar_lados(self, vertices, color=WHITE, buff=0.3):
        if len(vertices) == 1: return
        lados = VGroup()
        centroid = np.mean(vertices, axis=0)
        n = len(vertices)
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            punto_medio = (p1 + p2) / 2
            direccion = punto_medio - centroid
            if np.linalg.norm(direccion) != 0:
                direccion = direccion / np.linalg.norm(direccion)
            else:
                direccion = UP
            desplazado = punto_medio + direccion * buff
            distancia = np.linalg.norm(p2 - p1)
            texto = self.formato_lindo(distancia)
            etiqueta = MathTex(texto).scale(0.6).move_to(desplazado)
            lados.add(etiqueta)
        self.play(*[FadeIn(l) for l in lados])

    @staticmethod
    def formato_lindo(numero):
        """
        Devuelve una representación limpia del número para mostrar como etiqueta en un lado de un polígono.

        - Si el número coincide (dentro de una tolerancia) con un múltiplo de una raíz cuadrada irracional,
          lo expresa en notación LaTeX (ej. '\\sqrt{2}', '2\\sqrt{3}', etc.).
        - Si la raíz (o su múltiplo) es casi un número entero (redonda), se descarta la notación radical
          y se devuelve el número en formato decimal.
        - Si no se encuentra coincidencia, devuelve el número en formato decimal con hasta 2 decimales,
          eliminando ceros innecesarios.

        Parámetros:
        ----------
        numero : float
            Longitud o valor a formatear.

        Retorna:
        -------
        str
            Cadena en formato LaTeX o numérico.
        """
        # Precalcula las raíces cuadradas para bases de 2 a 99
        raices = {i: math.sqrt(i) for i in range(2, 100)}
        tolerancia = 1e-4
        max_multiplo = 10

        for base, raiz_val in raices.items():
            # Si la raíz es casi entera (redonda), la descartamos
            if math.isclose(raiz_val, round(raiz_val), rel_tol=tolerancia):
                continue
            for k in range(1, max_multiplo + 1):
                estimado = k * raiz_val
                if math.isclose(numero, estimado, rel_tol=tolerancia):
                    if k == 1:
                        return rf"\sqrt{{{base}}}"
                    else:
                        return rf"{k}\sqrt{{{base}}}"
        return f"{numero:.2f}".rstrip("0").rstrip(".")





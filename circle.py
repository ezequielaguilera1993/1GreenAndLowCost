from manim import *
import math
import numpy as np  # Se usa para cálculos vectoriales

FAST_RUN = True  # Si es True, las animaciones se ejecutan casi instantáneamente


class TrianguloConIncentro(Scene):
    def play(self, *animations, run_time=1, **kwargs):
        if FAST_RUN:  # Si FAST_RUN está activado, fuerza run_time a 0.01
            run_time = 0.01
        return super().play(*animations, run_time=run_time, **kwargs)

    def construct(self):
        self.crear_ejes_con_camara(x_range=[-10, 10, 1], y_range=[-7, 7, 1])  # Crea y ajusta el plano cartesiano
        puntos = [("A", 0, 0), ("B", 3, 0), ("C", 3, 3)]  # Vértices del polígono (etiqueta, x, y)
        self.crear_poligono(puntos)  # Crea y anima el polígono
        self.wait(3)  # Espera 3 segundos antes de finalizar la escena

    def crear_ejes_con_camara(self, x_range, y_range):
        ejes = NumberPlane(
            x_range=x_range,  # Rango del eje x
            y_range=y_range,  # Rango del eje y
            background_line_style={
                "stroke_opacity": 0.4,
                "stroke_color": LIGHT_GRAY,  # Color gris clarito para la cuadrícula
                "stroke_width": 1  # Grosor de las líneas de fondo
            },
            axis_config={
                # "include_numbers": True,
                "stroke_color": LIGHT_GRAY,  # Color gris clarito para los ejes
                "stroke_width": 1,  # Grosor de los ejes
            }
        )
        plane_width = ejes.width  # Ancho del plano
        plane_height = ejes.height  # Alto del plano
        aspect_ratio = config.frame_width / config.frame_height  # Relación de aspecto de la cámara
        if plane_width / plane_height > aspect_ratio:  # Ajusta el encuadre según la relación de aspecto
            self.camera.frame_width = plane_width
            self.camera.frame_height = plane_width / aspect_ratio
        else:
            self.camera.frame_height = plane_height
            self.camera.frame_width = plane_height * aspect_ratio
        self.play(Create(ejes), run_time=0.5)  # Anima la creación del plano (ejes)

    def crear_poligono(self, puntos_etiquetados, color=WHITE, config=None):
        if config is None:  # Si no se pasa una configuración, se usa un diccionario vacío
            config = {}
        mostrar_vertices = config.get("mostrar_vertices", True)  # Control para mostrar vértices
        mostrar_lados = config.get("mostrar_lados", True)  # Control para mostrar longitudes de lados

        vertices = []  # Lista para almacenar las coordenadas 3D de los vértices
        etiquetas = []  # Lista para almacenar las etiquetas de cada vértice
        for tag, x, y in puntos_etiquetados:  # Recorre cada punto etiquetado
            vertices.append(np.array([x, y, 0]))  # Convierte a coordenadas 3D
            etiquetas.append(tag)  # Guarda la etiqueta correspondiente

        poligono = Polygon(*vertices, color=color)  # Crea el polígono con los vértices
        self.play(Create(poligono))  # Anima la creación del polígono

        if mostrar_vertices:
            etiquetas_mobs = self.etiquetar_vertices(vertices, etiquetas)  # Crea etiquetas para los vértices
            self.play(*[FadeIn(p) for p in etiquetas_mobs])  # Anima su aparición

        if mostrar_lados:
            lados = self.etiquetar_lados(vertices)  # Crea etiquetas para las longitudes de los lados
            self.play(*[FadeIn(l) for l in lados])  # Anima su aparición

    def etiquetar_vertices(self, vertices, etiquetas, color=WHITE, offset=0.3):
        puntos = VGroup()  # Grupo para agrupar los puntos y sus etiquetas
        centroid = np.mean(vertices, axis=0)  # Calcula el centroide del polígono
        for p, etiqueta in zip(vertices, etiquetas):  # Itera sobre cada vértice y su etiqueta
            dot = Dot(p, color=color)  # Crea un punto en la posición del vértice
            # Calcula la dirección desde el centroide hacia el vértice (para colocar la etiqueta afuera)
            direccion = p - centroid
            if np.linalg.norm(direccion) != 0:
                direccion = direccion / np.linalg.norm(direccion)  # Normaliza el vector
            else:
                direccion = UP
            # Coloca la etiqueta a una distancia 'offset' fuera del vértice en la dirección calculada
            label = MathTex(etiqueta).scale(0.6).move_to(p + direccion * offset)
            puntos.add(VGroup(dot, label))  # Agrupa el punto y la etiqueta
        return puntos

    def etiquetar_lados(self, vertices, color=WHITE, buff=0.3):
        lados = VGroup()  # Grupo para almacenar las etiquetas de los lados
        centroid = np.mean(vertices, axis=0)  # Calcula el centroide del polígono
        n = len(vertices)
        for i in range(n):  # Recorre cada par de vértices consecutivos
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]  # Conecta el último vértice con el primero
            punto_medio = (p1 + p2) / 2  # Calcula el punto medio del lado
            # Calcula la dirección desde el centroide hacia el punto medio para colocar la etiqueta afuera
            direccion = punto_medio - centroid
            if np.linalg.norm(direccion) != 0:
                direccion = direccion / np.linalg.norm(direccion)
            else:
                direccion = UP
            desplazado = punto_medio + direccion * buff  # Desplaza el punto medio en la dirección calculada
            distancia = np.linalg.norm(p2 - p1)  # Calcula la longitud del lado
            texto = self.formato_lindo(distancia)  # Formatea la longitud de manera limpia
            etiqueta = MathTex(texto).scale(0.6).move_to(desplazado)  # Crea y posiciona la etiqueta
            lados.add(etiqueta)
        return lados

    def formato_lindo(self, numero):
        return f"{numero:.1f}".rstrip("0").rstrip(".")  # Formatea el número: máximo 1 decimal sin ceros innecesarios

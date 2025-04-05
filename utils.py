from manim import *
import math
import numpy as np
from manim.utils.color import color_to_rgb, rgb_to_color
import random


class Poligono(Polygon):
    def __init__(self, vertices, etiquetas, **kwargs):
        super().__init__(*vertices, **kwargs)
        self.etiquetas = etiquetas
        self.mapeo = {etiqueta: vertice for etiqueta, vertice in zip(etiquetas, vertices)}

    def get_vertice(self, nombre):
        if nombre not in self.mapeo:
            raise ValueError(f"El vértice '{nombre}' no existe.")
        coord = self.mapeo[nombre]
        return (coord[0], coord[1])

    def get_tamaño_lado(self, nombre_lado):
        if len(nombre_lado) != 2:
            raise ValueError("El nombre del lado debe tener exactamente dos letras (por ejemplo, 'AB').")
        a, b = nombre_lado
        if a not in self.mapeo or b not in self.mapeo:
            raise ValueError(f"Uno o ambos vértices '{a}', '{b}' no existen.")
        p1 = self.mapeo[a]
        p2 = self.mapeo[b]
        return np.linalg.norm(p2 - p1)

    def get_length(self):
        """Calcula el perímetro (longitud total) del polígono."""
        vertices = self.get_vertices()
        length = 0
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            length += np.linalg.norm(p2 - p1)
        return length

    def get_lado(self, nombre_lado):
        if len(nombre_lado) != 2:
            raise ValueError("El nombre del lado debe tener exactamente dos letras (por ejemplo, 'AB').")
        a, b = nombre_lado
        if a not in self.mapeo or b not in self.mapeo:
            raise ValueError(f"Uno o ambos vértices '{a}', '{b}' no existen.")
        return (self.get_vertice(a), self.get_vertice(b))



class Tools(Scene):
    def invertir_color(color):
        rgb = np.array(color_to_rgb(color))
        inv_rgb = 1 - rgb
        return rgb_to_color(inv_rgb)

    @staticmethod
    def seleccionar_punto(hora, puntos):
        angulo_deg = 90 - (hora % 12) * 30
        angulo_rad = math.radians(angulo_deg)
        direccion = np.array([math.cos(angulo_rad), math.sin(angulo_rad)])
        max_dot = -float('inf')
        punto_seleccionado = None
        for p in puntos:
            dot = np.dot(np.array(p), direccion)
            if dot > max_dot:
                max_dot = dot
                punto_seleccionado = p
        return punto_seleccionado

    def interseccion_figuras(self, f1, f2):
        # Caso especial para círculos
        if isinstance(f1, CirculoConAccesos) and isinstance(f2, CirculoConAccesos):
            c1 = f1._centro[:2]
            c2 = f2._centro[:2]
            r1 = f1._radio
            r2 = f2._radio
            d = np.linalg.norm(c2 - c1)
            if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
                return []
            a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
            h = math.sqrt(r1 ** 2 - a ** 2)
            p3 = c1 + a * (c2 - c1) / d
            offset = h * np.array([-(c2[1] - c1[1]) / d, (c2[0] - c1[0]) / d])
            i1 = p3 + offset
            i2 = p3 - offset
            return [i1.tolist(), i2.tolist()]

        # Caso general (segmentos)
        def segmentos(figura):
            v = figura.get_vertices()
            return [(v[i], v[(i + 1) % len(v)]) for i in range(len(v))]

        def interseccion_segmentos(a1, a2, b1, b2):
            da = a2 - a1
            db = b2 - b1
            dp = a1 - b1
            dap = np.array([-da[1], da[0], 0])
            denom = np.dot(dap, db)
            if abs(denom) < 1e-8:
                return None
            num = np.dot(dap, dp)
            inter = (num / denom) * db + b1

            def dentro(p, v1, v2):
                return all(np.min([v1[i], v2[i]]) - 1e-8 <= p[i] <= np.max([v1[i], v2[i]]) + 1e-8 for i in range(2))

            if dentro(inter, a1, a2) and dentro(inter, b1, b2):
                return [inter[0], inter[1]]
            return None

        intersecciones = []
        for s1 in segmentos(f1):
            for s2 in segmentos(f2):
                p = interseccion_segmentos(s1[0], s1[1], s2[0], s2[1])
                if p is not None:
                    intersecciones.append(p)
        return intersecciones

    def dibujar_linea_perpendicular(self, figura, t, escala=2.0, color=YELLOW, stroke_width=0.5,
                                    dash_length=0.1, dashed_ratio=0.7):
        # Función de referencia para líneas dash
        vertices = figura.get_vertices()
        if len(vertices) != 2:
            raise ValueError("La figura debe tener exactamente 2 puntos para calcular una perpendicular.")
        p1, p2 = vertices[0], vertices[1]
        if (p1[0] > p2[0]) or (np.isclose(p1[0], p2[0]) and p1[1] < p2[1]):
            p1, p2 = p2, p1
        punto_en_segmento = p1 + t * (p2 - p1)
        vector = p2 - p1
        perpendicular = np.array([-vector[1], vector[0], 0])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        perpendicular *= np.linalg.norm(vector) * escala / 4
        extremo1 = punto_en_segmento + perpendicular
        extremo2 = punto_en_segmento - perpendicular
        line_perp = DashedLine(extremo1, extremo2, color=color, stroke_width=stroke_width,
                               dash_length=dash_length, dashed_ratio=dashed_ratio)
        self.play(Create(line_perp))
        return line_perp

    def dibujar_mediatriz(self, figura, escala=1.0, color=YELLOW):
        return self.dibujar_linea_perpendicular(figura, t=0.5, escala=escala, color=color)


class Scene(Scene):
    def crear_ejes_con_camara(self, x_range, y_range):
        ejes = NumberPlane(
            x_range=x_range,
            y_range=y_range,
            background_line_style={"stroke_opacity": 0, "stroke_color": LIGHT_GRAY, "stroke_width": 0.8},
            axis_config={"stroke_color": LIGHT_GRAY, "stroke_width": 0.8,
                         "include_ticks": True, "include_numbers": False}
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
        return ejes


class CrearFiguras(Scene):

    def dibujar_circulo_con_compas_juntos(self, r=2, color=BLUE, stroke_width=2.0, run_time=4):
        """
        Dibuja un círculo con la animación normal de Manim (Create),
        y superpone un compás de dos palitos (uno fijo y otro rotante) en el centro.
        Ambos se animan en paralelo para que parezca que el compás dibuja el círculo.

        Parámetros:
          r           : Radio del círculo
          color       : Color del círculo y del compás
          stroke_width: Grosor de las líneas
          run_time    : Duración de la animación total (en segundos)
        """
        # 1. Crear el círculo en el centro de la pantalla
        circle = Circle(radius=r, color=color, stroke_width=stroke_width)
        circle.move_to(ORIGIN)  # Asegura que el centro sea (0,0). Cambia si deseas otro lugar.
        center = circle.get_center()  # Esto debería ser [0,0,0] si no lo moviste a otro lado.

        # 2. Crear los elementos del compás
        #    a) Un pivot en el centro (la "unión" de ambos palitos)
        pivot = Dot(center, color=color)

        #    b) Palito fijo: por ejemplo, vertical y más corto
        #       (de -0.3*r a +0.3*r alrededor del centro para que se vea anclado)
        palito_fijo = Line(
            start=center + DOWN * 0.3 * r,
            end=center + UP * 0.3 * r,
            color=color,
            stroke_width=stroke_width
        )

        #    c) Palito rotante: desde el centro hasta la periferia a la derecha
        palito_rotante = Line(
            start=center,
            end=center + RIGHT * r,
            color=color,
            stroke_width=stroke_width
        )

        # Agrupamos pivot y palitos en un VGroup
        compas = VGroup(pivot, palito_fijo, palito_rotante)

        # 3. Animar todo junto en un solo self.play
        self.play(
            # El círculo se dibuja con la animación normal de Manim
            Create(circle),
            # El compás aparece (los dos palitos y el pivot)
            Create(compas),
            # Mientras tanto, el palito rotante da una vuelta completa (360°) alrededor del centro
            Rotate(palito_rotante, angle=TAU, about_point=center, rate_func=linear),
            run_time=run_time
        )

        # 4. (Opcional) Esperar un momento, o remover el compás
        # self.play(FadeOut(compas))  # si deseas ocultar el compás y dejar solo el círculo
        # self.wait()

        return circle, compas

    def crear_punto(self, coords, color=BLUE, radius=0.03, stroke_width=0.3, fill_opacity=1.0, animar=False, **kwargs):
        if len(coords) == 2:
            coords = np.array([coords[0], coords[1], 0])
        punto = Dot(point=coords, color=color, radius=radius, stroke_width=stroke_width,
                    fill_opacity=fill_opacity, **kwargs)
        if animar:
            self.play(Create(punto))
        return punto

    def _crear_punto(self, coords, **kwargs):
        # Se filtran los argumentos que sean None para no sobreescribir los defaults de crear_punto.
        params = {k: v for k, v in kwargs.items() if v is not None}
        return self.crear_punto(coords, animar=False, **params)

    def dibujar_circulo(self, *puntos, c=None, d=None, r=None, color=BLUE, stroke_width=2.0,
                        dashed=False, dash_length=0.5,
                        dashed_ratio=0.7, num_dashes=None):
        def to_array(p):
            arr = np.array(p)
            if arr.shape == (2,):
                return np.array([arr[0], arr[1], 0.0])
            return arr

        if d is not None and r is not None:
            raise ValueError("No podés usar 'd' y 'r' al mismo tiempo. Elegí uno.")
        puntos = [to_array(p) for p in puntos]
        if c is not None:
            c = to_array(c)
        if len(puntos) > 3:
            raise ValueError("Se permite como máximo 3 puntos.")
        if c is not None:
            if len(puntos) > 1:
                raise ValueError("Si usás 'c', solo podés pasar 1 punto o 'd'/'r'.")
            centro = c
            self.crear_punto(centro, color=color)
            if r is not None:
                radio = r
            elif d is not None:
                radio = d / 2
            elif len(puntos) == 1:
                radio = np.linalg.norm(puntos[0] - c)
            else:
                raise ValueError("Si usás 'c', tenés que pasar también 'r', 'd' o un punto.")
        else:
            if len(puntos) == 1:
                centro = puntos[0]
                if r is not None:
                    radio = r
                elif d is not None:
                    radio = d / 2
                else:
                    raise ValueError("Si pasás un solo punto, necesitás pasar 'r' o 'd'.")
            elif len(puntos) == 2:
                if r is not None or d is not None:
                    raise ValueError("No se puede usar 'r' o 'd' si pasás 2 puntos.")
                p1, p2 = puntos
                centro = (p1 + p2) / 2
                radio = np.linalg.norm(p2 - p1) / 2
            elif len(puntos) == 3:
                if r is not None or d is not None:
                    raise ValueError("No se puede usar 'r' o 'd' si pasás 3 puntos.")
                A, B, C = puntos
                a = B - A
                b = C - A
                ab = np.cross(a, b)
                if np.linalg.norm(ab) < 1e-6:
                    raise ValueError("Los 3 puntos son colineales; no se puede definir un círculo.")
                d_mat = np.array([[np.dot(a, a) / 2, np.dot(a, b)],
                                  [np.dot(b, a), np.dot(b, b) / 2]])
                mat = np.array([[a[0], a[1]], [b[0], b[1]]])
                sol = np.linalg.solve(mat.T, d_mat.T[0])
                centro = A + sol[0] * a + sol[1] * b
                radio = np.linalg.norm(centro - A)
            else:
                raise ValueError("Entrada no válida. Usá 1, 2 o 3 puntos, o c con r/d/punto.")
        # Crear el círculo original
        circulo = CirculoConAccesos(center=centro, radius=radio, color=color, stroke_width=stroke_width)
        if dashed:
            # Calculamos la circunferencia (2*pi*radio) y definimos num_dashes si no se pasa
            longitud = 2 * math.pi * radio
            if num_dashes is None:
                num_dashes = max(3, int(longitud / dash_length))
            circuloConDashs = DashedCircleWithProperties(circulo, dashed_ratio=dashed_ratio, num_dashes=num_dashes)
            self.play(Create(circuloConDashs))
        else:
            self.play(Create(circulo))
        return circulo

    def dibujar_circulo_auxiliar(self, *puntos, c=None, d=None, r=None, color=BLUE, stroke_width=2.0):
        """
        Función auxiliar que dibuja el círculo en versión dash, similar a crear_figura_auxiliar.
        """
        return self.dibujar_circulo(*puntos, c=c, d=d, r=r, color=color, stroke_width=stroke_width, dashed=True, )

    import random

    def crear_figura(self, puntos_etiquetados, color=WHITE, etiquetar_vertices=True, mostrar_lados=True,
                     dashed=False, stroke_width=5, con_labels=True, fill_random=False, fill_after=True):
        vertices = []
        etiquetas = []
        letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, punto in enumerate(puntos_etiquetados):
            if len(punto) == 3:
                # Si el primer elemento es string, asumimos (etiqueta, x, y)
                if isinstance(punto[0], str):
                    tag, x, y = punto
                # Sino, si el último elemento es string, asumimos (x, y, etiqueta)
                elif isinstance(punto[-1], str):
                    x, y, tag = punto
                else:
                    # Si ninguno es string, se asigna etiqueta por defecto
                    x, y, _ = punto
                    tag = letras[i % len(letras)]
            elif len(punto) == 2:
                x, y = punto
                tag = letras[i % len(letras)]
            else:
                raise ValueError("Cada punto debe ser (etiqueta, x, y), (x, y, etiqueta) o (x, y)")
            vertices.append(np.array([x, y, 0]))
            etiquetas.append(tag)
        figura = Poligono(vertices, etiquetas, color=color, stroke_width=stroke_width)

        if fill_random:
            random_color = rgb_to_color([random.random(), random.random(), random.random()])
            if not fill_after:
                figura.set_fill(random_color, opacity=0.5)

        if dashed:
            longitud = figura.get_length()
            num_dashes = max(40, int(longitud))
            figuraConDashs = DashedVMobject(figura, dashed_ratio=0.4, num_dashes=num_dashes, stroke_width=stroke_width)
            self.play(Create(figuraConDashs))
        else:
            self.play(Create(figura))

        if etiquetar_vertices:
            if con_labels:
                self.etiquetar_vertices(vertices, etiquetas)
            else:
                self.dibujar_puntos(vertices, color)
        if mostrar_lados:
            self.etiquetar_lados(vertices, etiquetas)

        if fill_random and fill_after:
            self.play(figura.animate.set_fill(random_color, opacity=0.5))

        return figura

    def crear_figura_coloreada(self, puntos_etiquetados, color=WHITE, etiquetar_vertices=True, mostrar_lados=True,
                               dashed=False, stroke_width=5, con_labels=True):
        return self.crear_figura(
            puntos_etiquetados,
            color=color,
            etiquetar_vertices=etiquetar_vertices,
            mostrar_lados=mostrar_lados,
            dashed=dashed,
            stroke_width=stroke_width,
            con_labels=con_labels,
            fill_random=True
        )

    def crear_figura_coloreada_sin_labels(self, puntos_etiquetados, color=WHITE, dashed=False, stroke_width=2,
                                          mostrar_puntos=True):
        return self.crear_figura(
            puntos_etiquetados,
            color=color,
            etiquetar_vertices=mostrar_puntos,
            mostrar_lados=False,
            dashed=dashed,
            stroke_width=stroke_width,
            con_labels=False,  # Sin etiquetas en los vértices
            fill_random=True,  # Activa el relleno aleatorio
            fill_after=True  # Se anima el relleno al finalizar el resto
        )

    def dibujar_puntos(self, vertices, color=WHITE):
        dots = VGroup()
        for p in vertices:
            # Usamos _crear_punto para obtener el Dot sin animarlo individualmente.
            dot = self._crear_punto(p, color=color)
            dots.add(dot)
        # Animamos todos los puntos juntos.
        self.play(Create(dots))
        return dots

    def crear_figura_sin_labels(self, puntos_etiquetados, color=WHITE, dashed=False,
                                stroke_width=2, mostrar_puntos=True):
        return self.crear_figura(
            puntos_etiquetados,
            color=color,
            etiquetar_vertices=mostrar_puntos,  # Por defecto se muestran los puntos
            mostrar_lados=False,
            dashed=dashed,
            stroke_width=stroke_width,
            con_labels=False  # No se muestran etiquetas en los vértices
        )

    def crear_figura_auxiliar(self, puntos_etiquetados, color=WHITE, dashed=True,
                              stroke_width=2, mostrar_puntos=True):
        return self.crear_figura_sin_labels(
            puntos_etiquetados,
            color=color,
            dashed=dashed,
            stroke_width=stroke_width,
            mostrar_puntos=mostrar_puntos
        )

    def etiquetar_vertices(self, vertices, etiquetas, color=WHITE, offset=0.3):
        puntos = VGroup()
        centroid = np.mean(vertices, axis=0)
        for p, etiqueta in zip(vertices, etiquetas):
            dot = self.crear_punto(p, color=color)
            direccion = p - centroid
            if np.linalg.norm(direccion) != 0:
                direccion = direccion / np.linalg.norm(direccion)
            else:
                direccion = UP
            label = MathTex(etiqueta).scale(0.6).move_to(p + direccion * offset)
            puntos.add(VGroup(dot, label))
        self.play(*[FadeIn(p) for p in puntos])
        return puntos

    def etiquetar_lados(self, vertices, etiquetas, color=WHITE, buff=0.3):
        if len(vertices) == 1:
            return VGroup()
        lados = VGroup()
        centroid = np.mean(vertices, axis=0)
        n = len(vertices)
        # Modo selectivo: se activa si al menos un vértice tiene la etiqueta "le"
        selective = any(tag == "le" for tag in etiquetas)
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            # En modo selectivo, solo se etiqueta el lado si ambos extremos tienen "le"
            if selective and not (etiquetas[i] == "le" and etiquetas[(i + 1) % n] == "le"):
                continue
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
        return lados

    @staticmethod
    def formato_lindo(numero):
        raices = {i: math.sqrt(i) for i in range(2, 100)}
        tolerancia = 1e-4
        max_multiplo = 10
        for base, raiz_val in raices.items():
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


class CirculoConAccesos(Circle):
    def __init__(self, center, radius, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)
        self._centro = center
        self._radio = radius

    @property
    def get_l(self):
        p = self._centro + np.array([-self._radio, 0, 0])
        return [p[0], p[1]]

    @property
    def get_r(self):
        p = self._centro + np.array([self._radio, 0, 0])
        return [p[0], p[1]]

    @property
    def get_u(self):
        p = self._centro + np.array([0, self._radio, 0])
        return [p[0], p[1]]

    @property
    def get_d(self):
        p = self._centro + np.array([0, -self._radio, 0])
        return [p[0], p[1]]

    @property
    def get_c(self):
        return [self._centro[0], self._centro[1]]


class DashedCircleWithProperties(DashedVMobject):
    def __init__(self, original, **kwargs):
        self.original = original
        super().__init__(original.copy(), **kwargs)

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.original, attr)

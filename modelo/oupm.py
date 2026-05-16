"""
modelo/oupm.py
Modelo de Universo Abierto (OUPM) — cap. 18.2 del libro.

Variables latentes:
  #Abonado ~ Poisson(lambda_prior)
  Nombre(a), Telefono(a), Ciudad(a) ~ priors
  Source(r) ~ UniformChoice({abonados})   ← variable de asociación
  Obs(r) ~ NoisyModel(atributos(Source(r)))

La probabilidad de un "mundo" W es el producto de todas las
probabilidades condicionales — exactamente la ecuación (18) del cap.
"""

import math
import random
from datos.registro import (Registro, similitud_cadena,
                            similitud_telefono, misma_subred)


# ──────────────────────────────────────────────
# Funciones de probabilidad (CPTs implementadas desde cero)
# ──────────────────────────────────────────────

def _log_poisson(k: int, lam: float) -> float:
    """log P(X=k) para X ~ Poisson(lambda). Implementada desde cero."""
    if k < 0 or lam <= 0:
        return -math.inf
    log_fact_k = sum(math.log(i) for i in range(1, k + 1)) if k > 0 else 0.0
    return k * math.log(lam) - lam - log_fact_k


def _log_prob_obs_dado_source(registro: Registro,
                               abonado: "Abonado") -> float:
    """
    log P(registro | abonado) — modelo de observación ruidosa.
    Implementa:  Text(r) ~ NoisyString(Nombre(Source(r)))  [cap. 18.2]

    Usa similitudes como proxy de probabilidad:
      P(obs | true) = epsilon + (1-epsilon) * similitud(obs, true)
    """
    sim_nombre   = similitud_cadena(registro.nombre_obs, abonado.nombre_real)
    sim_telefono = similitud_telefono(registro.telefono_obs, abonado.telefono_real)
    sim_ip       = misma_subred(registro.ip_obs, abonado.ip_real)
    sim_ciudad   = similitud_cadena(registro.ciudad_obs, abonado.ciudad_real)

    epsilon = 0.05
    p_nombre   = epsilon + (1 - epsilon) * sim_nombre
    p_telefono = epsilon + (1 - epsilon) * sim_telefono
    p_ip       = epsilon + (1 - epsilon) * sim_ip
    p_ciudad   = epsilon + (1 - epsilon) * sim_ciudad

    return (0.40 * math.log(p_nombre) +
            0.30 * math.log(p_telefono) +
            0.20 * math.log(p_ip) +
            0.10 * math.log(p_ciudad))


def _log_prob_uniforme_source(n_abonados: int) -> float:
    """log P(Source(r) = a) = log(1/n_abonados) — prior uniforme."""
    if n_abonados <= 0:
        return -math.inf
    return -math.log(n_abonados)


# ──────────────────────────────────────────────
# Clase Abonado (objeto latente)
# ──────────────────────────────────────────────

class Abonado:
    """
    Un abonado real hipotético en el mundo actual.
    En el OUPM, cada abonado es un objeto generado por:
      #Abonado ~ Poisson(lambda)   [number statement, cap. 18.2]

    NOTA: el ID ya no viene de un contador de clase (_contador eliminado).
    Ahora lo asigna ModeloOUPM via _siguiente_id, evitando el bug de
    IDs desfasados al clonar mundos múltiples veces.
    """

    def __init__(self, nombre_real: str, telefono_real: str,
                 ip_real: str, ciudad_real: str,
                 es_fraudulento: bool = False,
                 id_forzado: int = None):
        # id_forzado permite que clonar() preserve el mismo ID
        self.id            = id_forzado if id_forzado is not None else -1
        self.nombre_real   = nombre_real
        self.telefono_real = telefono_real
        self.ip_real       = ip_real
        self.ciudad_real   = ciudad_real
        self.es_fraudulento = es_fraudulento

    def log_prob_genera(self, registro: Registro) -> float:
        """P(registro | este abonado lo generó)."""
        return _log_prob_obs_dado_source(registro, self)

    def __repr__(self):
        return (f"Abonado(id={self.id}, nombre={self.nombre_real!r}, "
                f"fraude={self.es_fraudulento})")


# ──────────────────────────────────────────────
# Clase ModeloOUPM — el "mundo" completo
# ──────────────────────────────────────────────

class ModeloOUPM:
    """
    Representa un mundo posible W en el OUPM.
    Un mundo consiste en:
      - Un conjunto de abonados hipotéticos {a_1, ..., a_k}
      - Una asignación Source: Registro → Abonado

    La probabilidad de este mundo es (cap. 18.2):
      P(W) = P(#Abonados=k) * prod_a P(atributos(a)) *
             prod_r P(Source(r)) * P(obs(r) | Source(r))

    El motor MCMC explora el espacio de mundos posibles
    cambiando Source(r) y añadiendo/eliminando abonados.
    """

    LAMBDA_PRIOR = 5.0

    def __init__(self, registros: list):
        self.registros   = registros
        self.abonados: list = []
        self.source: dict  = {}
        # Contador local de IDs — reemplaza Abonado._contador (bug fix)
        self._siguiente_id = 1

    def _nuevo_abonado(self, nombre_real, telefono_real,
                       ip_real, ciudad_real,
                       es_fraudulento=False) -> Abonado:
        """Crea un Abonado con ID único dentro de este mundo."""
        ab = Abonado(nombre_real=nombre_real,
                     telefono_real=telefono_real,
                     ip_real=ip_real,
                     ciudad_real=ciudad_real,
                     es_fraudulento=es_fraudulento,
                     id_forzado=self._siguiente_id)
        self._siguiente_id += 1
        return ab

    # ── Inicialización ──────────────────────────────

    def inicializar_aleatorio(self, rng: random.Random,
                               n_abonados_inicial: int = None):
        """
        Crea un mundo inicial aleatorio.
        Número de abonados ~ Poisson(lambda), luego asigna Source al azar.
        """
        self._siguiente_id = 1   # reiniciar contador local
        if n_abonados_inicial is None:
            n_abonados_inicial = max(1, int(rng.gauss(
                self.LAMBDA_PRIOR, math.sqrt(self.LAMBDA_PRIOR))))
            n_abonados_inicial = min(n_abonados_inicial, 2 * len(self.registros))

        self.abonados = []
        for _ in range(n_abonados_inicial):
            r_ref = rng.choice(self.registros)
            self.abonados.append(self._nuevo_abonado(
                nombre_real=r_ref.nombre_obs,
                telefono_real=r_ref.telefono_obs,
                ip_real=r_ref.ip_obs,
                ciudad_real=r_ref.ciudad_obs,
            ))

        for reg in self.registros:
            self.source[reg.id] = rng.choice(self.abonados)

    # ── Probabilidad del mundo ──────────────────────

    def log_prob_mundo(self) -> float:
        """
        Calcula log P(W) completo — usado solo para inicialización
        y para snapshots de diagnóstico.
        En el motor MCMC se usa log_prob_markov_blanket() para eficiencia.
        """
        k = len(self.abonados)
        if k == 0:
            return -math.inf

        lp = _log_poisson(k, self.LAMBDA_PRIOR)
        log_p_source = _log_prob_uniforme_source(k)

        for reg in self.registros:
            ab = self.source.get(reg.id)
            if ab is None or ab not in self.abonados:
                return -math.inf
            lp += log_p_source
            lp += _log_prob_obs_dado_source(reg, ab)

        return lp

    def log_prob_markov_blanket(self, registros_afectados: list) -> float:
        """
        Calcula la parte de log P(W) que cambia tras un movimiento.

        En cada paso M-H solo uno o pocos registros cambian de abonado,
        por lo que recalcular log P(W) completo es innecesario.
        Este método computa únicamente:

          log P(k) + sum_{r in afectados} [log P(Source(r)) + log P(obs(r)|Source(r))]

        Complejidad: O(|registros_afectados|) en vez de O(|R|).
        Conectado al motor en PropuestaMCMC._calcular_lp_propuesta().
        """
        k = len(self.abonados)
        if k == 0:
            return -math.inf
        lp = _log_poisson(k, self.LAMBDA_PRIOR)
        log_p_source = _log_prob_uniforme_source(k)
        for reg in registros_afectados:
            ab = self.source.get(reg.id)
            if ab is None or ab not in self.abonados:
                return -math.inf
            lp += log_p_source
            lp += _log_prob_obs_dado_source(reg, ab)
        return lp

    # ── Operaciones de grounding ────────────────────

    def grounding(self) -> dict:
        """
        Instancia la red bayesiana correspondiente al mundo actual.
        Retorna estructura de adjacencia: {variable: [padres]}.
        Implementa el 'unrolling' descrito en cap. 18.1.3.
        """
        red = {}
        for ab in self.abonados:
            red[f"Nombre_{ab.id}"]   = []
            red[f"Telefono_{ab.id}"] = []
            red[f"Ciudad_{ab.id}"]   = []
        for reg in self.registros:
            ab = self.source[reg.id]
            red[f"NombreObs_{reg.id}"]   = [f"Nombre_{ab.id}"]
            red[f"TelObs_{reg.id}"]      = [f"Telefono_{ab.id}"]
            red[f"CiudadObs_{reg.id}"]   = [f"Ciudad_{ab.id}"]
            red[f"Source_{reg.id}"]      = list(
                f"Abonado_{a.id}" for a in self.abonados)
        return red

    def clonar(self) -> "ModeloOUPM":
        """
        Crea una copia profunda del mundo actual para el paso M-H.
        Preserva _siguiente_id para que IDs de abonados nuevos no colisionen.
        """
        nuevo = ModeloOUPM(self.registros)
        nuevo._siguiente_id = self._siguiente_id   # hereda el contador
        mapa_ids = {}
        for ab in self.abonados:
            ab_nuevo = Abonado(
                nombre_real=ab.nombre_real,
                telefono_real=ab.telefono_real,
                ip_real=ab.ip_real,
                ciudad_real=ab.ciudad_real,
                es_fraudulento=ab.es_fraudulento,
                id_forzado=ab.id,
            )
            mapa_ids[ab.id] = ab_nuevo
            nuevo.abonados.append(ab_nuevo)
        for rid, ab in self.source.items():
            nuevo.source[rid] = mapa_ids[ab.id]
        return nuevo

    # ── Consultas sobre el mundo ────────────────────

    def clusters_actuales(self) -> dict:
        """Agrupa registros por abonado asignado. {id_abonado: [ids_registros]}"""
        clusters = {ab.id: [] for ab in self.abonados}
        for reg in self.registros:
            ab = self.source[reg.id]
            clusters[ab.id].append(reg.id)
        return clusters

    def __repr__(self):
        return (f"ModeloOUPM(abonados={len(self.abonados)}, "
                f"registros={len(self.registros)})")

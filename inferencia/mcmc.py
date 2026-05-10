"""
inferencia/mcmc.py
Motor de inferencia MCMC para el OUPM — cap. 18.2.2 del libro.

Implementa Metropolis-Hastings con tres tipos de movimientos:
  1. Reasignar: cambiar Source(r) de un registro a otro abonado
  2. Birth:     añadir un nuevo abonado al mundo  ← clave para universo abierto
  3. Death:     eliminar un abonado sin registros asignados

El algoritmo acepta/rechaza cada movimiento con probabilidad:
  α = min(1, P(W') * Q(W|W') / (P(W) * Q(W'|W)))

Todo implementado desde cero — sin librerías de inferencia.
"""

import math
import random
from modelo.oupm import ModeloOUPM, Abonado
from datos.registro import Registro


# ──────────────────────────────────────────────
# Propuestas de movimiento MCMC
# ──────────────────────────────────────────────

class PropuestaMCMC:
    """
    Genera movimientos propuestos en el espacio de mundos posibles.
    Cada movimiento tiene una probabilidad de propuesta Q(W'|W)
    necesaria para calcular la razón de Hastings.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def proponer(self, mundo: ModeloOUPM) -> tuple:
        """
        Elige aleatoriamente un tipo de movimiento y lo aplica.
        Retorna (mundo_nuevo, log_q_forward, log_q_backward, tipo)
        donde log_q_* son los log de las probabilidades de propuesta.
        """
        # Pesos de cada tipo de movimiento
        n_ab = len(mundo.abonados)
        n_reg = len(mundo.registros)

        # Calcular cuántos abonados quedarían vacíos si hacemos death
        vacios = self._abonados_vacios(mundo)

        peso_reasignar = 0.60
        peso_birth     = 0.25
        peso_death     = 0.15 if vacios else 0.0

        total = peso_reasignar + peso_birth + peso_death
        u = self.rng.random() * total

        if u < peso_reasignar:
            return self._mover_reasignar(mundo)
        elif u < peso_reasignar + peso_birth:
            return self._mover_birth(mundo)
        else:
            return self._mover_death(mundo, vacios)

    def _abonados_vacios(self, mundo: ModeloOUPM) -> list:
        """Abonados sin ningún registro asignado (candidatos para death)."""
        asignados = set(ab.id for ab in mundo.source.values())
        return [ab for ab in mundo.abonados if ab.id not in asignados]

    def _mover_reasignar(self, mundo: ModeloOUPM) -> tuple:
        """
        Movimiento 1 — Reasignar: elige un registro r al azar
        y le asigna un abonado diferente a' al azar.
        Q(W'|W) = 1/n_reg * 1/n_ab  (uniforme sobre r y a')
        """
        mundo_nuevo = mundo.clonar()
        reg = self.rng.choice(mundo_nuevo.registros)
        ab_nuevo = self.rng.choice(mundo_nuevo.abonados)
        mundo_nuevo.source[reg.id] = ab_nuevo

        n_reg = len(mundo.registros)
        n_ab  = len(mundo.abonados)
        log_q_fwd = -math.log(n_reg) - math.log(n_ab)
        log_q_bwd = -math.log(n_reg) - math.log(n_ab)
        return mundo_nuevo, log_q_fwd, log_q_bwd, 'reasignar'

    def _mover_birth(self, mundo: ModeloOUPM) -> tuple:
        """
        Movimiento 2 — Birth: añade un nuevo abonado al mundo.
        Sus atributos se toman de un registro aleatorio (propuesta informada).
        Opcionalmente migra un registro al nuevo abonado.

        Q(W'|W) depende de la probabilidad de escoger este registro base
        y la probabilidad de mover el registro elegido.
        """
        mundo_nuevo = mundo.clonar()
        r_base = self.rng.choice(mundo_nuevo.registros)

        # Crear abonado nuevo con atributos del registro base (con ruido leve)
        ab_nuevo = Abonado(
            nombre_real=r_base.nombre_obs,
            telefono_real=r_base.telefono_obs,
            ip_real=r_base.ip_obs,
            ciudad_real=r_base.ciudad_obs,
        )
        ab_nuevo.id = max((a.id for a in mundo_nuevo.abonados), default=0) + 1
        mundo_nuevo.abonados.append(ab_nuevo)

        # Mover el registro base al nuevo abonado
        mundo_nuevo.source[r_base.id] = ab_nuevo

        n_reg = len(mundo.registros)
        n_ab_nuevo = len(mundo_nuevo.abonados)
        n_ab_viejo = len(mundo.abonados)

        log_q_fwd = -math.log(n_reg)          # escoger r_base
        log_q_bwd = -math.log(n_ab_nuevo)     # death escoge el abonado a eliminar
        return mundo_nuevo, log_q_fwd, log_q_bwd, 'birth'

    def _mover_death(self, mundo: ModeloOUPM, vacios: list) -> tuple:
        """
        Movimiento 3 — Death: elimina un abonado vacío.
        Solo se puede hacer si hay abonados sin registros asignados.
        Q(W'|W) = 1/|vacios| (uniforme sobre abonados vacíos)
        """
        mundo_nuevo = mundo.clonar()
        ab_eliminar_id = self.rng.choice(vacios).id

        # Encontrar el abonado en el mundo clonado
        ab_eliminar = next(a for a in mundo_nuevo.abonados
                          if a.id == ab_eliminar_id)
        mundo_nuevo.abonados.remove(ab_eliminar)

        n_vacios = len(vacios)
        log_q_fwd = -math.log(n_vacios)
        log_q_bwd = -math.log(len(mundo.registros))  # birth eligió r_base
        return mundo_nuevo, log_q_fwd, log_q_bwd, 'death'


# ──────────────────────────────────────────────
# Motor MCMC principal
# ──────────────────────────────────────────────

class MotorMCMC:
    """
    Motor de inferencia Metropolis-Hastings para el OUPM.
    Implementa el algoritmo descrito en cap. 18.2.2:

      'MCMC for OUPMs explores the space of possible worlds.
       A move can add or remove objects, changing the relational
       structure. Each step takes constant time.'

    Parámetros:
      n_iter:   número total de iteraciones MCMC
      burn_in:  iteraciones descartadas al inicio (calentamiento)
      thin:     guardar una muestra cada 'thin' iteraciones
    """

    def __init__(self, modelo: ModeloOUPM,
                 n_iter: int = 2000,
                 burn_in: int = 500,
                 thin: int = 5,
                 semilla: int = 42,
                 verbose: bool = True):
        self.modelo   = modelo
        self.n_iter   = n_iter
        self.burn_in  = burn_in
        self.thin     = thin
        self.rng      = random.Random(semilla)
        self.verbose  = verbose
        self.propuesta = PropuestaMCMC(self.rng)

        # Historial para análisis
        self.muestras: list[dict] = []
        self.log_probs: list[float] = []
        self.tasa_aceptacion_por_tipo: dict = {
            'reasignar': [0, 0],
            'birth':     [0, 0],
            'death':     [0, 0],
        }

    def correr(self) -> "Posterior":
        """
        Ejecuta el MCMC completo y retorna el objeto Posterior.
        """
        # Inicializar mundo
        self.modelo.inicializar_aleatorio(self.rng)
        mundo_actual = self.modelo
        lp_actual = mundo_actual.log_prob_mundo()

        if self.verbose:
            print(f"  Mundo inicial: {len(mundo_actual.abonados)} abonados, "
                  f"log P(W) = {lp_actual:.3f}")

        for it in range(self.n_iter):
            mundo_actual, lp_actual = self._paso_mh(mundo_actual, lp_actual)

            # Guardar muestra (post burn-in, con thinning)
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                self.muestras.append(self._snapshot(mundo_actual))
                self.log_probs.append(lp_actual)

            if self.verbose and (it + 1) % 500 == 0:
                n_ab = len(mundo_actual.abonados)
                print(f"  Iter {it+1:4d}/{self.n_iter} | "
                      f"abonados={n_ab:3d} | log P={lp_actual:.3f} | "
                      f"muestras={len(self.muestras)}")

        if self.verbose:
            self._imprimir_estadisticas()

        return Posterior(self.muestras, self.log_probs,
                        self.modelo.registros)

    def _paso_mh(self, mundo: ModeloOUPM,
                 lp_mundo: float) -> tuple:
        """
        Un paso de Metropolis-Hastings.
        Implementa:
          α = min(1, P(W')/P(W) * Q(W|W')/Q(W'|W))
          Aceptar W' con probabilidad α.
        """
        try:
            mundo_prop, log_q_fwd, log_q_bwd, tipo = \
                self.propuesta.proponer(mundo)
        except Exception:
            return mundo, lp_mundo

        lp_prop = mundo_prop.log_prob_mundo()

        # Razón de Metropolis-Hastings (en log)
        log_alpha = (lp_prop - lp_mundo) + (log_q_bwd - log_q_fwd)
        log_alpha = min(0.0, log_alpha)  # min(1, alpha) en log

        # Decisión de aceptación
        self.tasa_aceptacion_por_tipo[tipo][1] += 1
        if math.log(self.rng.random() + 1e-300) < log_alpha:
            self.tasa_aceptacion_por_tipo[tipo][0] += 1
            return mundo_prop, lp_prop
        else:
            return mundo, lp_mundo

    def _snapshot(self, mundo: ModeloOUPM) -> dict:
        """Captura el estado actual del mundo para agregar en la posterior."""
        source_snapshot = {
            rid: ab.id for rid, ab in mundo.source.items()
        }
        return {
            'n_abonados': len(mundo.abonados),
            'source': source_snapshot,
            'abonados': {ab.id: {
                'nombre': ab.nombre_real,
                'telefono': ab.telefono_real,
                'ciudad': ab.ciudad_real,
            } for ab in mundo.abonados},
        }

    def _imprimir_estadisticas(self):
        print("\n  Tasas de aceptación por tipo de movimiento:")
        for tipo, (acep, total) in self.tasa_aceptacion_por_tipo.items():
            tasa = acep / total if total > 0 else 0
            print(f"    {tipo:12s}: {acep:4d}/{total:4d} = {tasa:.1%}")
        print(f"  Muestras guardadas: {len(self.muestras)}")


# ──────────────────────────────────────────────
# Clase Posterior — agrega las muestras MCMC
# ──────────────────────────────────────────────

class Posterior:
    """
    Agrega las muestras del MCMC para responder consultas probabilísticas.
    Implementa las cuatro consultas definidas en el planteamiento:
      1. P(Source(rA) = Source(rB)) — ¿misma identidad?
      2. P(#Abonados = k)           — ¿cuántos abonados reales?
      3. Clustering MAP              — agrupación más probable
      4. Convergencia                — diagnóstico del MCMC
    """

    def __init__(self, muestras: list[dict], log_probs: list[float],
                 registros: list[Registro]):
        self.muestras   = muestras
        self.log_probs  = log_probs
        self.registros  = registros
        self.n_muestras = len(muestras)

    def p_misma_identidad(self, id_reg_a: str, id_reg_b: str) -> float:
        """
        P(Source(rA) = Source(rB) | evidencia)
        Fracción de muestras donde A y B están asignados al mismo abonado.
        """
        if self.n_muestras == 0:
            return 0.0
        cuenta = sum(
            1 for m in self.muestras
            if m['source'].get(id_reg_a) == m['source'].get(id_reg_b)
        )
        return cuenta / self.n_muestras

    def p_n_abonados(self, k: int) -> float:
        """P(#Abonados = k | evidencia)"""
        if self.n_muestras == 0:
            return 0.0
        return sum(1 for m in self.muestras
                   if m['n_abonados'] == k) / self.n_muestras

    def distribucion_n_abonados(self) -> dict:
        """Distribución completa sobre el número de abonados."""
        dist = {}
        for m in self.muestras:
            k = m['n_abonados']
            dist[k] = dist.get(k, 0) + 1
        return {k: v / self.n_muestras for k, v in sorted(dist.items())}

    def cluster_map(self) -> dict:
        """
        Estimación MAP: agrupación de registros más frecuente.
        Retorna {id_registro: cluster_id} donde cluster_id es un entero
        comenzando en 0 (los IDs de abonados se renormalizan).
        """
        if not self.muestras:
            return {}

        # Encontrar la muestra con mayor log-prob
        idx_map = max(range(self.n_muestras),
                      key=lambda i: self.log_probs[i])
        muestra_map = self.muestras[idx_map]

        # Renormalizar IDs de abonados a 0, 1, 2, ...
        ids_vistos = {}
        contador = 0
        resultado = {}
        for rid, ab_id in muestra_map['source'].items():
            if ab_id not in ids_vistos:
                ids_vistos[ab_id] = contador
                contador += 1
            resultado[rid] = ids_vistos[ab_id]
        return resultado

    def matriz_cocluster(self) -> dict:
        """
        Matriz de co-clustering: para cada par (rA, rB),
        P(misma identidad | evidencia).
        Útil para visualizar la incertidumbre de identidad.
        """
        ids = [r.id for r in self.registros]
        matriz = {}
        for i, ra in enumerate(ids):
            for rb in ids[i:]:
                p = self.p_misma_identidad(ra, rb)
                matriz[(ra, rb)] = p
                matriz[(rb, ra)] = p
        return matriz

    def resumen(self) -> str:
        """Imprime un resumen legible de la posterior."""
        if not self.muestras:
            return "Sin muestras."
        dist = self.distribucion_n_abonados()
        k_map = max(dist, key=dist.get)
        lineas = [
            f"Muestras MCMC: {self.n_muestras}",
            f"Abonados más probable (MAP): k={k_map} "
            f"(P={dist[k_map]:.1%})",
            "Distribución sobre #Abonados:",
        ]
        for k, p in dist.items():
            barra = '#' * int(p * 30)
            lineas.append(f"  k={k:3d}: {barra:<30s} {p:.1%}")
        return '\n'.join(lineas)

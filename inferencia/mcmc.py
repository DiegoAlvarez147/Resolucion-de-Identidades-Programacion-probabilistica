"""
inferencia/mcmc.py
Motor de inferencia MCMC para el OUPM — cap. 18.2.2 del libro.

Implementa Metropolis-Hastings con tres tipos de movimientos:
  1. Reasignar: cambiar Source(r) de un registro a otro abonado
  2. Birth:     añadir un nuevo abonado al mundo
  3. Death:     eliminar un abonado sin registros asignados

Correcciones respecto a la versión anterior:
  1. Birth/death ahora son movimientos inversos balanceados:
       Q(death | birth) / Q(birth | death) se calcula correctamente.
     Antes el ratio de Hastings para birth era siempre positivo porque
     log_q_bwd < log_q_fwd, haciendo que casi todo birth se aceptara
     y death nunca tuviera chance de equilibrar.
  2. El prior Poisson sobre k se recalcula completo en birth/death
     (son movimientos que cambian la estructura del mundo, no solo Source).
  3. Para reasignar se mantiene la optimización de Markov blanket.
  4. proponer() sigue retornando 5 elementos (compatible con simulacion_vivo).
"""

import math
import random
from modelo.oupm import ModeloOUPM, Abonado
from datos.registro import Registro


# ──────────────────────────────────────────────
# Propuestas de movimiento MCMC
# ──────────────────────────────────────────────

class PropuestaMCMC:

    def __init__(self, rng: random.Random):
        self.rng = rng

    def proponer(self, mundo: ModeloOUPM) -> tuple:
        """
        Retorna (mundo_nuevo, log_q_fwd, log_q_bwd, tipo, regs_afectados).
        """
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
        asignados = set(ab.id for ab in mundo.source.values())
        return [ab for ab in mundo.abonados if ab.id not in asignados]

    def _mover_reasignar(self, mundo: ModeloOUPM) -> tuple:
        """
        Reasigna un registro a un abonado elegido uniformemente.
        Q(W'|W) = 1/n_reg * 1/n_ab  (simétrico, se cancela en M-H)
        """
        mundo_nuevo = mundo.clonar()
        reg    = self.rng.choice(mundo_nuevo.registros)
        ab_nuevo = self.rng.choice(mundo_nuevo.abonados)
        mundo_nuevo.source[reg.id] = ab_nuevo

        n_reg = len(mundo.registros)
        n_ab  = len(mundo.abonados)
        log_q = -math.log(n_reg) - math.log(n_ab)

        return mundo_nuevo, log_q, log_q, 'reasignar', [reg]

    def _mover_birth(self, mundo: ModeloOUPM) -> tuple:
        """
        Propone la creación de un nuevo objeto latente (abonado) en el mundo posible W.
        Para garantizar el balance detallado con el movimiento de muerte (death), 
        el nuevo abonado se inicializa sin registros asignados. La asignación de 
        registros a este nuevo abonado se delegará a movimientos posteriores de reasignación.
        """
        mundo_nuevo = mundo.clonar()
        r_base = self.rng.choice(mundo_nuevo.registros)

        # Se instancia el nuevo abonado utilizando los atributos de un registro base
        # como heurística inicial, pero el registro no se vincula a este abonado en este paso.
        ab_nuevo = mundo_nuevo._nuevo_abonado(
            nombre_real=r_base.nombre_obs,
            telefono_real=r_base.telefono_obs,
            ip_real=r_base.ip_obs,
            ciudad_real=r_base.ciudad_obs,
        )
        mundo_nuevo.abonados.append(ab_nuevo)

        # Cálculo de la probabilidad de transición directa (Q forward): 
        # Probabilidad de seleccionar el movimiento 'birth' y elegir el registro r_base.
        p_birth = 0.25
        n_reg   = len(mundo.registros)
        log_q_fwd = math.log(p_birth) - math.log(n_reg)

        # Cálculo de la probabilidad de transición inversa (Q backward):
        # Probabilidad de proponer un 'death' en el nuevo mundo W' que elimine este abonado.
        vacios_en_W_prima = self._abonados_vacios(mundo_nuevo)
        n_vacios_W_prima  = len(vacios_en_W_prima)
        
        if n_vacios_W_prima == 0:
            # Caso de seguridad: si no hay vacíos en W', la reversibilidad se rompe.
            log_q_bwd = -math.inf
        else:
            p_death = 0.15
            log_q_bwd = math.log(p_death) - math.log(n_vacios_W_prima)

        # Se retorna una lista vacía de registros afectados ya que la variable Source(r) no muta.
        return mundo_nuevo, log_q_fwd, log_q_bwd, 'birth', []

    def _mover_death(self, mundo: ModeloOUPM, vacios: list) -> tuple:
        """
        Death: elimina un abonado vacío elegido uniformemente.

        Q(death → W') = p_death * (1/n_vacios)
        Q(birth → W)  = p_birth * (1/n_reg)
          el birth inverso elegiría r_base = el registro que antes
          estaba en el abonado eliminado; pero ese abonado está vacío,
          así que cualquier registro podría ser r_base.
        """
        mundo_nuevo = mundo.clonar()
        ab_eliminar = self.rng.choice(vacios)
        ab_obj = next(a for a in mundo_nuevo.abonados
                      if a.id == ab_eliminar.id)
        mundo_nuevo.abonados.remove(ab_obj)

        p_death   = 0.15
        n_vacios  = len(vacios)
        log_q_fwd = math.log(p_death) - math.log(n_vacios)

        # Birth inverso: p_birth * 1/n_reg
        p_birth = 0.25
        n_reg   = len(mundo.registros)
        log_q_bwd = math.log(p_birth) - math.log(n_reg)

        return mundo_nuevo, log_q_fwd, log_q_bwd, 'death', []


# ──────────────────────────────────────────────
# Motor MCMC principal
# ──────────────────────────────────────────────

class MotorMCMC:

    def __init__(self, modelo: ModeloOUPM,
                 n_iter: int = 2000,
                 burn_in: int = 500,
                 thin: int = 5,
                 semilla: int = 42,
                 verbose: bool = True):
        self.modelo    = modelo
        self.n_iter    = n_iter
        self.burn_in   = burn_in
        self.thin      = thin
        self.rng       = random.Random(semilla)
        self.verbose   = verbose
        self.propuesta = PropuestaMCMC(self.rng)

        self.muestras: list  = []
        self.log_probs: list = []
        self.tasa_aceptacion_por_tipo: dict = {
            'reasignar': [0, 0],
            'birth':     [0, 0],
            'death':     [0, 0],
        }

    def correr(self) -> "Posterior":
        self.modelo.inicializar_aleatorio(self.rng)
        mundo_actual = self.modelo
        lp_actual    = mundo_actual.log_prob_mundo()

        if self.verbose:
            print(f"  Mundo inicial: {len(mundo_actual.abonados)} abonados, "
                  f"log P(W) = {lp_actual:.3f}")

        for it in range(self.n_iter):
            mundo_actual, lp_actual = self._paso_mh(mundo_actual, lp_actual)

            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                self.muestras.append(self._snapshot(mundo_actual))
                self.log_probs.append(lp_actual)

            if self.verbose and (it + 1) % 500 == 0:
                print(f"  Iter {it+1:4d}/{self.n_iter} | "
                      f"abonados={len(mundo_actual.abonados):3d} | "
                      f"log P={lp_actual:.3f} | "
                      f"muestras={len(self.muestras)}")

        if self.verbose:
            self._imprimir_estadisticas()

        return Posterior(self.muestras, self.log_probs, self.modelo.registros)

    def _paso_mh(self, mundo: ModeloOUPM, lp_mundo: float) -> tuple:
        try:
            mundo_prop, log_q_fwd, log_q_bwd, tipo, regs_afectados = \
                self.propuesta.proponer(mundo)
        except Exception:
            return mundo, lp_mundo

        # Birth/death cambian la estructura del mundo (k distinto):
        # usamos log_prob_mundo() completo para capturar el prior Poisson.
        # Reasignar: solo cambia Source(r) → blanket es suficiente.
        if tipo == 'reasignar':
            lp_blanket_prop = mundo_prop.log_prob_markov_blanket(regs_afectados)
            lp_blanket_act  = mundo.log_prob_markov_blanket(regs_afectados)
            lp_prop = lp_mundo + (lp_blanket_prop - lp_blanket_act)
        else:
            lp_prop = mundo_prop.log_prob_mundo()

        log_alpha = (lp_prop - lp_mundo) + (log_q_bwd - log_q_fwd)
        log_alpha = min(0.0, log_alpha)

        self.tasa_aceptacion_por_tipo[tipo][1] += 1
        if math.log(self.rng.random() + 1e-300) < log_alpha:
            self.tasa_aceptacion_por_tipo[tipo][0] += 1
            return mundo_prop, lp_prop

        return mundo, lp_mundo

    def _snapshot(self, mundo: ModeloOUPM) -> dict:
        return {
            'n_abonados': len(mundo.abonados),
            'source':     {rid: ab.id for rid, ab in mundo.source.items()},
            'abonados':   {ab.id: {
                'nombre':   ab.nombre_real,
                'telefono': ab.telefono_real,
                'ciudad':   ab.ciudad_real,
            } for ab in mundo.abonados},
        }

    def _imprimir_estadisticas(self):
        print("\n  Tasas de aceptación por tipo de movimiento:")
        for tipo, (acep, total) in self.tasa_aceptacion_por_tipo.items():
            tasa = acep / total if total > 0 else 0
            print(f"    {tipo:12s}: {acep:4d}/{total:4d} = {tasa:.1%}")
        print(f"  Muestras guardadas: {len(self.muestras)}")


# ──────────────────────────────────────────────
# Clase Posterior
# ──────────────────────────────────────────────

class Posterior:

    def __init__(self, muestras: list, log_probs: list, registros: list):
        self.muestras   = muestras
        self.log_probs  = log_probs
        self.registros  = registros
        self.n_muestras = len(muestras)

    def p_misma_identidad(self, id_reg_a: str, id_reg_b: str) -> float:
        if self.n_muestras == 0:
            return 0.0
        cuenta = sum(
            1 for m in self.muestras
            if m['source'].get(id_reg_a) == m['source'].get(id_reg_b)
        )
        return cuenta / self.n_muestras

    def p_n_abonados(self, k: int) -> float:
        if self.n_muestras == 0:
            return 0.0
        return sum(1 for m in self.muestras
                   if m['n_abonados'] == k) / self.n_muestras

    def distribucion_n_abonados(self) -> dict:
        dist = {}
        for m in self.muestras:
            k = m['n_abonados']
            dist[k] = dist.get(k, 0) + 1
        return {k: v / self.n_muestras for k, v in sorted(dist.items())}

    def cluster_map(self) -> dict:
        if not self.muestras:
            return {}
        idx_map     = max(range(self.n_muestras), key=lambda i: self.log_probs[i])
        muestra_map = self.muestras[idx_map]
        ids_vistos  = {}
        contador    = 0
        resultado   = {}
        for rid, ab_id in muestra_map['source'].items():
            if ab_id not in ids_vistos:
                ids_vistos[ab_id] = contador
                contador += 1
            resultado[rid] = ids_vistos[ab_id]
        return resultado

    def matriz_cocluster(self) -> dict:
        ids    = [r.id for r in self.registros]
        matriz = {}
        for i, ra in enumerate(ids):
            for rb in ids[i:]:
                p = self.p_misma_identidad(ra, rb)
                matriz[(ra, rb)] = p
                matriz[(rb, ra)] = p
        return matriz

    def resumen(self) -> str:
        if not self.muestras:
            return "Sin muestras."
        dist  = self.distribucion_n_abonados()
        k_map = max(dist, key=dist.get)
        lineas = [
            f"Muestras MCMC: {self.n_muestras}",
            f"Abonados más probable (MAP): k={k_map} (P={dist[k_map]:.1%})",
            "Distribución sobre #Abonados:",
        ]
        for k, p in dist.items():
            barra = '#' * int(p * 30)
            lineas.append(f"  k={k:3d}: {barra:<30s} {p:.1%}")
        return '\n'.join(lineas)
"""
main.py
Punto de entrada del sistema de resolución de identidades.
Ejecuta los tres escenarios de prueba y muestra resultados.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from datos.registro import GeneradorSintetico
from modelo.oupm import ModeloOUPM
from inferencia.mcmc import MotorMCMC


def evaluar_resultado(posterior, ground_truth: dict,
                      registros: list) -> dict:
    """
    Compara el resultado del sistema con el ground truth.
    Calcula:
      - Precisión de pares: % de pares (rA,rB) con identidad correcta
      - Error en k: |k_estimado - k_real|
    """
    ids = [r.id for r in registros]
    pares_correctos = 0
    pares_total = 0
    for i, ra in enumerate(ids):
        for rb in ids[i+1:]:
            pares_total += 1
            mismos_real = (ground_truth[ra] == ground_truth[rb])
            p_mismos    = posterior.p_misma_identidad(ra, rb)
            predicho    = p_mismos >= 0.5
            if predicho == mismos_real:
                pares_correctos += 1

    precision = pares_correctos / pares_total if pares_total > 0 else 0

    dist = posterior.distribucion_n_abonados()
    k_estimado = max(dist, key=dist.get)
    k_real = len(set(ground_truth.values()))

    return {
        'precision_pares': precision,
        'k_estimado': k_estimado,
        'k_real': k_real,
        'error_k': abs(k_estimado - k_real),
    }


def correr_escenario(nombre: str, tipo: str,
                     n_iter: int = 2000, burn_in: int = 500):
    print(f"\n{'='*60}")
    print(f"  ESCENARIO {nombre}")
    print(f"{'='*60}")

    gen = GeneradorSintetico(semilla=7)
    registros, ground_truth, abonados_reales = gen.generar_escenario(tipo)

    print(f"  Registros observados : {len(registros)}")
    print(f"  Abonados reales (GT) : {len(set(ground_truth.values()))}")
    print(f"  Corriendo MCMC ({n_iter} iteraciones)...")

    modelo = ModeloOUPM(registros)
    motor  = MotorMCMC(modelo, n_iter=n_iter, burn_in=burn_in,
                       thin=5, semilla=42, verbose=True)

    t0 = time.time()
    posterior = motor.correr()
    t1 = time.time()

    print(f"\n  Tiempo de ejecución: {t1-t0:.2f}s")
    print(f"\n  {posterior.resumen()}")

    metricas = evaluar_resultado(posterior, ground_truth, registros)
    print(f"\n  Métricas de evaluación:")
    print(f"    Precisión de pares  : {metricas['precision_pares']:.1%}")
    print(f"    k real              : {metricas['k_real']}")
    print(f"    k estimado (MAP)    : {metricas['k_estimado']}")
    print(f"    Error en k          : {metricas['error_k']}")

    print(f"\n  Ejemplos de consultas de identidad:")
    ids = [r.id for r in registros[:6]]
    for i in range(min(3, len(ids))):
        for j in range(i+1, min(4, len(ids))):
            ra, rb = ids[i], ids[j]
            p = posterior.p_misma_identidad(ra, rb)
            mismos = ground_truth[ra] == ground_truth[rb]
            marca = "OK" if (p >= 0.5) == mismos else "FALLO"
            print(f"    P({ra}={rb}) = {p:.2f}  [real: {'mismos' if mismos else 'distintos'}] {marca}")

    return posterior, metricas, t1 - t0


if __name__ == "__main__":
    print("\nSistema Probabilístico de Resolución de Identidades")
    print("Programación Probabilística — Cap. 18 AIMA 4ed.\n")

    resultados = {}

    p_a, m_a, t_a = correr_escenario("A — Bajo ruido",   'A',
                                      n_iter=2000, burn_in=400)
    p_b, m_b, t_b = correr_escenario("B — Alto ruido",   'B',
                                      n_iter=3000, burn_in=600)
    p_c, m_c, t_c = correr_escenario("C — Escalabilidad",'C',
                                      n_iter=4000, burn_in=800)

    print(f"\n{'='*60}")
    print("  RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    print(f"  {'Escenario':<20} {'Precisión':>10} {'Error k':>8} {'Tiempo':>8}")
    print(f"  {'-'*50}")
    for nombre, m, t in [("A - bajo ruido", m_a, t_a),
                          ("B - alto ruido", m_b, t_b),
                          ("C - escala",     m_c, t_c)]:
        print(f"  {nombre:<20} {m['precision_pares']:>9.1%} "
              f"{m['error_k']:>8d} {t:>7.1f}s")

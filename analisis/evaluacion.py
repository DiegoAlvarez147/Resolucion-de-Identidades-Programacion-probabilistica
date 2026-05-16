"""
analisis/evaluacion.py
Módulo de evaluación del sistema MCMC.

Funciones públicas:
  evaluar_resultado(posterior, ground_truth, registros) → dict
  imprimir_metricas(metricas)                           → None
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def evaluar_resultado(posterior, ground_truth: dict, registros: list) -> dict:
    """
    Compara el resultado del MCMC con el ground truth conocido.

    Retorna un dict con:
      precision_pares : fracción de pares (rA,rB) con identidad predicha correcta
      k_estimado      : número de abonados MAP estimado por el MCMC
      k_real          : número de abonados reales (del generador)
      error_k         : |k_estimado - k_real|
      f1_pares        : F1 sobre la tarea de decidir si dos registros son del mismo
                        abonado (threshold 0.5 en P_misma_identidad)
    """
    ids = [r.id for r in registros]

    # ── Precisión de pares ────────────────────────────────────────────────
    tp = fp = tn = fn = 0
    for i, ra in enumerate(ids):
        for rb in ids[i+1:]:
            mismos_real = (ground_truth[ra] == ground_truth[rb])
            p_mismos    = posterior.p_misma_identidad(ra, rb)
            predicho    = p_mismos >= 0.5
            if predicho and mismos_real:
                tp += 1
            elif predicho and not mismos_real:
                fp += 1
            elif not predicho and not mismos_real:
                tn += 1
            else:
                fn += 1

    total = tp + fp + tn + fn
    precision_pares = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ── Error en k ───────────────────────────────────────────────────────
    dist       = posterior.distribucion_n_abonados()
    k_estimado = max(dist, key=dist.get) if dist else 0
    k_real     = len(set(ground_truth.values()))

    return {
        'precision_pares': precision_pares,
        'f1_pares':        f1,
        'precision':       prec,
        'recall':          rec,
        'k_estimado':      k_estimado,
        'k_real':          k_real,
        'error_k':         abs(k_estimado - k_real),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def imprimir_metricas(metricas: dict, prefijo: str = "  ") -> None:
    """Imprime las métricas de evaluación en formato legible."""
    print(f"{prefijo}Métricas de evaluación:")
    print(f"{prefijo}  Precisión de pares   : {metricas['precision_pares']:.1%}")
    print(f"{prefijo}  F1 (misma identidad) : {metricas['f1_pares']:.1%}")
    print(f"{prefijo}  Precisión            : {metricas['precision']:.1%}")
    print(f"{prefijo}  Recall               : {metricas['recall']:.1%}")
    print(f"{prefijo}  k real               : {metricas['k_real']}")
    print(f"{prefijo}  k estimado (MAP)     : {metricas['k_estimado']}")
    print(f"{prefijo}  Error en k           : {metricas['error_k']}")
    print(f"{prefijo}  TP/FP/TN/FN          : "
          f"{metricas['tp']}/{metricas['fp']}/{metricas['tn']}/{metricas['fn']}")

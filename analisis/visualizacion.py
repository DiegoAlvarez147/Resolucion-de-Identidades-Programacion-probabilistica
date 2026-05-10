"""
analisis/visualizacion.py
Visualización completa del sistema MCMC con matplotlib.

Genera 4 figuras al ejecutar:
  1. Animación de convergencia MCMC (los 3 escenarios en tiempo real)
  2. Distribuciones posteriores P(k | evidencia)
  3. Mundos posibles — registros coloreados por abonado asignado
  4. Diagnóstico: tasas de aceptación y fórmula M-H

Ejecutar desde la raíz del proyecto:
  python analisis/visualizacion.py
"""

import sys
import os
import time
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datos.registro import GeneradorSintetico
from modelo.oupm import ModeloOUPM
from inferencia.mcmc import MotorMCMC

# ──────────────────────────────────────────────
# Paleta de colores
# ──────────────────────────────────────────────
COLOR_A    = '#1D9E75'   # verde  — escenario A
COLOR_B    = '#E6A817'   # naranja — escenario B
COLOR_C    = '#993C1D'   # rojo   — escenario C
COLOR_REAL = '#7F77DD'   # violeta — valor real (ground truth)
COLORES_AB = ['#7F77DD','#1D9E75','#993C1D','#E6A817',
              '#3B82F6','#EC4899','#8B5CF6','#06B6D4']


# ──────────────────────────────────────────────
# Paso 1: Recolectar datos de los 3 escenarios
# ──────────────────────────────────────────────

def correr_escenario(tipo, n_iter, burn_in):
    """Corre el MCMC y retorna todos los datos necesarios para graficar."""
    print(f"  Corriendo escenario {tipo}...", end=' ', flush=True)
    t0 = time.time()

    gen = GeneradorSintetico(semilla=7)
    registros, ground_truth, abonados_reales = gen.generar_escenario(tipo)

    modelo = ModeloOUPM(registros)
    motor  = MotorMCMC(modelo, n_iter=n_iter, burn_in=burn_in,
                       thin=5, semilla=42, verbose=False)
    posterior = motor.correr()

    t1 = time.time()
    print(f"listo en {t1-t0:.1f}s")

    return {
        'tipo':        tipo,
        'registros':   registros,
        'ground_truth': ground_truth,
        'k_real':      len(set(ground_truth.values())),
        'n_reg':       len(registros),
        'muestras':    motor.muestras,
        'log_probs':   motor.log_probs,
        'posterior':   posterior,
        'evolucion':   [m['n_abonados'] for m in motor.muestras],
        'dist':        posterior.distribucion_n_abonados(),
        'tasas':       motor.tasa_aceptacion_por_tipo,
        'tiempo':      t1 - t0,
    }


# ──────────────────────────────────────────────
# Figura 1: Convergencia MCMC animada
# ──────────────────────────────────────────────

def figura_convergencia(datos_abc):
    """
    Muestra la evolución de k abonados a lo largo de las muestras MCMC.
    La línea punteada verde es el valor real (ground truth).
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=False)
    fig.suptitle('Figura 1 — Convergencia MCMC\n'
                 'Evolución de k (número de abonados estimados) por muestra',
                 fontsize=13, fontweight='bold', y=1.01)

    configs = [
        (datos_abc['A'], COLOR_A, 'Escenario A — Bajo ruido (12 registros, k real = 5)'),
        (datos_abc['B'], COLOR_B, 'Escenario B — Alto ruido (36 registros, k real = 8)'),
        (datos_abc['C'], COLOR_C, 'Escenario C — Escalabilidad (102 registros, k real = 20)'),
    ]

    for ax, (d, color, titulo) in zip(axes, configs):
        evol   = d['evolucion']
        k_real = d['k_real']
        xs     = list(range(len(evol)))

        # Curva de evolución
        ax.plot(xs, evol, color=color, linewidth=1.4,
                alpha=0.85, label='k estimado por MCMC')

        # Línea del valor real
        ax.axhline(k_real, color=COLOR_REAL, linewidth=1.5,
                   linestyle='--', label=f'k real = {k_real}')

        # Zona de aceptación (±2 del valor real)
        ax.axhspan(k_real - 2, k_real + 2, alpha=0.08,
                   color=COLOR_REAL, label='Zona ±2 del real')

        # Nota de convergencia
        k_map = max(d['dist'], key=d['dist'].get)
        nota  = f'MAP: k={k_map}  |  Real: k={k_real}  |  Error: {abs(k_map - k_real)}'
        ax.set_title(f'{titulo}\n{nota}', fontsize=10, loc='left', pad=4)
        ax.set_ylabel('k abonados', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

        # Anotación especial para escenario C
        if d['tipo'] == 'C':
            ax.annotate('⚠ Mezcla lenta:\nMCMC atascado en k=4\n(cap. 18.2.2 del libro)',
                        xy=(len(evol)//2, 4),
                        xytext=(len(evol)//2, k_real - 4),
                        fontsize=8, color=COLOR_C,
                        arrowprops=dict(arrowstyle='->', color=COLOR_C, lw=1),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAECE7',
                                  edgecolor=COLOR_C, alpha=0.9))

    axes[-1].set_xlabel('Número de muestra MCMC (post burn-in)', fontsize=9)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Figura 2: Distribuciones posteriores
# ──────────────────────────────────────────────

def figura_distribuciones(datos_abc):
    """
    Histograma de P(k | evidencia) para los 3 escenarios.
    Muestra la incertidumbre del modelo sobre el número de abonados reales.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle('Figura 2 — Distribución posterior P(k | evidencia)\n'
                 'Probabilidad de cada número de abonados reales dado lo observado',
                 fontsize=13, fontweight='bold')

    configs = [
        (datos_abc['A'], COLOR_A, 'A — Bajo ruido'),
        (datos_abc['B'], COLOR_B, 'B — Alto ruido'),
        (datos_abc['C'], COLOR_C, 'C — Escalabilidad'),
    ]

    for ax, (d, color, titulo) in zip(axes, configs):
        dist   = d['dist']
        k_real = d['k_real']
        ks     = sorted(dist.keys())
        probs  = [dist[k] * 100 for k in ks]

        bars = ax.bar(ks, probs, color=color, alpha=0.7,
                      edgecolor='white', linewidth=0.5, zorder=3)

        # Resaltar k real
        for k, bar in zip(ks, bars):
            if k == k_real:
                bar.set_edgecolor(COLOR_REAL)
                bar.set_linewidth(2.5)
                bar.set_alpha(1.0)

        # Línea k real
        ax.axvline(k_real, color=COLOR_REAL, linewidth=2,
                   linestyle='--', label=f'k real = {k_real}', zorder=4)

        # Línea k MAP
        k_map = max(dist, key=dist.get)
        if k_map != k_real:
            ax.axvline(k_map, color=color, linewidth=1.5,
                       linestyle=':', alpha=0.8,
                       label=f'k MAP = {k_map}', zorder=4)

        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_xlabel('k (número de abonados)', fontsize=9)
        ax.set_ylabel('Probabilidad (%)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

        # Estadísticas en el gráfico
        media = sum(k * dist[k] for k in ks)
        textstr = (f'Media: {media:.1f}\n'
                   f'MAP: {k_map}\n'
                   f'Real: {k_real}\n'
                   f'Error: {abs(k_map-k_real)}')
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Figura 3: Mundos posibles (visualización de Source)
# ──────────────────────────────────────────────

def figura_mundos_posibles(datos_abc):
    """
    Muestra 3 mundos posibles del escenario A:
      - Estado inicial del MCMC
      - Una muestra intermedia
      - La muestra MAP (más probable)
    Cada punto es un registro, el color indica su abonado asignado (Source).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle('Figura 3 — Mundos posibles del MCMC (Escenario A)\n'
                 'Cada punto = un registro. Color = abonado asignado (variable Source)',
                 fontsize=13, fontweight='bold')

    d           = datos_abc['A']
    registros   = d['registros']
    gt          = d['ground_truth']
    muestras    = d['muestras']
    n           = len(registros)

    # Posiciones fijas para los registros (reproducibles)
    rng_pos = random.Random(99)
    posiciones = [(rng_pos.uniform(0.05, 0.95),
                   rng_pos.uniform(0.05, 0.95)) for _ in range(n)]

    titulos = ['Mundo inicial (iter 1)',
               f'Mundo intermedio (iter {len(muestras)//2})',
               'Mundo MAP (más probable)']

    idx_muestras = [0, len(muestras)//2,
                    max(range(len(d['log_probs'])),
                        key=lambda i: d['log_probs'][i])]

    for ax, titulo, idx in zip(axes, titulos, idx_muestras):
        muestra = muestras[min(idx, len(muestras)-1)]
        source  = muestra['source']  # {id_reg: id_abonado}

        # Mapear id_abonado → color consecutivo
        ab_ids_vistos = {}
        contador_color = 0
        colores_puntos = []
        grupos = {}

        for reg in registros:
            ab_id = source.get(reg.id, 0)
            if ab_id not in ab_ids_vistos:
                ab_ids_vistos[ab_id] = contador_color
                contador_color += 1
            c_idx = ab_ids_vistos[ab_id]
            color = COLORES_AB[c_idx % len(COLORES_AB)]
            colores_puntos.append(color)
            if c_idx not in grupos:
                grupos[c_idx] = []
            grupos[c_idx].append(reg.id)

        # Dibujar líneas al centroide del grupo
        for c_idx, reg_ids in grupos.items():
            idxs  = [i for i, r in enumerate(registros) if r.id in reg_ids]
            cx    = sum(posiciones[i][0] for i in idxs) / len(idxs)
            cy    = sum(posiciones[i][1] for i in idxs) / len(idxs)
            color = COLORES_AB[c_idx % len(COLORES_AB)]
            for i in idxs:
                ax.plot([posiciones[i][0], cx], [posiciones[i][1], cy],
                        color=color, alpha=0.15, linewidth=0.8, zorder=1)
            # Centroide (abonado hipotético)
            ax.scatter(cx, cy, s=180, color=color, marker='*',
                       zorder=4, edgecolors='white', linewidths=0.8)

        # Dibujar registros
        xs = [p[0] for p in posiciones]
        ys = [p[1] for p in posiciones]
        ax.scatter(xs, ys, s=60, c=colores_puntos,
                   zorder=3, edgecolors='white', linewidths=0.6)

        # Etiquetas de registros
        for i, reg in enumerate(registros):
            ax.annotate(reg.id, (posiciones[i][0], posiciones[i][1]),
                        fontsize=5.5, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points', color='#555')

        n_ab  = muestra['n_abonados']
        lp    = d['log_probs'][min(idx, len(d['log_probs'])-1)]
        ax.set_title(f'{titulo}\nk={n_ab} abonados  |  log P(W) = {lp:.2f}',
                     fontsize=9, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axis('off')

        # Leyenda de grupos
        parches = [mpatches.Patch(color=COLORES_AB[c % len(COLORES_AB)],
                                  label=f'Abonado {c+1} ({len(g)} regs)')
                   for c, g in grupos.items()]
        ax.legend(handles=parches, fontsize=6.5, loc='lower right',
                  framealpha=0.85, ncol=2)

    # Anotación de ground truth a la derecha
    fig.text(0.99, 0.5,
             f'Ground truth:\n{d["k_real"]} abonados\nreales',
             va='center', ha='right', fontsize=8,
             color=COLOR_REAL, fontweight='bold')

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Figura 4: Diagnóstico — tasas y teoría M-H
# ──────────────────────────────────────────────

def figura_diagnostico(datos_abc):
    """
    Panel de diagnóstico con:
      - Tasas de aceptación por tipo de movimiento y escenario
      - Evolución de log P(W) a lo largo del MCMC
      - Tabla resumen con métricas finales
    """
    fig = plt.figure(figsize=(13, 6))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    fig.suptitle('Figura 4 — Diagnóstico del motor MCMC\n'
                 'Tasas de aceptación, log-probabilidad y métricas finales',
                 fontsize=13, fontweight='bold')

    configs = [
        (datos_abc['A'], COLOR_A, 'A — Bajo ruido'),
        (datos_abc['B'], COLOR_B, 'B — Alto ruido'),
        (datos_abc['C'], COLOR_C, 'C — Escalabilidad'),
    ]

    # ── Fila 0: Tasas de aceptación por tipo ──────────
    for col, (d, color, titulo) in enumerate(configs):
        ax = fig.add_subplot(gs[0, col])
        tas = d['tasas']
        tipos = ['reasignar', 'birth', 'death']
        tasas_pct = []
        for t in tipos:
            a, total = tas[t]
            tasas_pct.append((a / total * 100) if total > 0 else 0)

        bars = ax.bar(tipos, tasas_pct, color=[color, color+'99', color+'55'],
                      edgecolor='white', linewidth=0.5)

        # Etiqueta en cada barra
        for bar, pct in zip(bars, tasas_pct):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{pct:.0f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # Línea de referencia 30%
        ax.axhline(30, color='gray', linewidth=0.8,
                   linestyle='--', alpha=0.5, label='30% referencia')

        ax.set_title(titulo, fontsize=9, fontweight='bold')
        ax.set_ylabel('Tasa aceptación (%)', fontsize=8)
        ax.set_ylim(0, 110)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis='y', alpha=0.2)

        # Nota de mezcla
        birth_pct = tasas_pct[1]
        if birth_pct == 0:
            ax.text(1, 50, 'MEZCLA\nLENTA\n(cap.18.2.2)',
                    ha='center', va='center', fontsize=8,
                    color=COLOR_C, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#FAECE7',
                              edgecolor=COLOR_C, alpha=0.9))

    # ── Fila 1: Log-probabilidad del mundo ─────────────
    for col, (d, color, titulo) in enumerate(configs):
        ax = fig.add_subplot(gs[1, col])
        lps = d['log_probs']
        ax.plot(lps, color=color, linewidth=1.2, alpha=0.8)

        # Media móvil suavizada
        ventana = max(1, len(lps)//15)
        mm = [sum(lps[max(0,i-ventana):i+1]) /
              len(lps[max(0,i-ventana):i+1]) for i in range(len(lps))]
        ax.plot(mm, color='black', linewidth=1.5,
                alpha=0.5, linestyle='-', label='Media móvil')

        ax.set_title(f'log P(W) — {titulo}', fontsize=9)
        ax.set_xlabel('Muestra MCMC', fontsize=8)
        ax.set_ylabel('log P(W)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)

        # Anotación de estabilidad
        if len(lps) > 10:
            std_final = (sum((x - mm[-1])**2
                            for x in lps[-len(lps)//4:]) /
                         (len(lps)//4)) ** 0.5
            ax.text(0.97, 0.05,
                    f'σ final: {std_final:.2f}',
                    transform=ax.transAxes,
                    fontsize=7, ha='right', color='gray')

    return fig


# ──────────────────────────────────────────────
# Figura 5: Tabla resumen interactiva
# ──────────────────────────────────────────────

def figura_resumen(datos_abc):
    """
    Tabla comparativa de los 3 escenarios con métricas clave.
    """
    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.suptitle('Figura 5 — Resumen comparativo de escenarios',
                 fontsize=13, fontweight='bold')
    ax.axis('off')

    columnas = ['Escenario', 'Registros', 'k real', 'k MAP',
                'Error k', 'Birth %', 'Tiempo', 'Estado']
    filas = []
    colores_filas = []

    for esc, color in [('A', COLOR_A), ('B', COLOR_B), ('C', COLOR_C)]:
        d  = datos_abc[esc]
        t  = d['tasas']
        bp = t['birth']
        birth_pct = f"{bp[0]/bp[1]*100:.0f}%" if bp[1] > 0 else "0%"
        k_map = max(d['dist'], key=d['dist'].get)
        error = abs(k_map - d['k_real'])

        if esc == 'A':   estado = '[OK]  Mezcla bien'
        elif esc == 'B': estado = '[~]   Mezcla parcial'
        else:            estado = '[!]   Sin mezcla'

        filas.append([
            f"Esc. {esc}",
            str(d['n_reg']),
            str(d['k_real']),
            str(k_map),
            str(error),
            birth_pct,
            f"{d['tiempo']:.1f}s",
            estado,
        ])
        colores_filas.append([color + '22'] * len(columnas))

    tabla = ax.table(
        cellText=filas,
        colLabels=columnas,
        cellLoc='center',
        loc='center',
        cellColours=colores_filas,
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 2.2)

    # Encabezado
    for j in range(len(columnas)):
        tabla[0, j].set_facecolor('#7F77DD')
        tabla[0, j].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  Visualización del Sistema MCMC de Resolución de Identidades")
    print("  Cap. 18 — Programación Probabilística (AIMA 4ed.)")
    print("="*55)
    print("\nPaso 1: Corriendo los 3 escenarios...")

    datos_abc = {}
    for esc, n_iter, burn in [('A', 2000, 400),
                               ('B', 3000, 600),
                               ('C', 4000, 800)]:
        datos_abc[esc] = correr_escenario(esc, n_iter, burn)
        datos_abc[esc]['tipo'] = esc

    print("\nPaso 2: Generando figuras...\n")

    # Generar todas las figuras
    fig1 = figura_convergencia(datos_abc)
    print("  ✓ Figura 1: Convergencia MCMC")

    fig2 = figura_distribuciones(datos_abc)
    print("  ✓ Figura 2: Distribuciones posteriores")

    fig3 = figura_mundos_posibles(datos_abc)
    print("  ✓ Figura 3: Mundos posibles")

    fig4 = figura_diagnostico(datos_abc)
    print("  ✓ Figura 4: Diagnóstico MCMC")

    fig5 = figura_resumen(datos_abc)
    print("  ✓ Figura 5: Resumen comparativo")

    # Guardar todas como PNG
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'resultados')
    os.makedirs(output_dir, exist_ok=True)

    for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5], 1):
        path = os.path.join(output_dir, f'figura_{i}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Guardada: resultados/figura_{i}.png")

    print("\nPaso 3: Mostrando figuras en pantalla...")
    print("  (Cierra cada ventana para ver la siguiente)\n")
    plt.show()
    print("\n¡Listo! Revisa la carpeta resultados/ para los PNG.")


if __name__ == '__main__':
    main()

"""
simulacion_vivo.py
Simulación EN VIVO del motor MCMC con pygame.

Controles:
  ESPACIO       → pausar / reanudar
  A / B / C     → cambiar escenario
  SLIDER        → arrastrar con el mouse para cambiar velocidad (1x–20x)
  R             → reiniciar
  Q / ESC       → salir

Ejecutar desde la raíz del proyecto:
  python simulacion_vivo.py
"""

import sys
import os
import math
import random
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datos.registro import GeneradorSintetico
from modelo.oupm import ModeloOUPM, Abonado
from inferencia.mcmc import PropuestaMCMC

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
W, H      = 1150, 700
FPS       = 60
PANEL_W   = 310
SIM_W     = W - PANEL_W

BG        = (15,  15,  22)
BG_PANEL  = (24,  24,  36)
GRID_C    = (32,  32,  48)
WHITE     = (238, 238, 245)
GRAY      = (120, 120, 145)
GRAY2     = (60,  60,  82)

COLORES_AB = [
    (127, 119, 221),
    ( 29, 158, 117),
    (200,  80,  50),
    (220, 160,  20),
    ( 59, 130, 246),
    (220,  80, 160),
    (139,  92, 246),
    (  6, 182, 212),
    (240, 130,  60),
    (100, 200, 150),
]

COLOR_GT    = ( 29, 158, 117)
COLOR_ACENT = (127, 119, 221)
COLOR_OK    = ( 70, 200, 120)
COLOR_WARN  = (220, 120,  40)
COLOR_BAD   = (200,  60,  60)


# ──────────────────────────────────────────────
# Slider interactivo
# ──────────────────────────────────────────────

class Slider:
    """
    Barra deslizante de velocidad en el panel lateral.
    Se arrastra con el mouse en tiempo real.
    """

    def __init__(self, x, y, w, h, val_min, val_max, valor_inicial):
        self.rect    = pygame.Rect(x, y, w, h)
        self.val_min = val_min
        self.val_max = val_max
        self.valor   = valor_inicial
        self.activo  = False

    def _handle_x(self):
        ratio = (self.valor - self.val_min) / (self.val_max - self.val_min)
        return self.rect.x + int(ratio * self.rect.w)

    def handle_rect(self):
        cx = self._handle_x()
        cy = self.rect.centery
        return pygame.Rect(cx - 9, cy - 9, 18, 18)

    def on_event(self, event, offset_x=0, offset_y=0):
        """Procesa eventos del mouse. offset_x/y = posición del panel en pantalla."""
        mx = event.pos[0] - offset_x
        my = event.pos[1] - offset_y

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect().collidepoint(mx, my) or \
               self.rect.collidepoint(mx, my):
                self.activo = True
                self._actualizar(mx)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.activo = False

        elif event.type == pygame.MOUSEMOTION and self.activo:
            self._actualizar(mx)

    def _actualizar(self, mx):
        ratio = (mx - self.rect.x) / self.rect.w
        ratio = max(0.0, min(1.0, ratio))
        self.valor = int(self.val_min + ratio * (self.val_max - self.val_min))
        self.valor = max(self.val_min, min(self.val_max, self.valor))

    def dibujar(self, surface, fuente):
        cy = self.rect.centery

        # Pista (track) de fondo
        pygame.draw.rect(surface, GRAY2,
                         (self.rect.x, cy - 3, self.rect.w, 6),
                         border_radius=3)

        # Relleno hasta el handle
        cx = self._handle_x()
        if cx > self.rect.x:
            pygame.draw.rect(surface, COLOR_ACENT,
                             (self.rect.x, cy - 3, cx - self.rect.x, 6),
                             border_radius=3)

        # Handle (círculo arrastrable)
        color_h = WHITE if self.activo else (180, 178, 240)
        pygame.draw.circle(surface, color_h, (cx, cy), 9)
        pygame.draw.circle(surface, COLOR_ACENT, (cx, cy), 9, 2)

        # Valor actual a la derecha
        lbl = fuente.render(f"{self.valor}x", True, WHITE)
        surface.blit(lbl, (self.rect.right + 8,
                           cy - lbl.get_height() // 2))

        # Min / Max debajo de la barra
        mn     = fuente.render(str(self.val_min), True, GRAY)
        mx_lbl = fuente.render(str(self.val_max), True, GRAY)
        surface.blit(mn,    (self.rect.x, self.rect.bottom + 2))
        surface.blit(mx_lbl,(self.rect.right - mx_lbl.get_width(),
                              self.rect.bottom + 2))


# ──────────────────────────────────────────────
# Estado visual del mundo MCMC
# ──────────────────────────────────────────────

class EstadoVisual:
    def __init__(self, registros, ground_truth):
        self.registros    = registros
        self.ground_truth = ground_truth
        self.n            = len(registros)
        self.pos_reg      = self._distribuir()
        self.pos_ab       = {}
        self.vel_ab       = {}
        self.source       = {}
        self.n_abonados   = 0
        self.ab_ids       = []
        self.flash_reg    = None
        self.flash_timer  = 0
        self.color_map    = {}
        self._cidx        = 0

    def _distribuir(self):
        """Posiciones fijas de registros en espiral con ruido."""
        rng = random.Random(42)
        cx, cy = SIM_W // 2, H // 2
        radio  = min(SIM_W, H) * 0.36
        pos    = {}
        for i, reg in enumerate(self.registros):
            ang = (2 * math.pi * i / self.n) + rng.uniform(-0.18, 0.18)
            r   = radio * (0.45 + 0.55 * rng.random())
            pos[reg.id] = [cx + r * math.cos(ang),
                           cy + r * math.sin(ang)]
        return pos

    def actualizar_mundo(self, muestra, reg_afectado):
        self.source     = muestra['source']
        self.ab_ids     = list(muestra['abonados'].keys())
        self.n_abonados = muestra['n_abonados']

        # Asignar colores persistentes a nuevos abonados
        for ab_id in self.ab_ids:
            if ab_id not in self.color_map:
                self.color_map[ab_id] = self._cidx % len(COLORES_AB)
                self._cidx += 1

        # Calcular centroides objetivo
        centroides = {}
        for ab_id in self.ab_ids:
            grupo = [r for r in self.registros
                     if self.source.get(r.id) == ab_id]
            if not grupo:
                continue
            centroides[ab_id] = (
                sum(self.pos_reg[r.id][0] for r in grupo) / len(grupo),
                sum(self.pos_reg[r.id][1] for r in grupo) / len(grupo),
            )

        # Física suave: mover abonados hacia centroides con muelle
        for ab_id, (tx, ty) in centroides.items():
            if ab_id not in self.pos_ab:
                self.pos_ab[ab_id] = [tx + random.uniform(-15, 15),
                                      ty + random.uniform(-15, 15)]
                self.vel_ab[ab_id] = [0.0, 0.0]
            dx = tx - self.pos_ab[ab_id][0]
            dy = ty - self.pos_ab[ab_id][1]
            self.vel_ab[ab_id][0] = self.vel_ab[ab_id][0] * 0.72 + dx * 0.14
            self.vel_ab[ab_id][1] = self.vel_ab[ab_id][1] * 0.72 + dy * 0.14
            self.pos_ab[ab_id][0] += self.vel_ab[ab_id][0]
            self.pos_ab[ab_id][1] += self.vel_ab[ab_id][1]

        # Eliminar abonados que ya no existen en este mundo
        for ab_id in list(self.pos_ab):
            if ab_id not in self.ab_ids:
                del self.pos_ab[ab_id]
                del self.vel_ab[ab_id]

        # Flash del registro afectado
        if reg_afectado:
            self.flash_reg   = reg_afectado
            self.flash_timer = 20

    def tick(self):
        if self.flash_timer > 0:
            self.flash_timer -= 1


# ──────────────────────────────────────────────
# Dibujo del área de simulación
# ──────────────────────────────────────────────

def dibujar_sim(surface, estado, fuente_sm, fuente_md):
    surface.fill(BG)

    # Grid de fondo
    for x in range(0, SIM_W, 40):
        pygame.draw.line(surface, GRID_C, (x, 0), (x, H))
    for y in range(0, H, 40):
        pygame.draw.line(surface, GRID_C, (0, y), (SIM_W, y))

    if not estado.source:
        msg = fuente_md.render("Iniciando MCMC...", True, GRAY)
        surface.blit(msg, (SIM_W // 2 - msg.get_width() // 2, H // 2))
        return

    # Líneas registro → abonado (con transparencia)
    alpha_surf = pygame.Surface((SIM_W, H), pygame.SRCALPHA)
    for reg in estado.registros:
        ab_id = estado.source.get(reg.id)
        if ab_id is None or ab_id not in estado.pos_ab:
            continue
        px, py = estado.pos_reg[reg.id]
        ax, ay = estado.pos_ab[ab_id]
        c_idx  = estado.color_map.get(ab_id, 0)
        color  = COLORES_AB[c_idx % len(COLORES_AB)]
        pygame.draw.line(alpha_surf, (*color, 55),
                         (int(px), int(py)), (int(ax), int(ay)), 1)
    surface.blit(alpha_surf, (0, 0))

    # Abonados hipotéticos (círculos grandes con halo)
    for ab_id in estado.ab_ids:
        if ab_id not in estado.pos_ab:
            continue
        ax, ay  = estado.pos_ab[ab_id]
        c_idx   = estado.color_map.get(ab_id, 0)
        color   = COLORES_AB[c_idx % len(COLORES_AB)]

        # Halo translúcido
        halo = pygame.Surface((80, 80), pygame.SRCALPHA)
        pygame.draw.circle(halo, (*color, 28), (40, 40), 40)
        surface.blit(halo, (int(ax) - 40, int(ay) - 40))

        # Círculo principal
        pygame.draw.circle(surface, color, (int(ax), int(ay)), 20)
        pygame.draw.circle(surface, BG,    (int(ax), int(ay)), 20, 2)

        # Etiqueta con número de registros asignados
        n_regs = sum(1 for r in estado.registros
                     if estado.source.get(r.id) == ab_id)
        lbl = fuente_sm.render(f"A{ab_id % 100}({n_regs})", True, WHITE)
        surface.blit(lbl, (int(ax) - lbl.get_width() // 2,
                           int(ay) - lbl.get_height() // 2))

    # Registros observables (círculos pequeños)
    for reg in estado.registros:
        px, py   = estado.pos_reg[reg.id]
        ab_id    = estado.source.get(reg.id)
        c_idx    = estado.color_map.get(ab_id, 0) if ab_id else 0
        color    = COLORES_AB[c_idx % len(COLORES_AB)]
        es_flash = (reg.id == estado.flash_reg and estado.flash_timer > 0)
        radio    = 13 if es_flash else 9

        pygame.draw.circle(surface, color, (int(px), int(py)), radio)
        if es_flash:
            pygame.draw.circle(surface, WHITE, (int(px), int(py)), radio, 2)

        # Contorno verde = ground truth
        pygame.draw.circle(surface, COLOR_GT, (int(px), int(py)), radio + 3, 1)

        # Etiqueta del registro
        lbl = fuente_sm.render(reg.id, True, WHITE)
        surface.blit(lbl, (int(px) - lbl.get_width() // 2,
                           int(py) + radio + 3))


# ──────────────────────────────────────────────
# Dibujo del panel lateral
# ──────────────────────────────────────────────

def dibujar_panel(surface, estado, slider, tasas, ultimo_mov,
                  fuente_sm, fuente_md, fuente_lg,
                  escenario, pausa, iteracion, aceptados, rechazados):

    panel = pygame.Surface((PANEL_W, H))
    panel.fill(BG_PANEL)

    y = 14

    def txt(texto, f, color, cx=False, x=16):
        nonlocal y
        s  = f.render(texto, True, color)
        px = PANEL_W // 2 - s.get_width() // 2 if cx else x
        panel.blit(s, (px, y))
        y += s.get_height() + 4

    def sep():
        nonlocal y
        pygame.draw.line(panel, GRID_C, (12, y), (PANEL_W - 12, y), 1)
        y += 8

    # Encabezado
    txt("MCMC EN VIVO", fuente_md, COLOR_ACENT, cx=True)
    txt("Resolución de identidades", fuente_sm, GRAY, cx=True)
    sep()

    # Escenario y estado
    labels = {'A': 'A — Bajo ruido  (k=5)',
              'B': 'B — Alto ruido  (k=8)',
              'C': 'C — Escalabilidad (k=20)'}
    txt(labels.get(escenario, ''), fuente_sm, WHITE)

    estado_txt = "  PAUSADO  " if pausa else " CORRIENDO "
    color_est  = COLOR_WARN if pausa else COLOR_OK
    s  = fuente_sm.render(estado_txt, True, BG_PANEL)
    sw = s.get_width() + 12
    pygame.draw.rect(panel, color_est,
                     (PANEL_W // 2 - sw // 2, y, sw, s.get_height() + 4),
                     border_radius=4)
    panel.blit(s, (PANEL_W // 2 - s.get_width() // 2, y + 2))
    y += s.get_height() + 10
    sep()

    # Iteración y k
    txt("ITERACIÓN", fuente_sm, GRAY, cx=True)
    txt(str(iteracion), fuente_lg, WHITE, cx=True)
    y += 2

    k      = estado.n_abonados
    k_real = len(set(estado.ground_truth.values())) if estado.ground_truth else 0
    diff   = abs(k - k_real)
    ck     = COLOR_OK if diff <= 2 else (COLOR_WARN if diff <= 6 else COLOR_BAD)
    txt("k abonados hipotéticos", fuente_sm, GRAY, cx=True)
    txt(f"{k}   (real: {k_real})", fuente_md, ck, cx=True)
    y += 4
    sep()

    # Barra de aceptación global
    total = aceptados + rechazados
    tasa  = aceptados / total * 100 if total > 0 else 0
    txt("ACEPTACIÓN GLOBAL", fuente_sm, GRAY)
    bw = PANEL_W - 32
    pygame.draw.rect(panel, GRAY2, (16, y, bw, 10), border_radius=4)
    fill = int(bw * tasa / 100)
    if fill > 0:
        color_b = COLOR_OK if tasa > 30 else COLOR_WARN
        pygame.draw.rect(panel, color_b, (16, y, fill, 10), border_radius=4)
    y += 14
    txt(f"{aceptados}/{total}  ({tasa:.0f}%)", fuente_sm, WHITE)
    y += 2

    # Barras por tipo de movimiento
    for tipo in ['reasignar', 'birth', 'death']:
        a, t  = tasas.get(tipo, [0, 0])
        p     = a / t * 100 if t > 0 else 0
        c     = COLOR_OK if p > 20 else (COLOR_WARN if p > 0 else COLOR_BAD)
        bw2   = int((PANEL_W - 32) * p / 100)
        pygame.draw.rect(panel, GRAY2, (16, y, PANEL_W - 32, 7), border_radius=3)
        if bw2 > 0:
            pygame.draw.rect(panel, c, (16, y, bw2, 7), border_radius=3)
        y += 9
        txt(f"  {tipo:<10} {a:3d}/{t:<4d}  {p:.0f}%", fuente_sm, c)
    y += 2
    sep()

    # Último movimiento
    txt("ÚLTIMO MOVIMIENTO", fuente_sm, GRAY)
    color_mov = {'reasignar': (170, 170, 255),
                 'birth':     COLOR_OK,
                 'death':     COLOR_WARN}.get(ultimo_mov, GRAY)
    txt(ultimo_mov.upper() if ultimo_mov != '—' else '—',
        fuente_md, color_mov, cx=True)
    y += 4
    sep()

    # ── SLIDER DE VELOCIDAD ─────────────────────
    txt("VELOCIDAD", fuente_sm, GRAY)
    slider.rect.x = 16
    slider.rect.y = y
    slider.rect.w = PANEL_W - 70
    slider.rect.h = 18
    slider.dibujar(panel, fuente_sm)
    y += 36
    sep()

    # Controles
    txt("CONTROLES", fuente_sm, GRAY)
    for ctrl in ["ESPACIO  pausar/reanudar",
                 "A B C    cambiar escenario",
                 "SLIDER   velocidad (1x-20x)",
                 "R        reiniciar",
                 "Q / ESC  salir"]:
        txt(ctrl, fuente_sm, GRAY2)

    surface.blit(panel, (SIM_W, 0))
    pygame.draw.line(surface, GRID_C, (SIM_W, 0), (SIM_W, H), 1)


# ──────────────────────────────────────────────
# Simulador paso a paso
# ──────────────────────────────────────────────

class SimuladorPasoAPaso:
    def __init__(self, escenario):
        self.escenario = escenario
        self._init()

    def _init(self):
        gen = GeneradorSintetico(semilla=7)
        self.registros, self.ground_truth, _ = \
            gen.generar_escenario(self.escenario)

        self.modelo    = ModeloOUPM(self.registros)
        self.rng       = random.Random(42)
        self.propuesta = PropuestaMCMC(self.rng)
        self.modelo.inicializar_aleatorio(self.rng)
        self.mundo     = self.modelo
        self.lp        = self.mundo.log_prob_mundo()

        self.iteracion  = 0
        self.aceptados  = 0
        self.rechazados = 0
        self.tasas      = {'reasignar':[0,0], 'birth':[0,0], 'death':[0,0]}
        self.ultimo_mov = '—'

    def paso(self):
        self.iteracion += 1
        try:
            prop, lq_fwd, lq_bwd, tipo, regs_afectados = self.propuesta.proponer(self.mundo)
        except Exception:
            return self._snap(), None

        # Mismo cálculo que MotorMCMC._paso_mh
        if tipo == 'reasignar' and regs_afectados:
            lp_blanket_prop = prop.log_prob_markov_blanket(regs_afectados)
            lp_blanket_act  = self.mundo.log_prob_markov_blanket(regs_afectados)
            lp_prop = self.lp + (lp_blanket_prop - lp_blanket_act)
        else:
            lp_prop = prop.log_prob_mundo()

        log_alpha = min(0.0, (lp_prop - self.lp) + (lq_bwd - lq_fwd))
        self.tasas[tipo][1] += 1
        self.ultimo_mov = tipo

        reg_afectado = None
        if tipo == 'reasignar' and regs_afectados:
            reg_afectado = regs_afectados[0].id

        if math.log(self.rng.random() + 1e-300) < log_alpha:
            self.mundo = prop
            self.lp    = lp_prop
            self.aceptados += 1
            self.tasas[tipo][0] += 1
        else:
            self.rechazados += 1

        return self._snap(), reg_afectado

    def _snap(self):
        m = self.mundo
        return {
            'n_abonados': len(m.abonados),
            'source':     {rid: ab.id for rid, ab in m.source.items()},
            'abonados':   {ab.id: {} for ab in m.abonados},
        }

    def reiniciar(self):
        self._init()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(
        "MCMC en Vivo — Resolución de Identidades | Cap. 18 AIMA")
    clock = pygame.time.Clock()

    fuente_sm = pygame.font.SysFont('consolas', 11)
    fuente_md = pygame.font.SysFont('consolas', 13, bold=True)
    fuente_lg = pygame.font.SysFont('consolas', 30, bold=True)

    # Slider: 1x a 20x, arranca en 3x
    slider = Slider(x=16, y=0, w=PANEL_W - 70, h=18,
                    val_min=1, val_max=20, valor_inicial=3)

    esc_actual = 'A'
    sim        = SimuladorPasoAPaso(esc_actual)
    estado     = EstadoVisual(sim.registros, sim.ground_truth)
    pausa      = False

    print("\n  Simulación en vivo — Cap. 18 AIMA")
    print("  ESPACIO=pausa  A/B/C=escenario  SLIDER=velocidad  R=reset  Q=salir\n")

    running = True
    while running:

        # ── Eventos ──────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # El slider recibe los eventos con el offset del panel
            if event.type in (pygame.MOUSEBUTTONDOWN,
                               pygame.MOUSEBUTTONUP,
                               pygame.MOUSEMOTION):
                slider.on_event(event, offset_x=SIM_W, offset_y=0)

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    pausa = not pausa
                elif event.key == pygame.K_r:
                    sim.reiniciar()
                    estado = EstadoVisual(sim.registros, sim.ground_truth)
                elif event.key == pygame.K_a:
                    esc_actual = 'A'
                    sim    = SimuladorPasoAPaso('A')
                    estado = EstadoVisual(sim.registros, sim.ground_truth)
                    print("  → Escenario A")
                elif event.key == pygame.K_b:
                    esc_actual = 'B'
                    sim    = SimuladorPasoAPaso('B')
                    estado = EstadoVisual(sim.registros, sim.ground_truth)
                    print("  → Escenario B")
                elif event.key == pygame.K_c:
                    esc_actual = 'C'
                    sim    = SimuladorPasoAPaso('C')
                    estado = EstadoVisual(sim.registros, sim.ground_truth)
                    print("  → Escenario C")

        # ── Avanzar MCMC slider.valor pasos por frame ──
        if not pausa:
            ultima_muestra, ultimo_reg = None, None
            for _ in range(slider.valor):
                muestra, reg = sim.paso()
                ultima_muestra = muestra
                ultimo_reg     = reg
            if ultima_muestra:
                estado.actualizar_mundo(ultima_muestra, ultimo_reg)

        estado.tick()

        # ── Renderizar ────────────────────────────
        sim_surf = pygame.Surface((SIM_W, H))
        dibujar_sim(sim_surf, estado, fuente_sm, fuente_md)
        screen.blit(sim_surf, (0, 0))

        dibujar_panel(
            screen, estado, slider, sim.tasas, sim.ultimo_mov,
            fuente_sm, fuente_md, fuente_lg,
            esc_actual, pausa, sim.iteracion,
            sim.aceptados, sim.rechazados
        )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print("  Simulación finalizada.")


if __name__ == '__main__':
    main()
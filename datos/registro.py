"""
datos/registro.py
Capa de datos: Registro de abonado y generador sintético de escenarios.
Teoría: los registros son las "observaciones" del modelo OUPM (cap. 18.2).
Cada registro fue generado por un abonado real, pero no sabemos cuál.
"""

import random
import math

# Pueden sobreescribirse al construir un Registro con pesos=(...).
PESOS_DEFAULT = (0.40, 0.30, 0.20, 0.10)   # nombre, teléfono, ip, ciudad

# ──────────────────────────────────────────────
# Utilidades de similitud (implementadas desde cero)
# ──────────────────────────────────────────────

def similitud_cadena(a: str, b: str) -> float:
    """
    Distancia de edición normalizada entre dos cadenas (Levenshtein).
    Retorna 1.0 si son idénticas, 0.0 si no comparten nada.
    Implementada desde cero — sin librerías externas.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    # Matriz de programación dinámica
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costo = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + costo
            )
    distancia = dp[n][m]
    return 1.0 - distancia / max(n, m)


def similitud_telefono(t1: str, t2: str) -> float:
    """
    Compara dos teléfonos dígito a dígito.
    Retorna fracción de dígitos coincidentes en posición.
    """
    d1 = ''.join(c for c in t1 if c.isdigit())
    d2 = ''.join(c for c in t2 if c.isdigit())
    if not d1 or not d2:
        return 0.0
    if d1 == d2:
        return 1.0
    # Comparar los últimos 8 dígitos (número local)
    d1, d2 = d1[-8:], d2[-8:]
    long_min = min(len(d1), len(d2))
    coinciden = sum(1 for a, b in zip(d1[-long_min:], d2[-long_min:]) if a == b)
    return coinciden / max(len(d1), len(d2))


def misma_subred(ip1: str, ip2: str) -> float:
    """
    Retorna 1.0 si comparten los primeros dos octetos (misma subred /16),
    0.5 si comparten solo el primero, 0.0 si ninguno.
    """
    p1 = ip1.split('.')
    p2 = ip2.split('.')
    if len(p1) < 2 or len(p2) < 2:
        return 0.0
    if p1[0] == p2[0] and p1[1] == p2[1]:
        return 1.0
    if p1[0] == p2[0]:
        return 0.5
    return 0.0


# ──────────────────────────────────────────────
# Clase principal: Registro
# ──────────────────────────────────────────────

class Registro:
    """
    Un registro observable en la red del operador.
    Corresponde a una "observación" en el modelo OUPM:
      Text(r) ~ NoisyString(Nombre(Source(r)))   [cap. 18.2]

    Parámetros
    ----------
    id_reg    : identificador único, e.g. 'R001'
    nombre    : nombre observado (puede tener ruido tipográfico)
    telefono  : teléfono observado (dígitos, puede tener errores)
    ip        : dirección IP observada (formato 'a.b.c.d')
    ciudad    : ciudad observada (puede ser abreviatura)
    pesos     : tupla (w_nombre, w_tel, w_ip, w_ciudad) con suma ≈ 1.
                Si es None se usan PESOS_DEFAULT.
    """

    def __init__(self, id_reg: str, nombre: str, telefono: str,
                 ip: str, ciudad: str,
                 pesos: tuple = None):
        # ── Validación de campos ──────────────────────────────────────────
        if not isinstance(id_reg, str) or not id_reg.strip():
            raise ValueError(f"id_reg debe ser un string no vacío, recibido: {id_reg!r}")
        if not isinstance(nombre, str) or not nombre.strip():
            raise ValueError(f"nombre vacío en registro {id_reg!r}")
        if not isinstance(telefono, str) or not telefono.strip():
            raise ValueError(f"telefono vacío en registro {id_reg!r}")
        if not isinstance(ip, str) or len(ip.split('.')) != 4:
            raise ValueError(f"ip con formato inválido en registro {id_reg!r}: {ip!r}")
        if not isinstance(ciudad, str) or not ciudad.strip():
            raise ValueError(f"ciudad vacía en registro {id_reg!r}")

        self.id           = id_reg
        self.nombre_obs   = nombre.strip()
        self.telefono_obs = telefono.strip()
        self.ip_obs       = ip.strip()
        self.ciudad_obs   = ciudad.strip()

        # ── Pesos configurables ───────────────────────────────────────────
        if pesos is not None:
            if len(pesos) != 4:
                raise ValueError("pesos debe tener exactamente 4 elementos")
            if abs(sum(pesos) - 1.0) > 0.01:
                raise ValueError(f"Los pesos deben sumar 1.0, suma actual: {sum(pesos)}")
            self._pesos = tuple(pesos)
        else:
            self._pesos = PESOS_DEFAULT

    def similitud(self, otro: "Registro") -> float:
        """
        Similitud agregada entre dos registros usando pesos configurables.
        Combina nombre, teléfono, IP y ciudad.
        Usada por el modelo para calcular P(obs | Source).
        """
        sim_nombre   = similitud_cadena(self.nombre_obs, otro.nombre_obs)
        sim_telefono = similitud_telefono(self.telefono_obs, otro.telefono_obs)
        sim_ip       = misma_subred(self.ip_obs, otro.ip_obs)
        sim_ciudad   = similitud_cadena(self.ciudad_obs, otro.ciudad_obs)

        w_n, w_t, w_i, w_c = self._pesos
        return (w_n * sim_nombre +
                w_t * sim_telefono +
                w_i * sim_ip +
                w_c * sim_ciudad)

    def __repr__(self):
        return (f"Registro({self.id!r}, nombre={self.nombre_obs!r}, "
                f"tel={self.telefono_obs!r}, ip={self.ip_obs!r})")


# ──────────────────────────────────────────────
# Generador sintético de escenarios
# ──────────────────────────────────────────────

# Datos base para generación aleatoria
_NOMBRES_BASE = [
    "Carlos Ruiz", "Ana Torres", "Luis García", "María López",
    "Juan Pérez", "Sandra Gómez", "Pedro Martínez", "Laura Herrera",
    "Diego Vargas", "Claudia Mora", "Felipe Castro", "Valentina Ríos"
]

_CIUDADES = ["Bogotá", "Medellín", "Cali", "Barranquilla",
             "Bucaramanga", "Cartagena", "Pereira", "Manizales"]

_ABREV_CIUDADES = {
    "Bogotá": ["Bta", "Bogota", "BOG"],
    "Medellín": ["Mde", "Medellin", "MED"],
    "Cali": ["CLO", "cali"],
    "Barranquilla": ["Bquilla", "BAQ"],
    "Bucaramanga": ["Buca", "BGA"],
    "Cartagena": ["Ctgna", "CTG"],
    "Pereira": ["Pei", "PEI"],
    "Manizales": ["Man", "MZL"],
}


def _ruido_nombre(nombre: str, nivel: float, rng: random.Random) -> str:
    """Aplica ruido tipográfico a un nombre según nivel [0,1]."""
    if nivel == 0 or rng.random() > nivel:
        return nombre
    ops = []
    partes = nombre.split()
    if len(partes) >= 2 and rng.random() < 0.4:
        partes[0] = partes[0][0] + "."
        ops.append(' '.join(partes))
    if len(partes) >= 2 and rng.random() < 0.3:
        ops.append(partes[-1] + str(rng.randint(10, 99)))
    if rng.random() < 0.3:
        ops.append(nombre.lower().replace('é', 'e').replace('á', 'a'))
    return rng.choice(ops) if ops else nombre


def _ruido_telefono(tel: str, nivel: float, rng: random.Random) -> str:
    """Aplica ruido al teléfono: dígitos cambiados o formatos distintos."""
    if nivel == 0 or rng.random() > nivel:
        return tel
    digitos = list(''.join(c for c in tel if c.isdigit()))
    if rng.random() < nivel * 0.5 and digitos:
        pos = rng.randint(0, len(digitos) - 1)
        digitos[pos] = str((int(digitos[pos]) + rng.randint(1, 3)) % 10)
    return ''.join(digitos)


def _ruido_ip(ip: str, nivel: float, rng: random.Random) -> str:
    """Cambia el último octeto de la IP según nivel de ruido."""
    partes = ip.split('.')
    if len(partes) != 4:
        return ip
    if rng.random() < nivel:
        partes[3] = str(rng.randint(1, 254))
    if nivel > 0.5 and rng.random() < nivel - 0.5:
        partes[2] = str(rng.randint(0, 255))
    return '.'.join(partes)


def _ruido_ciudad(ciudad: str, nivel: float, rng: random.Random) -> str:
    """Usa abreviaturas o variantes ortográficas de la ciudad."""
    if nivel == 0 or rng.random() > nivel:
        return ciudad
    abrevs = _ABREV_CIUDADES.get(ciudad, [ciudad])
    return rng.choice(abrevs)


class GeneradorSintetico:
    """
    Genera registros sintéticos con ruido controlado.
    Simula el proceso generativo del OUPM (cap. 18.2):
      1. Crea abonados reales con atributos verdaderos.
      2. Genera registros como observaciones ruidosas de esos abonados.
    Esto permite evaluar el sistema con ground truth conocido.
    """

    def __init__(self, semilla: int = 42):
        self.rng = random.Random(semilla)

    def generar_escenario(self, tipo: str,
                           n_abonados: int = None,
                           registros_por_abonado: tuple = None,
                           nivel_ruido: float = None,
                           fraude_ratio: float = None) -> tuple:
        """
        Genera un escenario de prueba.
        Retorna (registros, ground_truth, abonados_reales).

        tipo:
          'A' — bajo ruido, identidades claras
          'B' — alto ruido, muchos nombres similares
          'C' — escalabilidad, muchos registros
          'D' — personalizado: usa los parámetros adicionales (todos
                obligatorios si tipo='D').

        Parámetros opcionales (solo para tipo='D'):
          n_abonados             : int, número de abonados reales
          registros_por_abonado  : (min, max) registros por abonado
          nivel_ruido            : float en [0, 1]
          fraude_ratio           : float en [0, 1]
        """
        configs = {
            'A': dict(n_abonados=5,  registros_por_abonado=(2, 3),
                      nivel_ruido=0.10, fraude_ratio=0.0),
            'B': dict(n_abonados=8,  registros_por_abonado=(3, 5),
                      nivel_ruido=0.55, fraude_ratio=0.20),
            'C': dict(n_abonados=20, registros_por_abonado=(3, 6),
                      nivel_ruido=0.35, fraude_ratio=0.10),
        }
        tipo_upper = tipo.upper()
        if tipo_upper == 'D':
            faltantes = [k for k, v in {
                'n_abonados': n_abonados,
                'registros_por_abonado': registros_por_abonado,
                'nivel_ruido': nivel_ruido,
                'fraude_ratio': fraude_ratio,
            }.items() if v is None]
            if faltantes:
                raise ValueError(
                    f"Para tipo='D' debes pasar: {faltantes}"
                )
            cfg = dict(n_abonados=n_abonados,
                       registros_por_abonado=registros_por_abonado,
                       nivel_ruido=nivel_ruido,
                       fraude_ratio=fraude_ratio)
        else:
            cfg = configs.get(tipo_upper, configs['A'])
        return self._generar(**cfg)

    def _generar(self, n_abonados: int, registros_por_abonado: tuple,
                 nivel_ruido: float, fraude_ratio: float) -> tuple:
        rng = self.rng
        nombres_disponibles = self.rng.sample(_NOMBRES_BASE,
                                              min(n_abonados, len(_NOMBRES_BASE)))
        if n_abonados > len(_NOMBRES_BASE):
            extra = [f"Usuario_{i}" for i in range(n_abonados - len(_NOMBRES_BASE))]
            nombres_disponibles += extra

        abonados_reales = []
        for i in range(n_abonados):
            ciudad = rng.choice(_CIUDADES)
            abonados_reales.append({
                'id': i,
                'nombre': nombres_disponibles[i],
                'telefono': f"3{rng.randint(0,2)}{rng.randint(0,9)}"
                            f"{rng.randint(1000000, 9999999)}",
                'ip': f"192.168.{rng.randint(1,20)}.{rng.randint(1,254)}",
                'ciudad': ciudad,
                'fraudulento': rng.random() < fraude_ratio,
            })

        registros = []
        ground_truth = {}
        contador = 0

        for ab in abonados_reales:
            n_regs = rng.randint(*registros_por_abonado)
            if ab['fraudulento']:
                n_regs = rng.randint(n_regs, n_regs + 3)
            for _ in range(n_regs):
                rid = f"R{contador:03d}"
                reg = Registro(
                    id_reg=rid,
                    nombre=_ruido_nombre(ab['nombre'], nivel_ruido, rng),
                    telefono=_ruido_telefono(ab['telefono'], nivel_ruido, rng),
                    ip=_ruido_ip(ab['ip'], nivel_ruido, rng),
                    ciudad=_ruido_ciudad(ab['ciudad'], nivel_ruido, rng),
                )
                registros.append(reg)
                ground_truth[rid] = ab['id']
                contador += 1

        rng.shuffle(registros)
        return registros, ground_truth, abonados_reales

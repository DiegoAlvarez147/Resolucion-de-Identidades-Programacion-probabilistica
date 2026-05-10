# Proyecto Identidades

Sistema probabilistico de resolucion de identidades basado en el modelo OUPM (cap. 18.2 de AIMA 4ed). Genera registros sinteticos con ruido, ejecuta MCMC para inferir la asignacion de registros a abonados reales, y compara con ground truth.

## Estructura del proyecto

- analisis/: visualizaciones y diagnosticos del MCMC (matplotlib). Ejecuta los 3 escenarios y genera figuras.
- datos/: capa de datos. Define `Registro` y el `GeneradorSintetico` de escenarios A/B/C con ruido controlado.
- inferencia/: motor MCMC (Metropolis-Hastings) con movimientos de reasignacion, birth y death.
- modelo/: definicion del modelo OUPM y la log-probabilidad del mundo.
- resultados/: carpeta sugerida para guardar salidas o figuras.
- main.py: punto de entrada que corre los 3 escenarios y muestra metricas.
- simulacion_vivo.py: simulacion interactiva en tiempo real con pygame.

## Requisitos

- Python 3.10+ recomendado
- Para `simulacion_vivo.py`: `pygame`
- Para `analisis/visualizacion.py`: `matplotlib`

## Como ejecutar

Desde la raiz del proyecto:

```powershell
python main.py
```

Esto corre los 3 escenarios (A, B, C), imprime el resumen comparativo y metricas de precision y error en k.

### Simulacion en vivo (pygame)

```powershell
python simulacion_vivo.py
```

**Controles en la simulacion:**

- ESPACIO: pausar / reanudar
- A / B / C: cambiar escenario
- Slider (mouse): velocidad de simulacion (1x-20x)
- R: reiniciar
- Q / ESC: salir

## Escenarios A/B/C

Los escenarios se generan en `datos/registro.py`:

- A: bajo ruido, identidades mas claras
- B: alto ruido, nombres mas similares
- C: escalabilidad, muchos registros

## Notas

Si quieres ver las figuras de analisis, ejecuta:

```powershell
python analisis/visualizacion.py
```

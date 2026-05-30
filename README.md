# challenge6__3
Repositorio para el Challenge 6 de Machine Learning: Aprendizaje no supervisado avanzado, detección de anomalías y aprendizaje de representaciones utilizando AutoEncoders y VAEs. Dominio: Salud Pública y Epidemiología (BRFSS). Grupo: 3.

## Punto de inicio

El flujo principal de Challenge 6 ya queda centralizado en [src/pipeline.py](src/pipeline.py) y se puede ejecutar desde el notebook [notebooks/challenge6/01_group3_challenge6.ipynb](notebooks/challenge6/01_group3_challenge6.ipynb).

Los artefactos de Challenge 5 se leen desde [results/challenge5](results/challenge5) y las salidas de Challenge 6 se escriben en carpetas separadas:

- [results/challenge6](results/challenge6)
- [figures/challenge6](figures/challenge6)
- [weights/challenge6](weights/challenge6)

Para correrlo desde consola:

```bash
python src/pipeline.py
```

Eso entrena AE, VAE e Isolation Forest con la misma matriz preprocesada de Challenge 5, genera las tablas resumen y guarda las figuras requeridas.

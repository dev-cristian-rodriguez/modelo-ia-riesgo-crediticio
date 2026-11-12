# Modelo de IA para riesgo crediticio con auditoría y mitigación de sesgos

Proyecto de la Unidad 2 de la Electiva II: Inteligencia Artificial Avanzada (Universidad Tecnológica del Oriente), desarrollado con Aprendizaje Basado en Retos.

Se predice si un solicitante de crédito incumplirá, usando el conjunto Statlog German Credit (1.000 solicitudes, 20 atributos), y se audita la equidad del modelo por edad (menores de 25 años frente a 25 o más) y por sexo.

## Modelos

| Nombre | Descripción |
|---|---|
| M0 Modelo simple | Regresión logística (línea base) |
| M1 Modelo avanzado | XGBoost con 60 configuraciones de hiperparámetros evaluadas por AUC de validación |
| M2 Sin edad ni sexo | XGBoost sin atributos sensibles (equidad por desconocimiento) |
| M3 Pesos por grupo | XGBoost con *reweighing* (Kamiran y Calders, 2012) |
| M4 Umbral por grupo | XGBoost con `ThresholdOptimizer` de Fairlearn (equalized odds) |

La partición es estratificada: 60 % entrenamiento, 20 % validación y 20 % prueba (usada una sola vez). El umbral de decisión de cada modelo es el que maximiza el F1 en validación.

## Resultados principales (conjunto de prueba)

| Modelo | F1 | AUC | Impacto dispar (edad) | EOD (edad) |
|---|---|---|---|---|
| M0 Modelo simple | 0,599 | 0,755 | 0,670 | 0,188 |
| M1 Modelo avanzado | 0,485 | 0,696 | 0,973 | 0,020 |
| M2 Sin edad ni sexo | 0,480 | 0,696 | 0,986 | 0,032 |
| M3 Pesos por grupo | 0,471 | 0,685 | 0,992 | 0,062 |
| M4 Umbral por grupo | 0,432 | n.d. | 0,919 | 0,188 |

- XGBoost se sobreajusta (AUC de 0,997 en entrenamiento y 0,696 en prueba) y no supera a la línea base.
- La regresión logística incumple la regla de los cuatro quintos por edad (impacto dispar de 0,67).
- Las mitigaciones aproximan la paridad en aprobación, pero desplazan el desequilibrio a otras métricas. Las conclusiones de equidad tienen incertidumbre alta por el tamaño de la muestra (30 menores de 25 años en prueba).

Los resultados completos están en `outputs/` (`results.json`, tablas en CSV y 6 figuras). La presentación está en `Presentacion_Modelo_IA_Riesgo_Crediticio.pdf`.

## Reproducibilidad

```bash
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
python credit_fairness.py --out outputs --seed 42 --n-iter 60
```

El script descarga el conjunto de datos desde OpenML (requiere conexión a internet), ejecuta el flujo completo y escribe los resultados en la carpeta indicada. Con la semilla 42 reproduce exactamente las cifras de esta página y del informe.

## Licencia

MIT. Ver [LICENSE.txt](LICENSE.txt).

## Autor

Cristian Estiven Rodriguez Benitez, Ingeniería de Software, Universidad Tecnológica del Oriente.

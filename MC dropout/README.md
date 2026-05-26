# Monte Carlo Dropout Dynamics Predictor para F1TENTH / Racing Control

Este módulo entrena una red neuronal con **Monte Carlo Dropout** para modelar la dinámica one-step de un vehículo a partir de datos en formato `.txt`. El objetivo es obtener un predictor dinámico capaz de estimar no solo el estado futuro medio del vehículo, sino también una medida de incertidumbre asociada a la predicción.

Este predictor está pensado como componente de un esquema mayor de tesis/paper sobre:

```text
Explainable and Uncertainty-Aware Multi-Step Dynamic Feasibility Prediction
for Adaptive Speed Scaling in Autonomous Racing
```

La idea general es usar un modelo aprendido para predecir el comportamiento dinámico futuro del vehículo bajo una secuencia de controles generada por un controlador Pure Pursuit, y usar esa predicción para adaptar la velocidad de forma más agresiva pero dinámica y probabilísticamente controlada.

---

## 1. Objetivo general

El objetivo del módulo es aprender una función de dinámica:

```text
x(t+1) = f([x(t-H), u(t-H), ..., x(t), u(t)])
```

donde:

```text
x(t) = [Vy, AVz, Yaw, Beta, Ax, Ay, AVx, Roll]
```

y:

```text
u(t) = [Steer, Vx]
```

El modelo recibe una ventana temporal histórica de estados y controles y predice el siguiente estado dinámico del vehículo.

La diferencia respecto a una red neuronal determinística convencional es que, usando **Monte Carlo Dropout**, el modelo puede producir:

```text
media de la predicción
incertidumbre de la predicción
```

Esto permite calcular márgenes dinámicos conservadores para variables críticas como:

```text
Beta
AVz
Ay
Roll
```

---

## 2. Motivación dentro del paper

El paper no busca simplemente entrenar un predictor dinámico. El objetivo más amplio es usar la predicción multi-step para regular la agresividad del vehículo.

El contexto es el siguiente:

1. Existe una trayectoria o racing line conocida.
2. Un controlador Pure Pursuit sigue esa trayectoria.
3. El Pure Pursuit entrega comandos nominales:

```text
u(t) = [Steer(t), Vx(t)]
```

4. El modelo aprendido predice cómo evolucionará el vehículo en una ventana futura.
5. Se evalúa si las variables dinámicas permanecerán dentro de límites aceptables.
6. Si hay margen dinámico suficiente, se puede aumentar la velocidad.
7. Si se predice una violación, se reduce o limita la velocidad.

La idea de control puede resumirse como:

```text
Aumentar la velocidad cuando el rollout futuro permanece dentro del envelope dinámico.
Reducir la velocidad cuando el rollout futuro predice riesgo dinámico.
```

---

## 3. Dataset

El dataset está formado por múltiples archivos `.txt`, cada uno correspondiente a un episodio, maniobra, vehículo o condición específica.

Ejemplo de estructura:

```text
DS/
├── A_ATB_02.txt
├── A_ATB_04.txt
├── A_DLC_10.txt
├── A_FH_10.txt
├── A_SSI_02.txt
├── A_SWD_10.txt
├── B_ATB_02.txt
├── B_DLC_10.txt
├── DSUV_ATB_10.txt
├── LEV_SSI_10.txt
├── ORP_SWD_10.txt
└── ...
```

Cada archivo debe contener como mínimo las columnas:

```text
time,Vx,Vy,AVz,Yaw,Beta,Ax,Ay,AVx,Roll,Steer
```

Ejemplo de encabezado:

```csv
time,Vx,Vy,AVz,Yaw,Beta,Ax,Ay,AVx,Roll,Steer
```

---

## 4. Variables usadas

### Estados dinámicos

```python
STATE_COLS = [
    "Vy", "AVz", "Yaw", "Beta",
    "Ax", "Ay", "AVx", "Roll"
]
```

### Controles

```python
CONTROL_COLS = [
    "Steer", "Vx"
]
```

### Features de entrada

```python
FEATURE_COLS = STATE_COLS + CONTROL_COLS
```

Es decir, cada instante de la ventana histórica contiene 10 variables:

```text
Vy, AVz, Yaw, Beta, Ax, Ay, AVx, Roll, Steer, Vx
```

Si se usa:

```text
history_len = 20
```

entonces la entrada contiene 21 instantes:

```text
t-20, t-19, ..., t
```

Por tanto, la dimensión de entrada será:

```text
21 × 10 = 210
```

---

## 5. Por qué no se deben mezclar directamente los archivos

Cada `.txt` representa un episodio o maniobra independiente. Por eso, el entrenamiento **no debe concatenar todos los archivos y luego construir ventanas**, porque eso podría crear una muestra artificial donde la historia viene del final de una maniobra y el target del inicio de otra.

Incorrecto:

```python
df_global = pd.concat([df1, df2, df3])
# Luego construir ventanas sobre df_global
```

Correcto:

```text
1. Leer cada archivo por separado.
2. Construir ventanas dentro de cada archivo.
3. Concatenar las muestras resultantes.
```

Esto evita transiciones dinámicamente falsas.

---

## 6. Monte Carlo Dropout

### 6.1. Idea básica

Dropout normalmente se usa durante entrenamiento para regularizar la red. Durante inferencia, usualmente se desactiva.

En Monte Carlo Dropout se mantiene dropout activo durante inferencia y se hacen varias predicciones para la misma entrada.

Para una misma ventana:

```text
X_window
```

se calculan varias salidas:

```text
y_1 = model(X_window)
y_2 = model(X_window)
...
y_N = model(X_window)
```

Luego se calcula:

```text
media = predicción final
desviación estándar = incertidumbre
```

---

### 6.2. Interpretación

Si el modelo predice:

```text
Beta(t+1) = 0.035 ± 0.010
```

eso significa:

```text
predicción media de Beta = 0.035
incertidumbre estimada = 0.010
```

Esta incertidumbre se puede usar para construir márgenes conservadores:

```text
|Beta_mean| + k * Beta_std <= Beta_max
```

donde `k` controla el nivel de conservadurismo.

Ejemplos:

```text
k = 0      → predicción determinística
k = 1      → margen moderado
k = 1.96   → aproximación tipo 95%
k = 2.58   → aproximación tipo 99%
k = 3      → margen conservador
```

---

## 7. Dependencias

Instalar dependencias principales:

```powershell
python -m pip install torch numpy pandas scikit-learn joblib matplotlib
```

Si se usa GPU, instalar PyTorch según la versión de CUDA desde la página oficial de PyTorch.

---

## 8. Script de entrenamiento

El entrenamiento se realiza con:

```text
train_mc_dropout_dyn.py
```

Este script:

1. Lee todos los archivos `.txt` de una carpeta.
2. Construye ventanas históricas dentro de cada episodio.
3. Crea el target `x(t+1)`.
4. Divide los datos en train, validation y test.
5. Escala entradas y salidas con `StandardScaler`.
6. Entrena una red neuronal multi-output con Dropout.
7. Guarda el mejor modelo según `val_loss`.
8. Guarda métricas determinísticas de test.
9. Guarda scalers y metadata.

---

## 9. Arquitectura de la red

La red base usada es una MLP con Dropout:

```text
Input
  ↓
Linear + ReLU + Dropout
  ↓
Linear + ReLU + Dropout
  ↓
Linear + ReLU + Dropout
  ↓
Output
```

La salida tiene dimensión 8:

```text
[Vy, AVz, Yaw, Beta, Ax, Ay, AVx, Roll] at t+1
```

Ejemplo de hiperparámetros iniciales:

```text
hidden_dim = 256
dropout_p = 0.10
batch_size = 512
learning_rate = 1e-3
weight_decay = 1e-5
epochs = 100
```

---

## 10. Entrenamiento rápido de prueba

Antes de entrenar con todo el dataset, se recomienda probar con pocos archivos:

```powershell
python train_mc_dropout_dyn.py `
  --data_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\GP\DS" `
  --save_dir saved_mc_dropout_test `
  --history_len 20 `
  --stride 20 `
  --max_files 5 `
  --epochs 30 `
  --batch_size 512
```

Esto sirve para verificar que:

```text
el dataset carga correctamente
las columnas son reconocidas
la red entrena sin errores
los scalers y metadata se guardan
```

---

## 11. Entrenamiento completo sugerido

Una vez validado el pipeline:

```powershell
python train_mc_dropout_dyn.py `
  --data_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\GP\DS" `
  --save_dir saved_mc_dropout_h20 `
  --history_len 20 `
  --stride 10 `
  --epochs 100 `
  --batch_size 512 `
  --dropout_p 0.10
```

---

## 12. Argumentos principales del entrenamiento

### `--data_folder`

Carpeta con archivos `.txt`.

```powershell
--data_folder "C:\...\DS"
```

### `--save_dir`

Carpeta donde se guardan modelo, scalers, métricas y metadata.

```powershell
--save_dir saved_mc_dropout_h20
```

### `--history_len`

Longitud de historia usada como entrada.

```powershell
--history_len 20
```

Con `history_len = 20`, la entrada contiene:

```text
[x(t-20), u(t-20), ..., x(t), u(t)]
```

### `--stride`

Salto temporal para construir muestras.

```powershell
--stride 10
```

Un valor mayor reduce la cantidad de muestras.

### `--dropout_p`

Probabilidad de dropout.

```powershell
--dropout_p 0.10
```

Valores recomendados para probar:

```text
0.05
0.10
0.20
```

### `--max_files`

Número máximo de archivos a usar.

```powershell
--max_files 5
```

Útil para pruebas rápidas.

### `--cpu`

Fuerza entrenamiento en CPU.

```powershell
--cpu
```

---

## 13. Archivos generados por el entrenamiento

El entrenamiento genera una carpeta como:

```text
saved_mc_dropout_h20/
├── model.pt
├── x_scaler.pkl
├── y_scaler.pkl
├── metadata.json
├── train_history.csv
└── test_metrics_deterministic.csv
```

### `model.pt`

Pesos del modelo PyTorch.

### `x_scaler.pkl`

Scaler usado para normalizar las entradas.

### `y_scaler.pkl`

Scaler usado para normalizar las salidas.

### `metadata.json`

Contiene información necesaria para inferencia:

```text
state_cols
control_cols
feature_cols
history_len
stride
input_dim
output_dim
hidden_dim
dropout_p
train_ratio
val_ratio
```

### `test_metrics_deterministic.csv`

Métricas de test con dropout desactivado:

```text
MAE
RMSE
error_mean
error_median
error_p90
error_p95
error_max
```

---

## 14. Inferencia con Monte Carlo Dropout

La inferencia se realiza con:

```text
infer_mc_dropout_test.py
```

Este script:

1. Carga el modelo entrenado.
2. Carga `x_scaler.pkl` y `y_scaler.pkl`.
3. Reconstruye el dataset de test usando los mismos parámetros del entrenamiento.
4. Activa Dropout durante inferencia.
5. Ejecuta `N` predicciones por muestra.
6. Calcula media y desviación estándar por variable.
7. Guarda métricas y predicciones detalladas.

---

## 15. Ejecutar inferencia

```powershell
python infer_mc_dropout_test.py `
  --data_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\GP\DS" `
  --model_dir saved_mc_dropout_h20 `
  --output_dir mc_dropout_results_h20 `
  --n_mc 50
```

---

## 16. Argumentos principales de inferencia

### `--model_dir`

Carpeta con el modelo entrenado.

```powershell
--model_dir saved_mc_dropout_h20
```

Debe contener:

```text
model.pt
x_scaler.pkl
y_scaler.pkl
metadata.json
```

### `--n_mc`

Número de predicciones Monte Carlo por muestra.

```powershell
--n_mc 50
```

Valores típicos:

```text
20  → rápido
50  → balanceado
100 → más estable, más lento
```

### `--output_dir`

Carpeta donde se guardan los resultados de inferencia.

```powershell
--output_dir mc_dropout_results_h20
```

---

## 17. Archivos generados por la inferencia

La inferencia genera:

```text
mc_dropout_results_h20/
├── mc_dropout_metrics.csv
└── mc_dropout_predictions.csv
```

---

## 18. `mc_dropout_metrics.csv`

Contiene métricas agregadas por variable:

```text
state
mae
rmse
error_mean
error_median
error_p90
error_p95
error_max
std_mean
std_median
std_p90
std_p95
std_max
```

Las métricas de error se calculan respecto a la media de las predicciones Monte Carlo.

Las métricas `std_*` describen la incertidumbre estimada por MC Dropout.

---

## 19. `mc_dropout_predictions.csv`

Contiene predicciones muestra por muestra.

Ejemplo de columnas:

```text
file
row_target

Beta_true
Beta_pred_mean
Beta_error
Beta_abs_error
Beta_std_mc

AVz_true
AVz_pred_mean
AVz_error
AVz_abs_error
AVz_std_mc

Ay_true
Ay_pred_mean
Ay_error
Ay_abs_error
Ay_std_mc
```

Donde:

```text
*_true       = valor real en x(t+1)
*_pred_mean  = media predicha por MC Dropout
*_error      = true - pred_mean
*_abs_error  = |true - pred_mean|
*_std_mc     = incertidumbre estimada por MC Dropout
```

---

## 20. Cómo interpretar los resultados

### Buen desempeño

Un modelo útil debería tener:

```text
MAE bajo
RMSE bajo
p95 error aceptable
```

Pero para este proyecto no basta con el error promedio.

También interesa:

```text
si la incertidumbre aumenta cuando aumenta el error
si los errores altos se concentran en ciertas maniobras
si Beta, AVz, Ay o Roll tienen colas de error
```

---

### Posibles patrones

#### Caso 1: bajo error y baja incertidumbre

```text
Región confiable
```

#### Caso 2: bajo error y alta incertidumbre

```text
Modelo conservador
```

#### Caso 3: alto error y alta incertidumbre

```text
Región difícil detectada
```

#### Caso 4: alto error y baja incertidumbre

```text
Caso peligroso: modelo sobreconfiado
```

Este último caso es especialmente importante para control cerca del límite dinámico.

---

## 21. Margen dinámico con incertidumbre

La salida del modelo puede usarse para calcular un margen dinámico.

Para una variable crítica:

```text
m_Beta = Beta_max - (|Beta_pred_mean| + k * Beta_std_mc)
```

Análogamente:

```text
m_AVz  = AVz_max  - (|AVz_pred_mean|  + k * AVz_std_mc)
m_Ay   = Ay_max   - (|Ay_pred_mean|   + k * Ay_std_mc)
m_Roll = Roll_max - (|Roll_pred_mean| + k * Roll_std_mc)
```

Para una ventana futura multi-step:

```text
M = min_k min_j m_j,k
```

donde:

```text
k = horizonte futuro
j = variable dinámica crítica
```

Interpretación:

```text
M > 0  → secuencia futura viable
M ≈ 0  → operación cerca del límite
M < 0  → predicción de violación dinámica
```

Este margen es el núcleo del supervisor predictivo de velocidad.

---

## 22. Uso del modelo en el supervisor de velocidad

El controlador Pure Pursuit genera:

```text
u_nom(t) = [Steer_nom(t), Vx_nom(t)]
```

El supervisor evalúa candidatos:

```text
Vx_cmd = lambda * Vx_nom
```

con:

```text
lambda > 1  → más agresivo
lambda = 1  → nominal
lambda < 1  → conservador
```

Para cada `lambda`:

1. Se genera una secuencia futura de controles.
2. El modelo predice el rollout futuro.
3. Se calcula el margen dinámico `M(lambda)`.
4. Se elige el mayor `lambda` viable.

La regla de decisión:

```text
lambda* = max lambda
subject to M(lambda) >= 0
```

---

## 23. Explicabilidad del supervisor

Además de decidir `lambda`, se puede identificar:

```text
variable crítica
horizonte crítico
```

La variable crítica es aquella que más limita la velocidad:

```text
critical_variable = argmin_j,k m_j,k
```

El horizonte crítico es el paso futuro donde aparece el menor margen:

```text
critical_horizon = argmin_k m_j,k
```

Ejemplo de explicación:

```text
La velocidad no fue aumentada porque Beta se aproxima al límite en t+8.
```

o:

```text
El supervisor redujo la velocidad porque Ay presenta bajo margen dinámico en la entrada de la curva.
```

Esto conecta el método con XAI a nivel de decisión de control.

---

## 24. Métricas útiles para el paper

Para evaluar el predictor:

```text
MAE por variable
RMSE por variable
p95 error por variable
error vs incertidumbre
calibración de incertidumbre
outliers por maniobra
```

Para evaluar el controlador:

```text
lap time
mean speed
max speed
track completion rate
crash rate
p95 |Beta|
p95 |AVz|
p95 |Ay|
p95 |Roll|
minimum dynamic margin
percentage of time near dynamic boundary
predicted dynamic violations
actual dynamic violations
```

---

## 25. Experimentos sugeridos

### Experimento 1: PP nominal

```text
Pure Pursuit con velocidad nominal de la racing line.
```

### Experimento 2: PP agresivo fijo

```text
Pure Pursuit con lambda fijo mayor que 1.
```

Ejemplo:

```text
lambda = 1.2
```

### Experimento 3: PP predictivo

```text
Pure Pursuit con supervisor predictivo basado en rollout e incertidumbre.
```

### Experimento 4: ablation de incertidumbre

Comparar:

```text
k = 0      → mean-only
k = 1      → incertidumbre moderada
k = 1.96   → margen más conservador
k = 3      → margen muy conservador
```

---

## 26. Figuras sugeridas para el paper

1. Arquitectura del método:

```text
Pure Pursuit → modelo predictivo → margen dinámico → speed scaling
```

2. Error de rollout vs horizonte.

3. Bandas de incertidumbre para:

```text
Beta
AVz
Ay
Roll
```

4. `lambda(t)` a lo largo de la pista.

5. Margen dinámico `M(t)` a lo largo de la pista.

6. Mapa de pista coloreado por variable crítica:

```text
Beta
AVz
Ay
Roll
```

7. Mapa de pista coloreado por horizonte crítico.

8. Comparación de controladores:

```text
PP nominal
PP agresivo fijo
PP predictivo
```

---

## 27. Cosas que conviene evitar en este paper

Evitar afirmar:

```text
El controlador explota los límites físicos reales del vehículo.
```

si la validación está hecha solo en F1TENTH Gym.

Mejor usar:

```text
dynamic feasibility envelope
predicted dynamic envelope
algorithmic validation in F1TENTH Gym
```

También conviene evitar incluir demasiadas contribuciones a la vez:

```text
SCBF formal completa
ECBF formal completa
MPC completo
control conjunto de steer y Vx
garantías matemáticas fuertes
```

Estos elementos pueden quedar como trabajo futuro.

---

## 28. Limitaciones importantes

F1TENTH Gym es útil para validar lógica de control y racing, pero su dinámica puede ser limitada frente a simuladores más realistas.

Por tanto, este trabajo debe presentarse como:

```text
validación algorítmica de un supervisor predictivo de viabilidad dinámica
```

y no como:

```text
validación física definitiva de conducción en el límite real de adherencia
```

Validaciones futuras pueden incluir:

```text
CarSim
simuladores de mayor fidelidad
hardware F1TENTH
MPC
CBF / SCBF
modelos híbridos físico-aprendidos
```

---

## 29. Recomendaciones iniciales

Entrenar primero con:

```text
history_len = 20
stride = 10
dropout_p = 0.10
n_mc = 50
```

Luego revisar principalmente:

```text
Beta
AVz
Ay
Roll
```

Analizar:

```text
error medio
error p95
std p95
relación entre error e incertidumbre
outliers por archivo o maniobra
```

Si la incertidumbre no acompaña bien el error, probar:

```text
dropout_p = 0.20
ensemble neuronal
GP residual
calibración de incertidumbre
```

---

## 30. Resumen del pipeline

```text
1. Dataset multi-episodio en .txt
2. Construcción de ventanas históricas dentro de cada episodio
3. Entrenamiento de MLP con Dropout
4. Inferencia Monte Carlo Dropout
5. Estimación de media e incertidumbre
6. Cálculo de márgenes dinámicos
7. Rollout multi-step
8. Speed scaling adaptativo
9. Identificación de variable crítica y horizonte crítico
10. Evaluación en F1TENTH Gym
```

---

## 31. Idea central del paper

La idea central no es simplemente entrenar una red neuronal.

La contribución principal es:

```text
Usar predicción multi-step con incertidumbre para estimar la viabilidad dinámica futura de una secuencia de control y adaptar la velocidad del vehículo de forma explicable.
```

En otras palabras:

```text
La incertidumbre responde:
¿cuánto puedo confiar en la predicción?

La explicabilidad responde:
¿qué variable y qué instante futuro están limitando la velocidad?

El supervisor responde:
¿cuánto puedo aumentar Vx sin cruzar el envelope dinámico predicho?
```

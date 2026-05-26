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
u(t) = [Steer, Vx]
```

El modelo recibe una ventana temporal histórica de estados y controles y predice el siguiente estado dinámico del vehículo. Con **Monte Carlo Dropout**, el modelo puede producir:

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

Cada archivo debe contener como mínimo las columnas:

```text
time,Vx,Vy,AVz,Yaw,Beta,Ax,Ay,AVx,Roll,Steer
```

Los nombres de archivo siguen una estructura aproximada:

```text
<vehiculo/configuracion>_<maniobra>_<nivel>.txt
```

Ejemplos:

```text
A_ATB_02.txt
A_DLC_10.txt
B_SWD_08.txt
DSUV_SSI_10.txt
LEV_FH_08.txt
ORP_SWD_10.txt
```

donde:

```text
A, B, DSUV, ER, LEV, ORP  → vehículo/configuración
ATB, DLC, FH, SSI, SWD    → tipo de maniobra
02, 04, 06, 08, 10        → nivel/condición/intensidad
```

---

## 4. Separación explícita en Train / Validation / Test

La versión actual del pipeline usa **carpetas separadas** para entrenamiento, validación y test.

La estructura esperada es:

```text
DS/
├── Train/
│   ├── A_ATB_02.txt
│   ├── A_DLC_10.txt
│   ├── B_SSI_08.txt
│   ├── DSUV_SWD_10.txt
│   └── ...
├── Validation/
│   ├── LEV_ATB_02.txt
│   ├── LEV_DLC_10.txt
│   ├── LEV_SSI_08.txt
│   └── ...
└── Test/
    ├── ORP_ATB_02.txt
    ├── ORP_DLC_10.txt
    ├── ORP_SSI_08.txt
    └── ...
```

La división recomendada para el paper es:

```text
Train:
    A, B, DSUV, ER

Validation:
    LEV

Test:
    ORP
```

Esta separación evita que el modelo sea evaluado en ventanas temporales casi idénticas a las usadas durante entrenamiento.

### Por qué no usar split aleatorio por muestras

No se recomienda dividir aleatoriamente ventanas individuales en train/validation/test, porque las ventanas temporales son altamente correlacionadas.

Ejemplo problemático:

```text
train: ventana de A_DLC_10.txt en t = 1.000 s
test:  ventana de A_DLC_10.txt en t = 1.010 s
```

Eso produce leakage temporal y puede inflar artificialmente el desempeño.

### Qué cambió respecto a la versión anterior

La versión anterior usaba un único argumento:

```text
--data_folder
```

y después dividía internamente las muestras en train/validation/test.

La versión actual usa tres argumentos separados:

```text
--train_folder
--val_folder
--test_folder
```

Por tanto, el script **no debe volver a dividir internamente** la carpeta `Train` en train/validation/test.

---

## 5. Variables usadas

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

Cada instante de la ventana histórica contiene 10 variables:

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

## 6. Tratamiento de Beta

Durante las pruebas iniciales se observó que la señal cruda de `Beta` puede presentar artefactos de envolvimiento angular, especialmente en maniobras tipo ATB, cuando:

```text
Vx ≈ 0 o Vx < 0 muy pequeño
Vy ≈ 0
```

En esos casos, un cálculo tipo:

```text
Beta = atan2(Vy, Vx)
```

puede generar valores espurios cercanos a:

```text
±180°
```

aunque físicamente no exista deriva lateral significativa.

Por ello, se recomienda recalcular `Beta` de forma robusta dentro de la función de lectura del episodio:

```python
def read_episode(file_path):
    df = pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "Vx" in df.columns and "Vy" in df.columns:
        vx = df["Vx"].values.astype(float)
        vy = df["Vy"].values.astype(float)

        eps = 1e-6
        beta_rad = np.arctan2(vy, np.maximum(np.abs(vx), eps))
        df["Beta"] = np.rad2deg(beta_rad)

    return df
```

Esta corrección debe aplicarse de forma idéntica en:

```text
entrenamiento
validación
test
inferencia
rollout multi-step
```

---

## 7. Por qué no se deben mezclar directamente los archivos

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
3. Concatenar las muestras resultantes dentro del split correspondiente.
```

Esto evita transiciones dinámicamente falsas entre maniobras.

---

## 8. Monte Carlo Dropout

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

donde `k` controla el nivel de conservadurismo:

```text
k = 0      → predicción determinística
k = 1      → margen moderado
k = 1.96   → aproximación tipo 95%
k = 2.58   → aproximación tipo 99%
k = 3      → margen conservador
```

---

## 9. Dependencias

Instalar dependencias principales:

```powershell
python -m pip install torch numpy pandas scikit-learn joblib matplotlib
```

Si se usa GPU, instalar PyTorch según la versión de CUDA desde la página oficial de PyTorch.

---

## 10. Script de entrenamiento

El entrenamiento se realiza con:

```text
MC_training.py
```

Este script:

1. Lee archivos `.txt` desde `--train_folder`.
2. Lee archivos `.txt` desde `--val_folder`.
3. Lee archivos `.txt` desde `--test_folder`.
4. Construye ventanas históricas dentro de cada episodio.
5. Crea el target `x(t+1)`.
6. Escala entradas y salidas con `StandardScaler` ajustado **solo con Train**.
7. Aplica el scaler de Train a Validation y Test.
8. Entrena una red neuronal multi-output con Dropout.
9. Usa `Validation` para early stopping.
10. Evalúa el mejor modelo en `Test`.
11. Guarda scalers, modelo, métricas y metadata.

El script **no debe hacer split interno** usando `train_ratio` o `val_ratio` cuando se usan carpetas separadas.

---

## 11. Arquitectura de la red

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

## 12. Entrenamiento rápido de prueba

Para una prueba rápida, se recomienda crear carpetas reducidas:

```text
DS_debug/
├── Train/
├── Validation/
└── Test/
```

con pocos archivos representativos en cada una.

Ejemplo:

```powershell
python MC_training.py `
  --train_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS_debug\Train" `
  --val_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS_debug\Validation" `
  --test_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS_debug\Test" `
  --save_dir saved_mc_dropout_debug `
  --history_len 20 `
  --stride 20 `
  --epochs 30 `
  --batch_size 512 `
  --dropout_p 0.10
```

Esto sirve para verificar que:

```text
el dataset carga correctamente
las columnas son reconocidas
las tres carpetas se leen sin errores
la red entrena sin errores
los scalers y metadata se guardan
```

---

## 13. Entrenamiento completo sugerido

Una vez validado el pipeline:

```powershell
python MC_training.py `
  --train_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS\Train" `
  --val_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS\Validation" `
  --test_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS\Test" `
  --save_dir saved_mc_dropout_h20 `
  --history_len 20 `
  --stride 10 `
  --epochs 100 `
  --batch_size 512 `
  --dropout_p 0.10
```

También puede usarse una sola línea:

```powershell
python MC_training.py --train_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS\Train" --val_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS\Validation" --test_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS\Test" --save_dir saved_mc_dropout_h20 --history_len 20 --stride 10 --epochs 100 --batch_size 512 --dropout_p 0.10
```

---

## 14. Argumentos principales del entrenamiento

### `--train_folder`

Carpeta con archivos `.txt` usados para ajustar los pesos de la red.

```powershell
--train_folder "C:\...\DS\Train"
```

### `--val_folder`

Carpeta con archivos `.txt` usados para validación y early stopping.

```powershell
--val_folder "C:\...\DS\Validation"
```

### `--test_folder`

Carpeta con archivos `.txt` usados para evaluación final.

```powershell
--test_folder "C:\...\DS\Test"
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

### `--max_train_files`, `--max_val_files`, `--max_test_files`

Opcionalmente, pueden usarse para limitar la cantidad de archivos por split durante pruebas rápidas:

```powershell
--max_train_files 10
--max_val_files 5
--max_test_files 5
```

Si tu script no implementa estos tres argumentos, se puede mantener temporalmente un único `--max_files`, pero lo más recomendable es que cada split tenga su propio límite.

### `--cpu`

Fuerza entrenamiento en CPU.

```powershell
--cpu
```

---

## 15. Archivos generados por el entrenamiento

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

Scaler usado para normalizar las entradas. Se ajusta únicamente con `Train`.

### `y_scaler.pkl`

Scaler usado para normalizar las salidas. Se ajusta únicamente con `Train`.

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
train_folder
val_folder
test_folder
num_train_samples
num_val_samples
num_test_samples
```

### `train_history.csv`

Histórico de pérdidas:

```text
epoch
train_loss
val_loss
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

## 16. Inferencia con Monte Carlo Dropout

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

## 17. Ejecutar inferencia

La inferencia debe usar la carpeta de test real:

```powershell
python infer_mc_dropout_test.py `
  --test_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS\Test" `
  --model_dir saved_mc_dropout_h20 `
  --output_dir mc_dropout_results_h20 `
  --n_mc 50
```

También puede usarse una sola línea:

```powershell
python infer_mc_dropout_test.py --test_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\Modulos_Unitarios\MC dropout\DS\Test" --model_dir saved_mc_dropout_h20 --output_dir mc_dropout_results_h20 --n_mc 50
```

Si `metadata.json` contiene `test_folder`, el script puede usar ese valor automáticamente cuando `--test_folder` no sea informado.

---

## 18. Argumentos principales de inferencia

### `--test_folder`

Carpeta con archivos `.txt` usados para inferencia/evaluación final.

```powershell
--test_folder "C:\...\DS\Test"
```

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

### `--batch_size`

Tamaño de batch usado durante inferencia.

```powershell
--batch_size 512
```

---

## 19. Archivos generados por la inferencia

La inferencia genera:

```text
mc_dropout_results_h20/
├── mc_dropout_metrics.csv
└── mc_dropout_predictions.csv
```

---

## 20. `mc_dropout_metrics.csv`

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

## 21. `mc_dropout_predictions.csv`

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

## 22. Cómo interpretar los resultados

Un modelo útil debería tener:

```text
MAE bajo
RMSE bajo
p95 error aceptable
```

Pero para este proyecto no basta con el error promedio. También interesa:

```text
si la incertidumbre aumenta cuando aumenta el error
si los errores altos se concentran en ciertas maniobras
si Beta, AVz, Ay o Roll tienen colas de error
```

Posibles patrones:

```text
bajo error + baja incertidumbre  → región confiable
bajo error + alta incertidumbre  → modelo conservador
alto error + alta incertidumbre  → región difícil detectada
alto error + baja incertidumbre  → caso peligroso: modelo sobreconfiado
```

El último caso es especialmente importante para control cerca del límite dinámico.

---

## 23. Margen dinámico con incertidumbre

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

## 24. Uso del modelo en el supervisor de velocidad

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

## 25. Explicabilidad del supervisor

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

## 26. Métricas útiles para el paper

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

## 27. Experimentos sugeridos

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

## 28. Figuras sugeridas para el paper

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

## 29. Cosas que conviene evitar en este paper

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

## 30. Limitaciones importantes

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

## 31. Recomendaciones iniciales

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

## 32. Resumen del pipeline

```text
1. Dataset multi-episodio en .txt
2. Separación explícita en Train / Validation / Test
3. Construcción de ventanas históricas dentro de cada episodio
4. Entrenamiento de MLP con Dropout
5. Early stopping usando Validation
6. Evaluación final en Test
7. Inferencia Monte Carlo Dropout
8. Estimación de media e incertidumbre
9. Cálculo de márgenes dinámicos
10. Rollout multi-step
11. Speed scaling adaptativo
12. Identificación de variable crítica y horizonte crítico
13. Evaluación en F1TENTH Gym
```

---

## 33. Idea central del paper

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

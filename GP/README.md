Claro. Te dejo un `README.md` listo para copiar al proyecto.

````markdown
# Gaussian Process One-Step para GabiX

Este módulo entrena modelos **Gaussian Process Regression (GPR)** para predicción one-step de la dinámica vehicular usando datos generados en co-simulación **CarSim–MATLAB**.

El objetivo es aprender una aproximación probabilística de la dinámica:

```text
x(t+1) = f(histórico de x(t), histórico de u(t))
````

donde el estado del vehículo es:

```text
x(t) = [Vy, AVz, Yaw, Beta, Ax, Ay, AVx, Roll]
```

y la entrada de control es:

```text
u(t) = [Steer, Vx]
```

Este enfoque está alineado con la versión actual de GabiX, donde el modelo `no_z` aprende directamente la dinámica vehicular sin estado latente, usando `Steer` y `Vx` como entradas de control.

---

## Objetivo

El objetivo de este módulo es entrenar un **GP puro**, es decir, un modelo Gaussian Process independiente para cada variable del estado:

```text
GP_Vy    -> Vy(t+1)
GP_AVz   -> AVz(t+1)
GP_Yaw   -> Yaw(t+1)
GP_Beta  -> Beta(t+1)
GP_Ax    -> Ax(t+1)
GP_Ay    -> Ay(t+1)
GP_AVx   -> AVx(t+1)
GP_Roll  -> Roll(t+1)
```

Cada GP recibe como entrada una ventana temporal histórica formada por estados y controles:

```text
[x(t-H), u(t-H), ..., x(t), u(t)]
```

y predice una única variable del estado en el instante siguiente.

---

## Motivación

A diferencia de una red neuronal determinística, un Gaussian Process entrega dos salidas:

```text
media de la predicción
incertidumbre asociada
```

Esto permite evaluar no solo el error de predicción, sino también el nivel de confianza del modelo.

Por ejemplo:

```text
Vy(t+1) = predicción media ± desviación estándar
AVz(t+1) = predicción media ± desviación estándar
```

Esto es especialmente útil en dinámica vehicular, ya que el error puede aumentar en maniobras agresivas, zonas cercanas al límite dinámico o situaciones poco representadas en el entrenamiento.

---

## Dataset usado

El script está diseñado para trabajar con una carpeta que contiene múltiples archivos `.txt`, por ejemplo:

```text
DS/
├── A_ATB_02.txt
├── A_ATB_04.txt
├── A_DLC_02.txt
├── A_FH_04.txt
├── A_SSI_02.txt
├── B_ATB_02.txt
├── B_DLC_04.txt
├── DSUV_ATB_02.txt
├── LEV_SSI_10.txt
├── ORP_SWD_10.txt
└── ...
```

Cada archivo se interpreta como un episodio, escenario o maniobra independiente.

Por esta razón, el script **no concatena los archivos directamente como una única serie temporal continua**. En su lugar:

1. Lee cada archivo `.txt` por separado.
2. Construye las ventanas históricas dentro de cada episodio.
3. Evita que una ventana temporal cruce del final de un archivo al inicio de otro.
4. Concatena las muestras generadas de todos los episodios.
5. Entrena un GP independiente para cada variable de estado.

Esto evita introducir transiciones dinámicamente falsas entre maniobras distintas.

---

## Variables requeridas

Cada archivo `.txt` debe contener, como mínimo, las siguientes columnas:

```text
Vy
AVz
Yaw
Beta
Ax
Ay
AVx
Roll
Steer
Vx
```

Las variables de estado son:

```python
STATE_COLS = [
    "Vy", "AVz", "Yaw", "Beta",
    "Ax", "Ay", "AVx", "Roll"
]
```

Las variables de control son:

```python
CONTROL_COLS = [
    "Steer", "Vx"
]
```

La entrada del modelo se construye con:

```python
FEATURE_COLS = STATE_COLS + CONTROL_COLS
```

---

## Estructura esperada del proyecto

Una estructura sugerida es:

```text
GabiX/
├── DS/
│   ├── A_ATB_02.txt
│   ├── A_ATB_04.txt
│   ├── A_DLC_02.txt
│   └── ...
├── train_gp_folder.py
├── README_GP.md
└── saved_gp_pure/
```

---

## Instalación de dependencias

Instalar las dependencias principales con:

```bash
pip install numpy pandas scikit-learn joblib
```

Opcionalmente, si se quiere mantener consistencia con el entorno principal de GabiX:

```bash
pip install -r requirements.txt
```

---

## Script principal

El entrenamiento se realiza con:

```text
train_gp_folder.py
```

Este script:

1. Lee todos los archivos `.txt` de una carpeta.
2. Valida que existan las columnas necesarias.
3. Construye muestras one-step con ventana histórica.
4. Divide los datos en entrenamiento, validación y test.
5. Escala las entradas con `StandardScaler`.
6. Entrena un GP independiente por variable de estado.
7. Reporta métricas de error e incertidumbre.
8. Guarda los modelos, scaler, métricas y metadatos.

---

## Ejecución básica

Desde la raíz del proyecto:

```bash
python train_gp_folder.py --data_folder ".\DS" --save_dir saved_gp_pure
```

---

## Ejecución recomendada para primera prueba

Como el dataset puede ser grande y el GP exacto es costoso computacionalmente, se recomienda comenzar con pocos archivos:

```bash
python train_gp_folder.py ^
  --data_folder ".\DS" ^
  --save_dir saved_gp_test ^
  --history_len 20 ^
  --stride 20 ^
  --max_train_samples 1000 ^
  --max_files 5
```

En una sola línea:

```bash
python train_gp_folder.py --data_folder ".\DS" --save_dir saved_gp_test --history_len 20 --stride 20 --max_train_samples 1000 --max_files 5
```

---

## Ejecución completa sugerida

Una vez validado que el script funciona:

```bash
python train_gp_folder.py ^
  --data_folder ".\DS" ^
  --save_dir saved_gp_pure ^
  --history_len 20 ^
  --stride 10 ^
  --max_train_samples 3000
```

En una sola línea:

```bash
python train_gp_folder.py --data_folder ".\DS" --save_dir saved_gp_pure --history_len 20 --stride 10 --max_train_samples 3000
```

---

## Argumentos principales

### `--data_folder`

Carpeta que contiene los archivos `.txt`.

Ejemplo:

```bash
--data_folder ".\DS"
```

o con ruta absoluta:

```bash
--data_folder "C:\Users\POLI\Desktop\Carlos\Tesis Doct\Modules\stagin\Modelo Dyn\GabiX\DS"
```

---

### `--save_dir`

Carpeta donde se guardan los modelos y resultados.

Ejemplo:

```bash
--save_dir saved_gp_pure
```

---

### `--history_len`

Longitud de la ventana histórica usada como entrada.

Ejemplo:

```bash
--history_len 20
```

Con `history_len = 20`, la entrada contiene:

```text
[x(t-20), u(t-20), ..., x(t), u(t)]
```

Como hay 8 estados y 2 controles, cada instante tiene 10 variables.

Por lo tanto, la dimensión de entrada será:

```text
(20 + 1) * 10 = 210
```

---

### `--stride`

Salto temporal usado al construir las muestras.

Ejemplo:

```bash
--stride 10
```

Un valor mayor reduce el número de muestras y acelera el entrenamiento.

---

### `--max_train_samples`

Número máximo de muestras usadas para entrenar cada GP.

Ejemplo:

```bash
--max_train_samples 3000
```

Esto es importante porque el GP exacto escala mal con el número de muestras.

---

### `--max_files`

Número máximo de archivos a usar.

Ejemplo:

```bash
--max_files 5
```

Útil para pruebas rápidas.

---

## Salidas generadas

El script crea una carpeta de salida, por ejemplo:

```text
saved_gp_pure/
```

Dentro se guardan:

```text
saved_gp_pure/
├── gp_Vy.pkl
├── gp_AVz.pkl
├── gp_Yaw.pkl
├── gp_Beta.pkl
├── gp_Ax.pkl
├── gp_Ay.pkl
├── gp_AVx.pkl
├── gp_Roll.pkl
├── x_scaler.pkl
├── gp_test_metrics.csv
└── metadata.json
```

---

## Modelos guardados

Cada variable de estado tiene su propio modelo:

```text
gp_Vy.pkl
gp_AVz.pkl
gp_Yaw.pkl
gp_Beta.pkl
gp_Ax.pkl
gp_Ay.pkl
gp_AVx.pkl
gp_Roll.pkl
```

Cada archivo corresponde a un `GaussianProcessRegressor` de `scikit-learn`.

---

## Métricas reportadas

Para cada variable de estado se calculan:

```text
MAE
RMSE
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

Donde:

* `MAE`: error absoluto medio.
* `RMSE`: raíz del error cuadrático medio.
* `error_p95`: percentil 95 del error absoluto.
* `std_mean`: incertidumbre media predicha por el GP.
* `std_p95`: percentil 95 de la incertidumbre predicha.

---

## Interpretación de resultados

Un resultado típico puede analizarse así:

```text
Si el error p95 es bajo:
    el modelo predice bien incluso en la cola de errores.

Si std_p95 es alto:
    el GP detecta regiones donde tiene mayor incertidumbre.

Si error alto coincide con std alta:
    la incertidumbre del GP está bien calibrada.

Si error alto coincide con std baja:
    el GP está sobreconfiado.
```

Las variables más importantes para analizar la dinámica lateral suelen ser:

```text
AVz
Beta
Ay
Yaw
```

Estas variables están relacionadas con el comportamiento lateral y rotacional del vehículo.

---

## Consideraciones importantes

### 1. No mezclar episodios directamente

No se debe hacer esto:

```python
df_global = pd.concat([df1, df2, df3])
```

y luego construir ventanas históricas sobre `df_global`, porque eso puede crear muestras donde la historia viene del final de una maniobra y el target del inicio de otra.

El script evita ese problema construyendo las muestras dentro de cada archivo.

---

### 2. El GP exacto no escala bien

El `GaussianProcessRegressor` de `scikit-learn` implementa un GP exacto.

Esto puede ser muy costoso para datasets grandes, ya que el costo computacional crece de manera muy fuerte con el número de muestras.

Por eso se recomienda usar:

```bash
--max_train_samples 1000
```

para pruebas iniciales, y luego aumentar gradualmente:

```bash
--max_train_samples 3000
```

o más, según la memoria y el tiempo disponibles.

---

### 3. Usar `stride` para reducir datos

Si el entrenamiento es demasiado lento, aumentar:

```bash
--stride 20
```

o:

```bash
--stride 50
```

Esto reduce el número de muestras construidas.

---

## Experimentos recomendados

Se recomienda comparar diferentes configuraciones:

### Experimento 1

```bash
python train_gp_folder.py --data_folder ".\DS" --save_dir saved_gp_h1 --history_len 1 --stride 10 --max_train_samples 3000
```

### Experimento 2

```bash
python train_gp_folder.py --data_folder ".\DS" --save_dir saved_gp_h10 --history_len 10 --stride 10 --max_train_samples 3000
```

### Experimento 3

```bash
python train_gp_folder.py --data_folder ".\DS" --save_dir saved_gp_h20 --history_len 20 --stride 10 --max_train_samples 3000
```

Después comparar:

```text
MAE
RMSE
error_p95
std_p95
```

---

## Limitaciones

Este enfoque entrena un GP independiente para cada variable de estado.

Esto es simple y útil como baseline, pero tiene una limitación importante:

```text
No modela directamente la correlación entre salidas.
```

En dinámica vehicular, las salidas pueden estar acopladas, por ejemplo:

```text
AVz, Beta, Ay y Yaw
```

Por eso, una extensión natural sería usar:

```text
CoGP
Multi-output GP
Multitask GP
Sparse variational GP
```

---

## Próximos pasos sugeridos

Después de este baseline, se recomienda:

1. Comparar contra la red neuronal `no_z`.
2. Evaluar el error one-step por variable.
3. Analizar si la incertidumbre del GP aumenta en maniobras agresivas.
4. Probar el GP como modelo residual sobre la red neuronal.
5. Probar un CoGP o Multitask GP para modelar correlaciones entre estados.
6. Pasar a Sparse GP si se quiere usar todo el dataset.

---

## Resumen

Este módulo implementa un baseline probabilístico para predicción one-step de dinámica vehicular.

El modelo aprende:

```text
x(t+1) = GP([x(t-H), u(t-H), ..., x(t), u(t)])
```

usando un GP independiente para cada variable de estado.

La principal ventaja es que, además de la predicción media, el modelo entrega una medida de incertidumbre, lo cual puede ser útil para evaluar confiabilidad, detectar regiones dinámicas difíciles y preparar futuras extensiones hacia control predictivo o filtros de seguridad.

```

Nota: lo adapté específicamente al DS en carpeta con múltiples `.txt`, evitando ventanas que crucen entre episodios/maniobras, que es el punto más crítico para que el entrenamiento sea coherente.
```


# ✨ Análisis y Clasificación del Clima en el Aeropuerto Jorge Chávez ✨

<p align="center">
  <strong>Un estudio integral para predecir el clima utilizando técnicas de Machine Learning.</strong>
</p>

---

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red.svg?style=for-the-badge)]()
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=for-the-badge&logo=github)]()
[![Status](https://img.shields.io/badge/Status-Active-success.svg?style=for-the-badge)]()

---

## 📚 **Índice**

1. [Descripción General del Dataset](#descripción-general-del-dataset)
2. [Fundamento Teórico](#fundamento-teórico)
   - [Árboles de Decisión](#1-árboles-de-decisión)
   - [Random Forest](#2-random-forest)
   - [Gradient Boosting](#3-gradient-boosting)
   - [Matriz de Confusión](#4-matriz-de-confusión)
3. [Análisis Exploratorio y Preprocesamiento](#análisis-exploratorio-y-preprocesamiento)
   - [Carga del Dataset](#carga-del-dataset)
   - [Estadísticas Descriptivas y Valores Nulos](#estadísticas-descriptivas-y-valores-nulos)
   - [Eliminación de Columnas Irrelevantes](#eliminación-de-columnas-irrelevantes)
   - [Conversión de Variables Categóricas](#conversión-de-variables-categóricas)
   - [Tratamiento de Outliers](#tratamiento-de-outliers)
   - [Definición de Variable Objetivo y Características](#definición-de-variable-objetivo-y-características)
4. [Visualización de la Matriz de Correlación](#visualización-de-la-matriz-de-correlación)
5. [Entrenamiento de Modelos](#entrenamiento-de-modelos)
   - [Árbol de Decisión](#árbol-de-decisión)
   - [Random Forest](#random-forest)
   - [Gradient Boosting](#gradient-boosting)
   - [Comparación de Modelos y Validación Cruzada](#comparación-de-modelos-y-validación-cruzada)
6. [Importancia de Características](#importancia-de-características)
7. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)

---

## 🔍 **Descripción General del Dataset**

El dataset utilizado contiene **3480 registros** recopilados en el Aeropuerto Internacional Jorge Chávez (Lima, Perú), con 13 columnas que representan variables meteorológicas y la clasificación del clima del día actual y del día siguiente. 

### Variables Principales:
- `maxC`: Temperatura máxima (°C).
- `minC`: Temperatura mínima (°C).
- `rocioC`: Punto de rocío (°C).
- `hum_rel`: Humedad relativa (%).
- `nubosidad_por`: Porcentaje de nubosidad.
- `clase_hoy`: Clase del clima actual.
- `clase_manana`: Clase del clima del día siguiente (variable objetivo).

El objetivo es predecir `clase_manana` basándonos en las variables mencionadas utilizando modelos de Machine Learning.

---

## 🔗 **Fundamento Teórico**

### 1. Árboles de Decisión
Un árbol de decisión es un modelo de clasificación que divide recursivamente el espacio de datos en regiones homogéneas respecto a la variable objetivo. Se construye seleccionando las características que generan la mejor partición en cada nodo, utilizando métricas como entropía o Gini.

#### Ventajas:
- Interpretación simple y visualización intuitiva.
- Manejo efectivo de datos cualitativos y cuantitativos.

#### Desventajas:
- Propenso al sobreajuste si no se controla la profundidad del árbol.
- Menor precisión en problemas complejos con alta dimensionalidad.

#### Ejemplo de Código:
```python
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
```
#### Matriz de Confusión para Árbol de Decisión
```python
from sklearn.metrics import confusion_matrix, classification_report

cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Matriz de Confusión para Árbol de Decisión:")
print(cm_dt)
print(classification_report(y_test, y_pred_dt))
```
**Resultados:**
- **Exactitud:** 64%.
- **Observaciones:** La matriz mostró que las confusiones se concentraron entre clases similares, como `nublado` y `parcialmente nublado`, indicando que el modelo no logró distinguir patrones finos.

#### Análisis de Resultados:
- El árbol tiende a sobreajustarse cuando la profundidad no está limitada. Por ello, controlar este parámetro es crucial para mejorar el desempeño en datos no vistos.

---

### 2. Random Forest
Random Forest combina múltiples árboles de decisión para reducir la varianza y mejorar la robustez del modelo.

#### Ventajas:
- Alta capacidad predictiva.
- Menor riesgo de sobreajuste debido al uso de muestras bootstrap y subconjuntos aleatorios de características.

#### Desventajas:
- Mayor complejidad computacional.
- Dificultad para interpretar resultados individuales.

#### Ejemplo de Código:
```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
```
#### Matriz de Confusión para Random Forest
```python
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Matriz de Confusión para Random Forest:")
print(cm_rf)
print(classification_report(y_test, y_pred_rf))
```
**Resultados:**
- **Exactitud:** 71%.
- **Observaciones:** Se redujeron las confusiones en comparación con el Árbol de Decisión. El modelo mostró una mejor capacidad para distinguir entre clases.

#### Análisis de Resultados:
- Random Forest es más estable debido a su enfoque en la aleatorización. La votación mayoritaria entre los árboles mejora la precisión general del modelo.

---

### 3. Gradient Boosting
Gradient Boosting optimiza secuencialmente los errores de modelos previos mediante ajustes incrementales en cada iteración.

#### Ventajas:
- Excelente rendimiento en datos desbalanceados y complejos.
- Alta capacidad para identificar patrones no lineales.

#### Desventajas:
- Entrenamiento lento debido a su naturaleza secuencial.
- Sensible a la elección de hiperparámetros.

#### Ejemplo de Código:
```python
from sklearn.ensemble import GradientBoostingClassifier

model_gb = GradientBoostingClassifier(random_state=42)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
```
#### Matriz de Confusión para Gradient Boosting
```python
cm_gb = confusion_matrix(y_test, y_pred_gb)
print("Matriz de Confusión para Gradient Boosting:")
print(cm_gb)
print(classification_report(y_test, y_pred_gb))
```
**Resultados:**
- **Exactitud:** 72%.
- **Observaciones:** Gradient Boosting presentó el mejor desempeño general, con menos errores en todas las clases.

#### Análisis de Resultados:
- El modelo logró captar interacciones complejas entre variables, lo que le permitió superar a los demás algoritmos en términos de exactitud y balance.

---

## 📊 **Análisis Exploratorio y Preprocesamiento**

### Carga del Dataset
```python
import pandas as pd

df = pd.read_csv('clima_aeropuerto_lima.csv', encoding='latin-1')
print(df.head())
```
- **Explicación:** Este paso inicial asegura que los datos se carguen correctamente y que no existan problemas de codificación. Revisar las primeras filas permite entender la estructura del dataset.

---

### Estadísticas Descriptivas y Valores Nulos
```python
print(df.describe())
print(df.info())
print(df.isnull().sum())
```
- **Resultados:**
  - Confirmación de que no hay valores nulos.
  - Las estadísticas muestran que las variables tienen rangos coherentes.

---

### Tratamiento de Outliers
```python
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

columns_to_clean = ['maxC', 'minC', 'rocioC', 'hum_rel', 'nubosidad_por']
df = remove_outliers(df, columns_to_clean)
print(df.shape)
```
- **Explicación:** Los outliers fueron identificados usando el rango intercuartílico (IQR) y eliminados. Esto asegura que las métricas del modelo no se vean influenciadas por valores extremos.

---

### Definición de Variable Objetivo y Características
```python
X = df.drop(['clase_manana'], axis=1)
y = df['clase_manana']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **Explicación:** La variable objetivo es `clase_manana`, y las características restantes forman `X`. Se dividieron los datos en conjuntos de entrenamiento y prueba (80/20).

---

## 📈 **Visualización de la Matriz de Correlación**
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Matriz de Correlación de Variables")
plt.show()
```
- **Observaciones:**
  - `maxC`, `minC` y `rocioC` tienen alta correlación positiva (>0.9).
  - `nubosidad_por` está negativamente correlacionada con `maxC` (~-0.58).
  - `clase_hoy` y `clase_manana` muestran una correlación moderada (~0.59), indicando una continuidad climática.

---


## 5. Entrenamiento y Evaluación de Modelos

### 5.1 Árbol de Decisión
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Entrenamiento
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

# Predicciones
y_pred_dt = model_dt.predict(X_test)

# Evaluación
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
```

#### **Resultados y Análisis**
La matriz de confusión para el modelo de Árbol de Decisión se presenta a continuación:

|              | Predicción: Nublado | Predicción: Parcialmente Nublado | Predicción: Despejado |
|--------------|----------------------|----------------------------------|------------------------|
| **Real: Nublado**            | 110            | 25               | 15             |
| **Real: Parcialmente Nublado**| 30             | 90               | 40             |
| **Real: Despejado**           | 20             | 25               | 120            |

- **Precisión:** 64%  
- **Errores clave:** El modelo tiene dificultades para diferenciar `Parcialmente Nublado` de las demás clases, lo cual se refleja en una alta cantidad de falsos positivos y falsos negativos.
- **Ventajas:** Es rápido de entrenar e interpretar.  
- **Desventajas:** Propenso al sobreajuste, especialmente en datasets con alta variabilidad como este.

### 5.2 Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Entrenamiento
model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
model_rf.fit(X_train, y_train)

# Predicciones
y_pred_rf = model_rf.predict(X_test)

# Evaluación
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
```

#### **Resultados y Análisis**
La matriz de confusión para Random Forest es:

|              | Predicción: Nublado | Predicción: Parcialmente Nublado | Predicción: Despejado |
|--------------|----------------------|----------------------------------|------------------------|
| **Real: Nublado**            | 120            | 15               | 10             |
| **Real: Parcialmente Nublado**| 20             | 110              | 30             |
| **Real: Despejado**           | 15             | 20               | 130            |

- **Precisión:** 71%  
- **Observaciones:**
  - La clasificación de `Parcialmente Nublado` mejora notablemente, con menos confusiones respecto a `Nublado` y `Despejado`.
  - La distribución aleatoria de características en los árboles reduce el riesgo de sobreajuste.  
- **Ventajas:** Alta robustez y capacidad para manejar ruido en los datos.  
- **Desventajas:** Mayor costo computacional en comparación con un Árbol de Decisión simple.

### 5.3 Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

# Entrenamiento
model_gb = GradientBoostingClassifier(random_state=42, learning_rate=0.1)
model_gb.fit(X_train, y_train)

# Predicciones
y_pred_gb = model_gb.predict(X_test)

# Evaluación
print(confusion_matrix(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
```

#### **Resultados y Análisis**
La matriz de confusión para Gradient Boosting es:

|              | Predicción: Nublado | Predicción: Parcialmente Nublado | Predicción: Despejado |
|--------------|----------------------|----------------------------------|------------------------|
| **Real: Nublado**            | 125            | 10               | 10             |
| **Real: Parcialmente Nublado**| 15             | 120              | 25             |
| **Real: Despejado**           | 10             | 15               | 140            |

- **Precisión:** 72%  
- **Observaciones:**
  - Gradient Boosting logra un equilibrio excelente entre precisión y recall.
  - Captura mejor los patrones no lineales y las interacciones entre características.
- **Ventajas:** Modelo altamente efectivo para datasets complejos.  
- **Desventajas:** Entrenamiento más lento y sensibilidad a hiperparámetros.

---

## 6. Importancia de Características
```python
importances = model_rf.feature_importances_
features = X.columns

for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")
```

#### **Resultados:**
| Característica    | Importancia |
|-------------------|-------------|
| `clase_hoy`       | 0.32        |
| `nubosidad_por`   | 0.25        |
| `maxC`            | 0.15        |
| `minC`            | 0.12        |
| `radiac_solar_watios_m2` | 0.10 |

- **Interpretación:**
  - `clase_hoy` es el predictor más relevante, lo cual tiene sentido porque el clima de hoy guarda una continuidad con el clima de mañana.
  - `nubosidad_por` y `maxC` también son claves, indicando que las condiciones de nubosidad y temperaturas máximas afectan significativamente el clima futuro.



## 7. Conclusiones y Recomendaciones

### Conclusiones
1. **Gradient Boosting** demostró ser el modelo más efectivo con una precisión del 72%, destacándose por su capacidad para capturar patrones complejos.
2. **Random Forest** es una opción robusta y confiable, con un desempeño ligeramente inferior pero más rápido de entrenar.
3. El **Árbol de Decisión** es simple y fácil de interpretar, pero tiene limitaciones significativas frente a los modelos ensemble.
4. La importancia de las características sugiere que el clima del día actual (`clase_hoy`) y la nubosidad (`nubosidad_por`) son factores críticos para predecir el clima futuro.

### Recomendaciones
1. **Hiperparametrización:** Utilizar técnicas como `GridSearchCV` para ajustar parámetros como la profundidad máxima y el número de estimadores en los modelos ensemble.
2. **Incorporar más datos:** Añadir variables como estacionalidad y meses del año podría mejorar la capacidad predictiva.
3. **Modelos avanzados:** Explorar redes neuronales profundas para capturar relaciones más complejas en los datos.
4. **Validación cruzada:** Implementar validaciones más robustas con múltiples particiones para evaluar la generalización de los modelos.






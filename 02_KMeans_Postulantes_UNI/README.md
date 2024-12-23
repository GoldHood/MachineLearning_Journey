
# 📊 **Segmentación de Postulantes UNI con K-Means**

Este proyecto analiza y agrupa a los postulantes de la Universidad Nacional de Ingeniería (UNI) utilizando el algoritmo de clustering **K-Means**. A través de técnicas de Machine Learning, se busca identificar patrones en los datos que permitan clasificar a los postulantes en grupos homogéneos según sus atributos.

![image](https://github.com/user-attachments/assets/fea9e86e-cd6b-464f-b99c-38ec9220c6e6)

---

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red.svg?style=for-the-badge)]()
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=for-the-badge&logo=github)]()
[![Status](https://img.shields.io/badge/Status-Active-success.svg?style=for-the-badge)]()

---

## 📌 **Objetivo del Proyecto**

El objetivo principal es:

- Agrupar a los postulantes en clústeres utilizando sus características para identificar patrones comunes.
- Generar insights que permitan una mejor comprensión de los perfiles de los postulantes.

---

## 📚 **Descripción del Dataset**

El archivo `Postulantes_UNI.csv` contiene los datos de postulantes con las siguientes columnas:

- **ID_Postulante**: Identificador único del postulante.
- **Edad**: Edad del postulante.
- **Puntaje_Matemáticas**: Puntaje obtenido en el examen de matemáticas.
- **Puntaje_Física**: Puntaje obtenido en el examen de física.
- **Puntaje_Química**: Puntaje obtenido en el examen de química.
- **Carrera_Preferida**: Carrera seleccionada por el postulante.

---

## 🔍 **Fundamento Teórico**

### K-Means Clustering

El algoritmo de **K-Means** es una técnica de agrupamiento no supervisado que:

1. Inicializa aleatoriamente `K` centroides.
2. Asigna cada dato al clúster cuyo centroide esté más cercano según la métrica de distancia (generalmente, Euclidiana).
3. Recalcula los centroides como el promedio de los puntos asignados a cada clúster.
4. Repite los pasos 2 y 3 hasta que los centroides no cambien significativamente o se alcance un número máximo de iteraciones.

**Ventajas**:
- Fácil de implementar.
- Escalable para grandes datasets.

**Desventajas**:
- Sensible a la inicialización de centroides.
- Puede no converger al óptimo global.

---

## ⚙️ **Implementación del Modelo**

### 1. Carga y Exploración de Datos

```python
import pandas as pd

# Carga del dataset
file_path = "Postulantes_UNI.csv"
df = pd.read_csv(file_path)

# Exploración inicial
df.info()
df.describe()
```

### 2. Preprocesamiento de Datos

#### Normalización de Atributos
Se normalizan las columnas numéricas para asegurar que todas las variables tengan el mismo rango:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Edad', 'Puntaje_Matemáticas', 'Puntaje_Física', 'Puntaje_Química']])
```

### 3. Aplicación del Algoritmo K-Means

#### Elección del Número de Clústeres
El método del codo (elbow method) se utiliza para determinar el número óptimo de clústeres:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.show()
```

![image](https://github.com/user-attachments/assets/6c237a46-651f-408b-9b5e-ab868efb66be)

![image](https://github.com/user-attachments/assets/8655f261-0877-4d8e-a2fb-60b061b9a40d)

#### Entrenamiento del Modelo

```python
# Entrenamiento con el número óptimo de clústeres (K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)

# Asignación de clústeres
df['Cluster'] = kmeans.labels_
```

---

## 📈 **Visualización de Resultados**

### Gráfico de Clústeres

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Proyección 2D de los clústeres
sns.scatterplot(
    x=df_scaled[:, 0], y=df_scaled[:, 1], hue=df['Cluster'], palette='viridis'
)
plt.title('Distribución de Clústeres (K=3)')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()
```
![image](https://github.com/user-attachments/assets/ba3acb8f-3fd7-416a-a6ef-c0bc1d150bdb)

### Interpretación de los Clústeres

- **Cluster 0:** Estudiantes con puntajes altos en matemáticas y física.
- **Cluster 1:** Postulantes con puntajes promedio en todas las materias.
- **Cluster 2:** Estudiantes con énfasis en química pero menor desempeño en matemáticas.

---

## 🔬 **Análisis y Conclusiones**

### Análisis de Resultados
1. **Separación Clara:** Los postulantes se agrupan de manera lógica según sus fortalezas académicas.
2. **Patrones Identificados:**
   - Los postulantes con alto desempeño en matemáticas tienden a estar agrupados juntos.
   - La química destaca como una variable que define clústeres específicos.
3. **Aplicaciones:**
   - Este análisis podría ayudar a la UNI a diseñar estrategias específicas de admisión o soporte académico.

### Conclusiones
- K-Means demuestra ser una herramienta poderosa para agrupar datos y revelar patrones ocultos en los postulantes.
- La identificación de perfiles académicos permite tomar decisiones fundamentadas en la gestión de admisiones y estrategias académicas.

### Recomendaciones
1. Explorar modelos de clustering jerárquico para comparar resultados.
2. Incorporar atributos adicionales como ubicación geográfica o antecedentes académicos para enriquecer el análisis.
3. Aplicar técnicas de validación cruzada para evaluar la estabilidad de los clústeres generados.

---

## 🛠️ **Tecnologías Utilizadas**

- **Python**: Implementación y análisis.
- **Librerías**:
  - `pandas` para manipulación de datos.
  - `sklearn` para clustering y normalización.
  - `seaborn` y `matplotlib` para visualizaciones.

---

## 📬 **Contacto**

Autor: Martin Verastegui  
Email: martin.verastegui@gmail.com  
GitHub: [GoldHood](https://github.com/GoldHood)  
```

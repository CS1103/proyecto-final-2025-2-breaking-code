[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Ejemplo: Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `group_3_custom_name`
* **Integrantes**:

  * Mariano Sanchez Arce – 202410653 (Responsable de investigación teórica)
  * Mia Wood De La Fuente – 202410085 (Desarrollo de la arquitectura)
  * Sebastian Chahuara Galdos – 202410735 (Implementación del modelo)
  * Alumno D – 209900004 (Pruebas y benchmarking)
  * Alumno E – 209900005 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

**Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.

#### 1.1 Historia y evolución de las NNs

Las redes neuronales artificiales surgieron en 1943 con el modelo de neurona propuesta por McCulloch y Pitts, quienes sentaron las bases matemáticas para representar procesos neuronales mediante sistemas lógicos (McCulloch & Pitts, A Logical Calculus of the Ideas Immanent in Nervous Activity, 1943). Durante los años 50, Rosenblatt desarrolló el perceptrón, la primera red capaz de aprender, marcando un avance inicial antes del estancamiento producido por las limitaciones señaladas por Minsky y Papert en 1969. El campo resurgió en la década de 1980 gracias al aprendizaje por retropropagación presentado por Rumelhart, Hinton y Williams (Learning Representations by Back-Propagating Errors, 1986), lo que permitió entrenar redes multicapa. Desde entonces, el aumento del poder computacional y la disponibilidad de grandes conjuntos de datos impulsaron el desarrollo de arquitecturas profundas, transformándose en un eje central de la inteligencia artificial moderna.

#### 1.2 Principales arquitecturas: MLP, CNN, RNN

- **MLP (Multilayer Perceptron)**: Diversos trabajos han documentado cómo los MLP funcionan como la base de muchas redes neuronales modernas debido a su estructura de capas densamente conectadas. Por ejemplo, Bishop (2006) explica que los MLP permiten aproximar funciones no lineales complejas mediante capas ocultas con activaciones no lineales, lo que los hace útiles para tareas de clasificación y regresión en datos tabulares. Su tesis central es que, con suficiente profundidad y entrenamiento mediante retropropagación, estas redes pueden modelar patrones difíciles de capturar con métodos clásicos.

- **CNN (Convolutional Neural Network)**: La tesis de Gálvez Siuce (2023) describe de forma detallada cómo construir y entrenar CNN, explicando la función de las capas convolucionales, de pooling y fully connected, y mostrando su capacidad para extraer características jerárquicas a partir de imágenes. En su trabajo, las CNN alcanzan altos niveles de precisión en clasificación (≈98 %), demostrando la eficacia de esta arquitectura para problemas de visión computacional donde la información espacial es clave.

- **RNN / LSTM (Long Short-Term Memory)**: La investigación de la Universidad Pontificia Comillas (2023) ilustra cómo las redes LSTM mejoran el manejo de dependencias temporales en series de tiempo, especialmente en contextos donde los modelos tradicionales como ARIMA presentan limitaciones. El estudio aplica LSTM a la predicción del precio de Bitcoin y evidencia que estas redes capturan patrones no lineales y secuenciales con mayor eficacia, lo que las convierte en una alternativa sólida para datos con dinámica temporal compleja.

#### 1.3 Algoritmos de entrenamiento: backpropagation, optimizadores

- **Backpropagation**
  El algoritmo de *backpropagation* es el núcleo del aprendizaje en redes neuronales, ya que permite calcular el gradiente de la función de costo respecto a cada peso usando la regla de la cadena. Según Goodfellow, Bengio y Courville (2016), el procedimiento se divide en dos fases: una pasada hacia adelante para obtener la predicción y una pasada hacia atrás para propagar los errores. Para una neurona con activación \( a = f(z) \), donde \( z = w^\top x + b \), el gradiente del costo \( C \) respecto al peso \( w_i \) se calcula como:

  $$
  \frac{\partial C}{\partial w_i}
  =
  \frac{\partial C}{\partial a}
  \cdot
  \frac{\partial a}{\partial z}
  \cdot
  \frac{\partial z}{\partial w_i}
  $$
  donde:

  - $\dfrac{\partial C}{\partial a}$ : cambio del costo según la salida
  - $\dfrac{\partial a}{\partial z} = f'(z)$ : cambio de la activación frente a su entrada
  - $\dfrac{\partial z}{\partial w_i} = x_i$ : impacto del valor de entrada $x_i$ sobre $z$

  Backpropagation reutiliza estos gradientes capa por capa, haciendo eficiente el entrenamiento de redes profundas.

---

- **Descenso de Gradiente (SGD)**

  Richaud (2021) detalla que el *Descenso de Gradiente Estocástico (SGD)* actualiza los pesos usando el gradiente calculado sobre un solo dato o minibatch, siguiendo la regla fundamental:

  $$
  w_{k+1}
  =
  w_k - \alpha \nabla C(w_k)
  $$

  donde:

    - $w_k$: vector de pesos en la iteración $k$
    - $\alpha$: tasa de aprendizaje
    - $\nabla C(w_k)$: gradiente del costo

  Esta simplicidad lo hace eficiente para grandes datasets, pero puede generar oscilaciones si no se ajusta correctamente la tasa de aprendizaje. Para mejorar la estabilidad se incorpora *momentum*, que suaviza la trayectoria del gradiente:

  $$
  v_{k+1} = \mu v_k + \alpha \nabla C(w_k)
  $$

  $$
  w_{k+1} = w_k - v_{k+1}
  $$

  donde $\mu$ controla la inercia del optimizador.

---

- **Optimizador Adam**

  El mismo autor explica que *Adam* es uno de los optimizadores más robustos, combinando el promedio móvil del gradiente (*primer momento*) y del gradiente al cuadrado (*segundo momento*). Según Kingma y Ba (2015), sus ecuaciones centrales son:

  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla C(w_t)
  $$

  $$
  v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla C(w_t))^2
  $$

  Con corrección de sesgo:

  $$
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
  \qquad ; \qquad
  \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
  $$

  Actualización final del peso:

  $$
  w_{t+1}
  =
  w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  $$

  Adam combina adaptabilidad, estabilidad y rapidez de convergencia, lo que lo convierte en una opción ideal para arquitecturas profundas y problemas con gradientes ruidosos.

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
  proyecto-final/
  ├── src/
  │   ├── layers/
  │   ├── optimizers/
  │   └── main.cpp
  ├── tests/
  └── docs/
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Mariano Sanchez | Documentar bases teóricas |
| Diseño de la arquitectura | Mia Wood | UML y esquemas de clases  |
| Implementación del modelo | Sebastian Chahuara | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> *Actualizar con tareas y nombres reales.*

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> *Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas.*

## Referencias

> <a name="ref1">[1]</a> McCulloch, W. S., & Pitts, W. H. (1943). *A logical calculus of the ideas immanent in nervous activity*. **Bulletin of Mathematical Biophysics, 5**, 115–133.

> <a name="ref2">[2]</a> Minsky, M., & Papert, S. (1969). *Perceptrons*. MIT Press.

> <a name="ref3">[3]</a> Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors*. **Nature, 323**(6088), 533–536.

> <a name="ref4">[4]</a> Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.

> <a name="ref5">[5]</a> Gálvez Siuce, J. (2023). *Diseño y entrenamiento de redes neuronales convolucionales para tareas de clasificación de imágenes* (Tesis de licenciatura). Universidad Nacional Mayor de San Marcos.

> <a name="ref6">[6]</a> Universidad Pontificia Comillas. (2023). *Predicción del precio de Bitcoin utilizando redes LSTM: Comparación con modelos ARIMA*. Instituto de Investigación Tecnológica.

> <a name="ref7">[7]</a> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

> <a name="ref8">[8]</a> Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. International Conference on Learning Representations (ICLR).

> <a name="ref9">[9]</a> Richaud, E. (2021). *Optimización en redes neuronales profundas*. Universidad Nacional de La Plata.

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---

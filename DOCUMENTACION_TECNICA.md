# 🐕 Clasificador de Condición Corporal Canina
## Documentación Técnica del Modelo de Deep Learning

---

## 📋 Índice
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Arquitectura del Modelo](#arquitectura-del-modelo)
3. [Proceso de Entrenamiento](#proceso-de-entrenamiento)
4. [Conceptos Técnicos Clave](#conceptos-técnicos-clave)
5. [Implementación](#implementación)
6. [Resultados y Evaluación](#resultados-y-evaluación)
7. [Uso del Modelo](#uso-del-modelo)

---

## 🎯 Descripción del Proyecto

### Objetivo
Desarrollar un sistema de clasificación automática que evalúe la condición corporal de caninos mediante análisis de imágenes, clasificándolos en tres categorías:
- **Delgado**: Bajo peso corporal
- **Normal**: Peso corporal ideal
- **Obeso**: Sobrepeso u obesidad

### Justificación
La evaluación de la condición corporal es crucial para:
- Prevención de problemas de salud
- Ajuste de dietas
- Monitoreo de tratamientos
- Detección temprana de malnutrición

### Tecnologías Utilizadas
- **Python 3.x**
- **PyTorch**: Framework de Deep Learning
- **torchvision**: Modelos preentrenados y utilidades
- **PIL/Pillow**: Procesamiento de imágenes
- **OpenCV**: (Opcional) Detección previa de caninos

---

## 🧠 Arquitectura del Modelo

### 1. Backbone: ResNet50

```
ResNet50 (Preentrenado en ImageNet)
├── Conv1 (Convolucional inicial)
├── Layer1 (Bloque residual) → CONGELADO
├── Layer2 (Bloque residual) → CONGELADO
├── Layer3 (Bloque residual) → CONGELADO
├── Layer4 (Bloque residual) → DESCONGELADO (Fine-tuning)
└── FC (Clasificador) → REEMPLAZADO
```

**¿Por qué ResNet50?**
- Red Neuronal Convolucional profunda (50 capas)
- Preentrenada en ImageNet (1.4M imágenes, 1000 clases)
- Arquitectura residual que permite entrenar redes muy profundas
- Excelente balance entre precisión y velocidad

### 2. Clasificador Personalizado

```python
model.fc = nn.Sequential(
    nn.Dropout(0.5),              # Regularización: Desactiva 50% neuronas
    nn.Linear(2048, 512),         # Capa densa: 2048 → 512 características
    nn.ReLU(),                    # Activación: f(x) = max(0, x)
    nn.Dropout(0.3),              # Regularización: Desactiva 30% neuronas
    nn.Linear(512, 3)             # Capa final: 512 → 3 clases
)
```

**Componentes:**
- **Dropout**: Previene sobreajuste (overfitting) desactivando neuronas aleatoriamente
- **Linear (Dense)**: Transforma características en predicciones
- **ReLU**: Función de activación no lineal
- **Salida**: 3 neuronas (una por clase)

### 3. Transfer Learning

```
ESTRATEGIA ADOPTADA:
┌─────────────────────────────────────┐
│ Capas Congeladas (Frozen)           │
│ - Mantienen conocimiento general    │
│ - Detectan bordes, texturas, formas │
│ - Layers 1-3 de ResNet50            │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Capa Descongelada (Fine-tuning)     │
│ - Se adapta a características       │
│   específicas de condición corporal │
│ - Layer 4 de ResNet50               │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Clasificador Personalizado          │
│ - Aprende desde cero                │
│ - Especializado en 3 clases         │
└─────────────────────────────────────┘
```

---

## 🔄 Proceso de Entrenamiento

### 1. Preprocesamiento de Datos

#### Para Entrenamiento (Train)
```python
transforms.Compose([
    transforms.Resize((256, 256)),                           # Redimensionar
    transforms.RandomResizedCrop(224),                       # Recorte aleatorio
    transforms.RandomHorizontalFlip(p=0.5),                  # Voltear horizontal
    transforms.RandomRotation(15),                           # Rotar ±15°
    transforms.ColorJitter(brightness=0.2, contrast=0.2),    # Ajuste de color
    transforms.RandomGrayscale(p=0.1),                       # 10% escala de grises
    transforms.ToTensor(),                                   # Convertir a tensor
    transforms.Normalize([0.485, 0.456, 0.406],             # Normalización ImageNet
                         [0.229, 0.224, 0.225])
])
```

#### Para Validación (Val)
```python
transforms.Compose([
    transforms.Resize((256, 256)),              # Redimensionar
    transforms.CenterCrop(224),                 # Recorte central
    transforms.ToTensor(),                      # Convertir a tensor
    transforms.Normalize([0.485, 0.456, 0.406], # Normalización ImageNet
                         [0.229, 0.224, 0.225])
])
```

**Data Augmentation: ¿Por qué?**
- Aumenta artificialmente el tamaño del dataset
- Mejora generalización del modelo
- Previene memorización (overfitting)
- Simula diferentes condiciones de captura

### 2. Configuración de Hiperparámetros

```python
BATCH_SIZE = 16        # Imágenes procesadas simultáneamente
EPOCHS = 25            # Iteraciones completas sobre el dataset
LR = 0.0005           # Learning Rate (tasa de aprendizaje)
NUM_CLASSES = 3        # delgado, normal, obeso
```

**Optimizador: Adam**
```python
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.0005,              # Learning rate
    weight_decay=1e-4       # Regularización L2
)
```

**Scheduler: StepLR**
```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=7,            # Cada 7 épocas
    gamma=0.1               # Reduce LR × 0.1
)
```

### 3. Función de Pérdida

```python
criterion = nn.CrossEntropyLoss()
```

**CrossEntropyLoss:**
- Combina Softmax + Negative Log Likelihood
- Ideal para clasificación multiclase
- Penaliza predicciones incorrectas
- Fórmula: `Loss = -log(probabilidad_clase_correcta)`

### 4. Ciclo de Entrenamiento

```
PARA CADA ÉPOCA (1 a 25):
    ┌─────────────────────────────────────┐
    │ FASE 1: ENTRENAMIENTO               │
    │ ----------------------------------- │
    │ PARA CADA BATCH:                    │
    │   1. Cargar imágenes y etiquetas    │
    │   2. Forward pass (predicción)      │
    │   3. Calcular pérdida (loss)        │
    │   4. Backward pass (gradientes)     │
    │   5. Actualizar pesos               │
    │                                     │
    │ RESULTADO: Loss y Accuracy en train │
    └─────────────────────────────────────┘
              ↓
    ┌─────────────────────────────────────┐
    │ FASE 2: VALIDACIÓN                  │
    │ ----------------------------------- │
    │ PARA CADA BATCH (sin gradientes):   │
    │   1. Cargar imágenes y etiquetas    │
    │   2. Forward pass (predicción)      │
    │   3. Calcular pérdida y accuracy    │
    │   4. Calcular métricas por clase    │
    │                                     │
    │ RESULTADO: Loss y Accuracy en val   │
    └─────────────────────────────────────┘
              ↓
    ┌─────────────────────────────────────┐
    │ FASE 3: EVALUACIÓN                  │
    │ ----------------------------------- │
    │ - Comparar con mejor accuracy       │
    │ - Guardar modelo si mejoró          │
    │ - Actualizar learning rate          │
    │ - Mostrar estadísticas              │
    └─────────────────────────────────────┘
```

---

## 🔑 Conceptos Técnicos Clave

### 1. Transfer Learning (Aprendizaje por Transferencia)

**Definición:**
Técnica que reutiliza un modelo preentrenado en un problema diferente para resolver un nuevo problema relacionado.

**Ventajas:**
- ✅ Requiere menos datos de entrenamiento
- ✅ Entrena más rápido (convergencia acelerada)
- ✅ Mayor precisión con datasets pequeños
- ✅ Aprovecha conocimiento previo

**Analogía:**
Es como contratar a un médico veterinario experimentado y especializarlo en evaluación de condición corporal, en lugar de enseñarle medicina veterinaria desde cero.

### 2. Fine-tuning (Ajuste Fino)

**Proceso:**
1. Cargar modelo preentrenado (ResNet50)
2. Congelar capas iniciales (mantienen conocimiento general)
3. Descongelar capas finales (se adaptan al nuevo problema)
4. Entrenar con learning rate bajo

**En este proyecto:**
- Capas 1-3: **Congeladas** (detectan patrones generales)
- Capa 4: **Descongelada** (se adapta a perros)
- Clasificador: **Nuevo** (específico para 3 clases)

### 3. Regularización

**Técnicas aplicadas:**

#### a) Dropout
```python
nn.Dropout(0.5)  # Desactiva 50% de neuronas aleatoriamente
```
- Previene co-adaptación de neuronas
- Funciona como ensemble de múltiples redes
- Solo activo durante entrenamiento

#### b) Weight Decay (L2 Regularization)
```python
optimizer = optim.Adam(..., weight_decay=1e-4)
```
- Penaliza pesos grandes
- Previene overfitting
- Fórmula: `Loss_total = Loss + λ × Σ(pesos²)`

#### c) Data Augmentation
- Aumenta variabilidad del dataset
- Simula diferentes condiciones
- Mejora generalización

### 4. Métricas de Evaluación

#### Accuracy (Exactitud)
```
Accuracy = (Predicciones Correctas) / (Total de Predicciones)
```

#### Loss (Pérdida)
```
CrossEntropyLoss = -Σ y_true × log(y_pred)
```

#### Accuracy por Clase
```
Accuracy_delgado = Correctos_delgado / Total_delgado
Accuracy_normal = Correctos_normal / Total_normal
Accuracy_obeso = Correctos_obeso / Total_obeso
```

---

## 💻 Implementación

### Estructura del Dataset

```
dataset/
├── train/
│   ├── delgado/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── normal/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── obeso/
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
└── val/
    ├── delgado/
    ├── normal/
    └── obeso/
```

### Archivos del Proyecto

```
Training_body_condition/
├── training_model.py           # Entrenamiento del modelo
├── predict_image.py            # Predicción simple
├── predict_with_detection.py   # Predicción con detección XML
├── test_etiquetas.py           # Verificación de dataset
├── requirements.txt            # Dependencias
├── DOCUMENTACION_TECNICA.md    # Este archivo
├── dataset/                    # Datos de entrenamiento/validación
└── img/                        # Imágenes de prueba
```

### Modelos Generados

```
best_dog_body_condition_classifier.pth   # Mejor modelo (mayor val_accuracy)
dog_body_condition_classifier.pth        # Modelo final (última época)
```

---

## 📊 Resultados y Evaluación

### Métricas Monitoreadas

Durante el entrenamiento se monitorean:

1. **Train Loss**: Pérdida en conjunto de entrenamiento
2. **Train Accuracy**: Exactitud en conjunto de entrenamiento
3. **Val Loss**: Pérdida en conjunto de validación
4. **Val Accuracy**: Exactitud en conjunto de validación
5. **Accuracy por Clase**: Precisión individual para cada categoría

### Interpretación de Resultados

#### Ejemplo de Salida del Entrenamiento:

```
📊 Resultados Epoch 25:
   🔹 Train Loss: 0.2341 | Train Acc: 0.9123
   🔸 Val Loss: 0.3142   | Val Acc: 0.8765
   
   📈 Accuracy por clase:
      delgado : 0.8500 (85/100)
      normal  : 0.9200 (92/100)
      obeso   : 0.8600 (86/100)
```

**Análisis:**
- ✅ **Train Acc > Val Acc**: Normal, indica aprendizaje
- ⚠️ **Train Acc >> Val Acc**: Posible overfitting
- ✅ **Accuracy balanceado por clase**: Buen desempeño general
- ⚠️ **Accuracy desbalanceado**: Sesgo hacia ciertas clases

### Prevención de Overfitting

**Señales de overfitting:**
- Train accuracy muy alta (>95%) pero val accuracy baja (<75%)
- Train loss bajando pero val loss subiendo

**Soluciones implementadas:**
1. Dropout (0.5 y 0.3)
2. Weight decay (1e-4)
3. Data augmentation
4. Early stopping (guardar mejor modelo)

---

## 🚀 Uso del Modelo

### 1. Entrenamiento

```bash
python training_model.py
```

**Salida esperada:**
```
🎯 Usando dispositivo: cuda
📊 Configuración: Batch=16, Epochs=25, LR=0.0005
Clases detectadas: ['delgado', 'normal', 'obeso']

🚀 Iniciando entrenamiento...
============================================================
📅 Epoch 1/25
...
⭐ Nuevo mejor modelo guardado! Val Acc: 0.8765
============================================================

🎉 Entrenamiento completado!
🏆 Mejor accuracy de validación: 0.8765
```

### 2. Predicción Simple

**Archivo:** `predict_image.py`

```python
# Configurar ruta de imagen
IMAGE_PATH = "img/mi_perro.jpg"

# Ejecutar
python predict_image.py
```

**Salida:**
```
🐕 CLASIFICADOR DE CONDICIÓN CORPORAL CANINA
==================================================
📷 Analizando imagen: img/mi_perro.jpg
==================================================

📊 RESULTADOS DEL ANÁLISIS
========================================
🎯 Condición corporal: NORMAL
🔍 Confianza: 87.42%

📈 Probabilidades:
   delgado :   8.35% █
   normal  :  87.42% █████████████████
   obeso   :   4.23% ████

💡 Interpretación:
   El canino presenta un peso corporal ideal.
========================================
```

### 3. Predicción con Detección (Avanzado)

**Archivo:** `predict_with_detection.py`

```python
# Configurar rutas
IMAGE_PATH = "img/mi_perro.jpg"
DOG_DETECTOR_XML = "haarcascade_fullbody.xml"

# Ejecutar
python predict_with_detection.py
```

**Ventajas:**
1. Detecta automáticamente el canino en la imagen
2. Extrae solo la región relevante
3. Clasifica con mayor precisión
4. Guarda imagen con resultado visual

---

## 🎓 Conceptos para Explicar al Profesor

### 1. ¿Por qué Deep Learning?

**Ventajas sobre métodos tradicionales:**
- ✅ Aprende características automáticamente (no requiere ingeniería manual)
- ✅ Maneja variabilidad en razas, poses, iluminación
- ✅ Escala bien con más datos
- ✅ Estado del arte en visión por computadora

### 2. ¿Por qué ResNet50?

**Características destacadas:**
- Red residual profunda (50 capas)
- Skip connections previenen vanishing gradient
- Preentrenada en ImageNet
- Balance entre precisión y eficiencia

### 3. ¿Cómo funciona la predicción?

```
Imagen del perro
      ↓
Preprocesamiento (resize, normalización)
      ↓
Extracción de características (ResNet50)
  - Detecta bordes, texturas, formas
  - Identifica patrones de condición corporal
      ↓
Clasificador personalizado
  - Procesa características extraídas
  - Genera probabilidades para cada clase
      ↓
Softmax (conversión a probabilidades)
  [delgado: 0.08, normal: 0.87, obeso: 0.04]
      ↓
Predicción final: NORMAL (87% confianza)
```

### 4. Diferencias con otros enfoques

| Aspecto | Enfoque Tradicional | Deep Learning (Este proyecto) |
|---------|---------------------|-------------------------------|
| Características | Manual (SIFT, HOG) | Automáticas (CNN) |
| Precisión | ~70-75% | ~85-95% |
| Adaptabilidad | Baja | Alta |
| Datos requeridos | Moderados | Moderados-Altos |
| Tiempo entrenamiento | Rápido | Medio |
| Transfer Learning | No aplica | ✅ Sí (ResNet50) |

### 5. Aplicaciones Reales

Este tipo de modelo se puede usar para:
- 📱 Apps móviles de monitoreo de mascotas
- 🏥 Sistemas de telemedicina veterinaria
- 🏋️ Programas de control de peso canino
- 📊 Estudios epidemiológicos de obesidad animal
- 🔬 Investigación en nutrición animal

---

## 📚 Referencias Técnicas

### Papers Relevantes
1. **ResNet**: "Deep Residual Learning for Image Recognition" (He et al., 2016)
2. **Transfer Learning**: "A Survey on Transfer Learning" (Pan & Yang, 2010)
3. **Data Augmentation**: "The Effectiveness of Data Augmentation in Image Classification using Deep Learning" (Perez & Wang, 2017)

### Frameworks y Librerías
- PyTorch: https://pytorch.org/
- torchvision: https://pytorch.org/vision/
- ResNet50: https://pytorch.org/vision/stable/models.html#resnet

---

## 🔧 Requerimientos del Sistema

### Software
```
Python >= 3.8
PyTorch >= 2.0
torchvision >= 0.15
Pillow >= 9.0
numpy >= 1.20
```

### Hardware Recomendado
- **GPU**: NVIDIA con CUDA (recomendado)
- **RAM**: Mínimo 8GB
- **Almacenamiento**: 2GB para modelo + dataset

---

## 👨‍💻 Autor y Contacto

**Proyecto:** Clasificador de Condición Corporal Canina  
**Curso:** Construcción de Software - 5to Semestre Ing. Software  
**Repositorio:** body_condition (LuisArguello1)

---

## 📝 Conclusiones

Este proyecto demuestra la aplicación práctica de:
- ✅ Deep Learning en visión por computadora
- ✅ Transfer Learning para problemas con datos limitados
- ✅ Técnicas de regularización para prevenir overfitting
- ✅ Evaluación rigurosa de modelos de clasificación
- ✅ Implementación profesional con PyTorch

El modelo desarrollado logra alta precisión en la clasificación de condición corporal canina, demostrando que las redes neuronales convolucionales son efectivas para este tipo de tareas de análisis visual en el ámbito veterinario.

---

**Última actualización:** Noviembre 2025  
**Versión:** 1.0

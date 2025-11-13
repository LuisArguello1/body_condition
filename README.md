# Clasificador de Condición Corporal Canina

Un modelo de inteligencia artificial que evalúa automáticamente la condición corporal de perros a partir de fotografías, clasificándolos en tres categorías: delgado, normal y obeso.

---

## Descripción del Proyecto

Este proyecto utiliza Deep Learning para analizar imágenes de perros y determinar su condición corporal mediante técnicas de visión por computadora. El sistema está diseñado para asistir en la evaluación veterinaria y el monitoreo de la salud canina.

### Objetivo

Desarrollar un clasificador automático capaz de identificar la condición corporal de caninos con alta precisión, proporcionando una herramienta útil para veterinarios, dueños de mascotas y profesionales del cuidado animal.

### Categorías de Clasificación

El modelo clasifica a los caninos en tres estados corporales:

1. **Delgado**: Bajo peso corporal
2. **Normal**: Peso corporal ideal
3. **Obeso**: Sobrepeso u obesidad

---

## Arquitectura del Modelo

### Transfer Learning con ResNet50

El proyecto implementa Transfer Learning utilizando ResNet50, una red neuronal convolucional profunda preentrenada en ImageNet (1.4 millones de imágenes). Esta técnica permite:

- Aprovechar conocimiento previo de reconocimiento de patrones visuales
- Reducir significativamente el tiempo de entrenamiento
- Lograr alta precisión con datasets relativamente pequeños
- Optimizar el uso de recursos computacionales

### Componentes del Modelo

```
ResNet50 (Preentrenado en ImageNet)
    ↓
Capas Congeladas (Layers 1-3)
    - Mantienen conocimiento general
    - Detectan bordes, texturas y formas básicas
    ↓
Capa de Fine-tuning (Layer 4)
    - Se adapta a características específicas
    - Aprende patrones de condición corporal
    ↓
Clasificador Personalizado
    - Dropout (0.5)
    - Linear (2048 → 512)
    - ReLU
    - Dropout (0.3)
    - Linear (512 → 3 clases)
```
## Estructura del Dataset

### Organización de Directorios

```
dataset/
├── train/
│   ├── delgado/
│   │   ├── imagen001.jpg
│   │   ├── imagen002.jpg
│   │   └── ...
│   ├── normal/
│   │   ├── imagen001.jpg
│   │   ├── imagen002.jpg
│   │   └── ...
│   └── obeso/
│       ├── imagen001.jpg
│       ├── imagen002.jpg
│       └── ...
└── val/
    ├── delgado/
    ├── normal/
    └── obeso/
```

### Distribución de Datos

- **Train**: 80% de las imágenes (para entrenamiento)
- **Val**: 20% de las imágenes (para validación)

Cada carpeta representa una clase de condición corporal, y el modelo aprende automáticamente las características visuales distintivas de cada categoría.

---

## Procesamiento de Imágenes

### Transformaciones de Entrenamiento

Las imágenes de entrenamiento pasan por las siguientes transformaciones:

1. **Resize**: Redimensionamiento a 256x256 píxeles
2. **RandomResizedCrop**: Recorte aleatorio a 224x224 píxeles
3. **RandomHorizontalFlip**: Volteo horizontal con probabilidad 0.5
4. **RandomRotation**: Rotación aleatoria de ±15 grados
5. **ColorJitter**: Variación de brillo, contraste y saturación (±20%)
6. **RandomGrayscale**: Conversión a escala de grises con probabilidad 0.1
7. **Normalización**: Estandarización usando medias y desviaciones de ImageNet

### Transformaciones de Validación

Las imágenes de validación usan transformaciones más conservadoras:

1. **Resize**: Redimensionamiento a 256x256 píxeles
2. **CenterCrop**: Recorte central a 224x224 píxeles
3. **Normalización**: Estandarización usando medias y desviaciones de ImageNet

---

## Configuración de Entrenamiento

### Hiperparámetros

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Batch Size | 16 | Imágenes procesadas simultáneamente |
| Épocas | 25 | Iteraciones completas sobre el dataset |

---

## Proceso de Entrenamiento

### Guardado de Modelos

El sistema guarda dos versiones del modelo:

- `best_dog_body_condition_classifier.pth`: Modelo con mejor accuracy de validación
- `dog_body_condition_classifier.pth`: Modelo de la última época

---

## Uso del Sistema

### Requisitos

```
Python >= 3.8
PyTorch >= 2.0
torchvision >= 0.15
Pillow >= 9.0
numpy >= 1.20
```

### Niveles de Confianza

- **Mayor a 80%**: Alta confianza - Resultado muy confiable
- **60% - 80%**: Confianza moderada - Resultado probable
- **Menor a 60%**: Baja confianza - Se recomienda verificación adicional

---

## Características Visuales Detectadas

El modelo aprende automáticamente a identificar características visuales indicativas de condición corporal:

### Para Condición "Delgado":
- Visibilidad prominente de costillas
- Cintura muy marcada
- Ausencia de grasa corporal visible
- Estructura ósea visible

### Para Condición "Normal":
- Costillas palpables pero no visibles
- Cintura definida vista desde arriba
- Abdomen recogido visto de lado
- Proporciones corporales equilibradas

### Para Condición "Obeso":
- Costillas no palpables
- Cintura ausente o apenas visible
- Acumulación de grasa en abdomen y base de cola
- Contorno corporal redondeado

---

## Tecnologías y Frameworks

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.8+ | Lenguaje de programación |
| PyTorch | 2.0+ | Framework de Deep Learning |
| torchvision | 0.15+ | Modelos y transformaciones |
| ResNet50 | - | Arquitectura de red neuronal |
| Pillow | 9.0+ | Procesamiento de imágenes |
| NumPy | 1.20+ | Operaciones numéricas |
| OpenCV | 4.5+ (opcional) | Detección de caninos |

---

## Conceptos Técnicos

### Transfer Learning

Técnica que reutiliza un modelo preentrenado en un problema diferente para resolver un nuevo problema relacionado. En este proyecto, aprovechamos ResNet50 entrenado en ImageNet para clasificación de condición corporal.

**Ventajas:**
- Requiere menos datos de entrenamiento
- Converge más rápido
- Logra mayor precisión con datasets limitados
- Reduce costos computacionales

### Fine-tuning

Proceso de ajustar un modelo preentrenado para una tarea específica. Se "descongelan" las últimas capas para permitir que se adapten al nuevo problema mientras las capas iniciales mantienen conocimiento general.

---

## Aplicaciones Potenciales

Este sistema puede ser utilizado en diversos contextos:

1. **Clínicas Veterinarias**: Evaluación rápida de condición corporal
2. **Aplicaciones Móviles**: Monitoreo de mascotas por parte de dueños
3. **Telemedicina Veterinaria**: Consultas remotas
4. **Refugios de Animales**: Evaluación masiva de condición de salud
5. **Estudios Epidemiológicos**: Investigación sobre obesidad canina
6. **Programas de Control de Peso**: Seguimiento de progreso

---

## Limitaciones

- El modelo está entrenado específicamente para perros, no funciona con otras especies
- La precisión puede verse afectada por la calidad de la imagen
- Ángulos extremos o imágenes parciales pueden reducir la confianza
- No reemplaza la evaluación clínica profesional
- Requiere imágenes donde el cuerpo del perro sea visible

---

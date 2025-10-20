import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# =====================
# CONFIGURACIÓN
# =====================
# 🔸 CAMBIA AQUÍ LA RUTA DE TU IMAGEN 🔸
IMAGE_PATH = "img/imagen9.webp"  # ← Pon aquí la ruta de tu imagen

NUM_CLASSES = 3  # delgado, normal, obeso
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Priorizar el mejor modelo si existe
BEST_MODEL_PATH = "best_dog_body_condition_classifier.pth"
MODEL_PATH = "dog_body_condition_classifier.pth"

# Clases correspondientes a los índices del modelo
CLASS_NAMES = ['delgado', 'normal', 'obeso']

# =====================
# TRANSFORMACIONES PARA LA IMAGEN
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    """Carga el modelo entrenado (prioriza el mejor modelo)"""
    print("🔍 Buscando modelo...")
    
    # Usar ResNet50 con la misma arquitectura del entrenamiento
    model = models.resnet50(pretrained=False)
    
    # Recrear la misma arquitectura del modelo entrenado
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    
    # Intentar cargar el mejor modelo primero
    if os.path.exists(BEST_MODEL_PATH):
        print(f"⭐ Cargando mejor modelo: {BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print("✅ Mejor modelo cargado exitosamente")
        return model
    elif os.path.exists(MODEL_PATH):
        print(f"📁 Cargando modelo estándar: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print("✅ Modelo estándar cargado exitosamente")
        return model
    else:
        print(f"❌ Error: No se encontró ningún modelo entrenado")
        print("Modelos buscados:")
        print(f"   - {BEST_MODEL_PATH}")
        print(f"   - {MODEL_PATH}")
        print("Entrena el modelo primero ejecutando: python training_model.py")
        return None

def predict_image(model, image_path):
    """Predice la condición corporal de un canino en una imagen"""
    try:
        # Verificar si la imagen existe
        if not os.path.exists(image_path):
            print(f"❌ Error: No se encontró la imagen en {image_path}")
            return None
        
        # Cargar y procesar la imagen
        print(f"Analizando imagen: {image_path}")
        image = Image.open(image_path).convert("RGB")
        
        # Aplicar transformaciones
        input_tensor = transform(image).unsqueeze(0)  # Añadir dimensión de batch
        input_tensor = input_tensor.to(DEVICE)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[predicted.item()]
            confidence_percent = probabilities[0][predicted.item()].item() * 100
        
        return predicted_class, confidence_percent, probabilities[0]
        
    except Exception as e:
        print(f"❌ Error al procesar la imagen: {str(e)}")
        return None

def main():
    # Cargar el modelo
    model = load_model()
    if model is None:
        return
    
    print("🐕 CLASIFICADOR DE CONDICIÓN CORPORAL CANINA")
    print("="*50)
    print(f"📷 Analizando imagen: {IMAGE_PATH}")
    print("="*50)
    
    # Realizar predicción de la imagen configurada
    result = predict_image(model, IMAGE_PATH)
    
    if result:
        predicted_class, confidence, all_probabilities = result
        
        print("\n📊 RESULTADOS DEL ANÁLISIS")
        print("="*40)
        print(f"🎯 Condición corporal: {predicted_class.upper()}")
        print(f"🔍 Confianza: {confidence:.2f}%")
        print("\n📈 Probabilidades:")
        for i, class_name in enumerate(CLASS_NAMES):
            prob = all_probabilities[i].item() * 100
            bar = "█" * int(prob / 5)
            print(f"   {class_name:8}: {prob:6.2f}% {bar}")
        
        # Interpretación
        print(f"\n💡 Interpretación:")
        if predicted_class == "delgado":
            print("   El canino presenta bajo peso corporal.")
        elif predicted_class == "normal":
            print("   El canino presenta un peso corporal ideal.")
        else:  # obeso
            print("   El canino presenta sobrepeso u obesidad.")
        
        if confidence < 60:
            print("⚠️  Advertencia: Confianza baja.")
        
        print("="*40)
    else:
        print("❌ No se pudo analizar la imagen.")
        print("� Verifica que:")
        print("   - La ruta de la imagen sea correcta")
        print("   - El archivo exista")
        print("   - Sea una imagen válida (.jpg, .jpeg, .png)")

if __name__ == "__main__":
    main()
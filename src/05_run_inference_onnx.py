# inference_onnx.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import sys
import json

def preprocess(img_path):
    img = Image.open(img_path).resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify(image_path):
    # Cargar nombres de clases
    with open("data/class_names.json", "r") as f:
        class_names = json.load(f)

    session = ort.InferenceSession("models/mobilenetv2_quant.onnx")
    input_name = session.get_inputs()[0].name

    img = preprocess(image_path)  # Cambia esto
    outputs = session.run(None, {input_name: img})

    # Obtener predicción
    pred_index = np.argmax(outputs[0])
    pred_class = class_names[pred_index]

    print("Probabilidades:", outputs[0])
    print("Predicción:", pred_class)

if __name__ == '__main__':
    image_path = sys.argv[1]
    classify(image_path)
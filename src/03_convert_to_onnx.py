# convert_to_onnx.py
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("models/mobilenetv2_trained.keras")

# Convertir a ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path="models/mobilenetv2.onnx")
print("Modelo convertido a ONNX: models/mobilenetv2.onnx")

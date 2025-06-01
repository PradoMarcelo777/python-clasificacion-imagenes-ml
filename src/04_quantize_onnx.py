# quantize_onnx.py
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="models/mobilenetv2.onnx",
    model_output="models/mobilenetv2_quant.onnx",
    weight_type=QuantType.QUInt8
)

print("Modelo cuantizado a 8-bit (cuantización extrema) guardado en models/mobilenetv2_quant.onnx")

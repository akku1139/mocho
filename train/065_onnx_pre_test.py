import onnxruntime as ort

session = ort.InferenceSession("../model/weights/v1/mocho.onnx")
for input in session.get_inputs():
    # ['length', 'batch'] のように名前がついていれば成功
    # 固定されている場合は [8, 1] のように数字になります
    print(f"Input: {input.name}, Shape: {input.shape}")

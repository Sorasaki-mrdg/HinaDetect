import os

import cv2
import onnxruntime as ort
import numpy as np

def load_model():
    # 加载 ONNX 模型
    global ort_session
    onnx_model_path = 'best_seia_model.onnx'
    ort_session = ort.InferenceSession(onnx_model_path)


# 图像预处理（使用 OpenCV）
def preprocess_image(image_path):
    # 使用 OpenCV 读取图像并转换为 RGB 格式
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 读取的是 BGR，转换为 RGB
    image = cv2.resize(image, (224, 224))  # 根据模型要求调整图像大小
    image = image.astype(np.float32) / 255.0  # 将像素值归一化到 [0, 1] 范围

    # 使用 ImageNet 数据集的均值和标准差进行标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std  # 标准化

    # 转换为 (C, H, W) 格式，并增加 batch 维度
    image = np.transpose(image, (2, 0, 1))  # 转换为 (C, H, W) 格式
    image = np.expand_dims(image, axis=0)  # 增加 batch 维度

    return image


# 处理用户输入的图像路径并进行推理
def predict_image(image_path):
    # 预处理图像
    image = preprocess_image(image_path)

    # 获取模型输入名称
    input_name = ort_session.get_inputs()[0].name

    # 推理：获取预测结果
    ort_inputs = {input_name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)

    # 使用 argmax 获取最大值所在的位置（即预测的类别）
    pred = np.argmax(ort_outs[0], axis=1)
    return pred.item() == 1, ort_outs[0][0][1]

def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

if __name__ == '__main__':
    load_model()
    files=get_all_files('input')
    # print(files)
    for file in files:
        try:
            # 使用 ONNX 模型进行推理
            pred, confidence = predict_image(file)
            print(f"File: {file} | Prediction: {pred} | Confidence: {confidence}")
        except Exception as e:
            print(f"Error processing {file}: {e}")


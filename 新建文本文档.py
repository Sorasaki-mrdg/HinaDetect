import os
import cv2
import onnxruntime as ort
import numpy as np
import shutil

# 支持的图片扩展名
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.jfif'}

def load_model():
    """加载 ONNX 模型"""
    global ort_session
    onnx_model_path = 'best_hina_model.onnx'
    ort_session = ort.InferenceSession(onnx_model_path)

def preprocess_image(image_path):
    """图像预处理"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # 使用 OpenCV 读取图像
    image = cv2.imread(image_path)
    if image is None:
        # 尝试用其他方式读取（处理中文路径问题）
        try:
            with open(image_path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Could not read image {image_path}: {e}")
    
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image

def predict_image(image_path):
    """预测图片"""
    image = preprocess_image(image_path)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = np.argmax(ort_outs[0], axis=1)
    return pred.item() == 1, ort_outs[0][0][1]

def get_all_image_files(directory):
    """获取目录中所有支持的图片文件"""
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                try:
                    file_path = os.path.join(root, file)
                    # 验证文件可读性
                    with open(file_path, 'rb') as f:
                        pass
                    all_files.append(file_path)
                except Exception as e:
                    print(f"Warning: Could not process file {file}: {e}")
    return all_files

def ensure_output_dir(directory):
    """确保输出目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created output directory: {directory}")

if __name__ == '__main__':
    # 输入和输出目录
    input_dir = r'F:\picture'
    output_dir = r'.\testhina'
    
    # 确保输出目录存在
    ensure_output_dir(output_dir)
    
    # 加载模型
    load_model()
    
    # 获取所有图片文件
    image_files = get_all_image_files(input_dir)
    print(f"Found {len(image_files)} image files to process")
    
    # 处理每张图片
    for image_file in image_files:
        try:
            # 预测图片
            pred, confidence = predict_image(image_file)
            
            # 如果预测为True，则复制到输出目录
            if pred:
                filename = os.path.basename(image_file)
                dest_path = os.path.join(output_dir, filename)
                
                # 处理文件名冲突
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    dest_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.copy2(image_file, dest_path)
                print(f"Copied: {image_file} -> {dest_path} (Confidence: {confidence:.4f})")
            else:
                print(f"Skipped: {image_file} (Not matched, Confidence: {confidence:.4f})")
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
import PredictONNX
from PIL import Image
from PIL import ImageGrab
import os
import shutil

if __name__ == '__main__':
    PredictONNX.load_model('best_hina_model.onnx')
    while True:
        image = ImageGrab.grabclipboard()
        if image is None:
            print("剪贴板中没有图像或文件路径！")

        if isinstance(image, list):
            # 提取列表中的第一个元素（假设是文件路径）
            file_path = image[0].strip('"\' ')  # 去除路径两端的引号和空格

            # 检查是否为文件
            if os.path.isfile(file_path):
                # 检查文件扩展名
                valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.jfif'}
                ext = os.path.splitext(file_path)[1].lower()
                if ext in valid_extensions:
                    try:
                        try:
                            # 使用 ONNX 模型进行推理
                            pred, confidence = PredictONNX.predict_image(file_path)
                            print(f"File: {file_path} | Prediction: {'是hina' if pred else '啊哈哈，不知道'} | Confidence: {confidence}")
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                    except Exception as e:
                        print(f"错误：无法打开图片文件，原因：{e}")
                else:
                    print(f"错误：不支持的文件类型 '{ext}'")
            else:
                print("错误：路径不存在或不是文件")
        if isinstance(image, Image.Image):
            #print("剪贴板的内容是图")
            try:
                # 临时保存到文件（因为PredictONNX需要文件路径）
                temp_path = "clipboard_temp.png"
                image.save(temp_path)
                # 调用预测函数
                pred, confidence = PredictONNX.predict_image(temp_path)
                print(f"File: {temp_path} | Prediction: {'是hina' if pred else '啊哈哈，不知道'} | Confidence: {confidence}")
                os.remove(temp_path)
            except Exception as e:
                print(f"Error processing {temp_path}: {e}")

        input("按任意键继续")
            


 

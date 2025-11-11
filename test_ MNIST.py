import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import io
from flask import Flask, request, render_template_string, redirect, url_for
from PIL import Image, ImageOps

# --- 检查数据文件夹 ---
if not os.path.exists("MNIST_npy"):
    print("=" * 50)
    print("错误：找不到 'MNIST_npy' 文件夹。")
    print("请确保数据文件 (train_images.npy 等) 存放在该文件夹中。")
    print("程序已退出。")
    print("=" * 50)
    exit()


# === 1. PyTorch 模型和数据加载 (来自你的代码) ===

class MNISTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        feature = torch.from_numpy(feature.astype(np.float32))
        label = torch.from_numpy(label.reshape(-1).astype(np.int64))

        sample = {'images': feature, 'label': label}
        return sample


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 网页预测时, x 的形状是 [1, 28, 28]
        # 训练时, x 的形状是 [64, 28, 28]
        # unsqueeze(1) 对两者都有效
        x = x.unsqueeze(1)  # [N, 1, 28, 28]
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out


def train_test_splitting(batch_size=64):
    train_images = np.load("MNIST_npy/train_images.npy")
    test_images = np.load("MNIST_npy/test_images.npy")
    train_labels = np.load("MNIST_npy/train_labels.npy")
    test_labels = np.load("MNIST_npy/test_labels.npy")
    train_data = MNISTDataset(train_images, train_labels)
    test_data = MNISTDataset(test_images, test_labels)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


def train_and_val(epoch_num=10, model_save_path="mnist_cnn.pth"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"开始训练... 使用设备: {device}")
    train_loader, test_loader = train_test_splitting()

    my_model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(epoch_num):
        for batch_idx, batch_data in enumerate(train_loader):
            images = batch_data['images'].to(device)
            labels = batch_data['label'].to(device)
            labels = labels.squeeze()
            outputs = my_model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {loss.item():.4f}')

    my_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_data in test_loader:
            images = batch_data['images'].to(device)
            labels = batch_data['label'].to(device)
            labels = labels.squeeze()
            outputs = my_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Test Accuracy : {100 * correct / total:.2f}%')

    # 保存模型
    torch.save(my_model.state_dict(), model_save_path)
    print(f"模型训练完成并保存到: {model_save_path}")


# === 2. Flask Web 应用部分 ===

# 将 HTML 内容嵌入到 Python 字符串中
# 我们使用 render_template_string，它仍然可以处理 Jinja2 模板语法 (例如 {% if ... %})
HTML_CONTENT = """
<!doctype html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>MNIST 手写数字预测</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 600px; 
            margin: 50px auto; 
            padding: 20px; 
            background-color: #f9f9f9; 
            border-radius: 10px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            text-align: center;
        }
        h1 { color: #333; }
        p { color: #555; line-height: 1.6; }
        form { 
            display: flex; 
            flex-direction: column; 
            gap: 15px; 
            margin-top: 25px;
        }
        input[type="file"] {
            padding: 12px;
            border: 2px dashed #007bff;
            border-radius: 5px;
            background-color: #f0f8ff;
            cursor: pointer;
        }
        button { 
            background-color: #007bff; 
            color: white; 
            padding: 12px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            font-size: 16px;
            font-weight: bold;
        }
        #result { 
            margin-top: 30px; 
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .prediction { 
            font-size: 3.5em; 
            font-weight: bold; 
            color: #d9534f; 
            margin: 10px 0;
        }
        .filename { font-style: italic; color: #777; }
        .error { color: #d9534f; font-weight: bold; }
    </style>
</head>
<body>
    <h1>MNIST 手写数字预测器</h1>
    <p>请上传一张包含单个手写数字的图片（最好是黑底白字或白底黑字）。</p>

    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">开始预测</button>
    </form>

    <div id="result">
        {% if prediction is not none %}
            <h3>预测结果:</h3>
            <p class="filename">文件: {{ filename }}</p>
            <p class="prediction">{{ prediction }}</p>
        {% elif error %}
            <p class="error">{{ error }}</p>
        {% else %}
            <p>请上传一张图片进行预测。</p>
        {% endif %}
    </div>
</body>
</html>
"""

# --- 全局变量和辅助函数 ---
app = Flask(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)  # 初始化模型结构
MODEL_PATH = "mnist_cnn.pth"


def process_image(image_bytes):
    """将上传的图像文件字节转换为模型所需的 [1, 28, 28] Tensor"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('L')  # 转为灰度
        image = ImageOps.invert(image)  # 反转颜色 (黑底白字 -> 白底黑字)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)  # 缩放
        img_np = np.array(image)
        img_np = img_np.astype(np.float32)
        tensor = torch.from_numpy(img_np)
        tensor = tensor.unsqueeze(0)  # 增加 batch 维度 [1, 28, 28]
        return tensor.to(device)
    except Exception as e:
        print(f"图像处理出错: {e}")
        return None


# --- Flask 路由 ---
@app.route('/', methods=['GET'])
def index():
    """渲染主页"""
    return render_template_string(HTML_CONTENT, prediction=None, error=None, filename=None)


@app.route('/predict', methods=['POST'])
def predict():
    """处理图像预测请求"""
    if 'file' not in request.files:
        return redirect(url_for('index'))  # 确保 import 了 url_for 和 redirect

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        try:
            img_bytes = file.read()
            tensor = process_image(img_bytes)
            if tensor is None:
                raise ValueError("无法处理图像")

            with torch.no_grad():
                output = model(tensor)
                _, predicted_idx = torch.max(output.data, 1)
                prediction = predicted_idx.item()

            return render_template_string(HTML_CONTENT, prediction=prediction, filename=file.filename)

        except Exception as e:
            return render_template_string(HTML_CONTENT, error=f"预测出错: {e}", filename=file.filename)
    return redirect(url_for('index'))

if __name__ == "__main__":
    # 1. 检查模型文件，如果不存在则进行训练
    if not os.path.exists(MODEL_PATH):
        print(f"--- 未找到模型文件 '{MODEL_PATH}' ---")
        print("--- 自动开始训练模型，请稍候... ---")
        train_and_val(epoch_num=10, model_save_path=MODEL_PATH)  # 训练并保存
        print("--- 训练完成 ---")
    else:
        print(f"--- 发现已存在的模型 '{MODEL_PATH}' ---")
    # 2. 加载模型
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    # 3. 启动 Web 服务器
    print("\n" + "=" * 50)
    print("Flask Web 服务器正在启动...")
    print("请在浏览器中打开: http://127.0.0.1:5000/")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
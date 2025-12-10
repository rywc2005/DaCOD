# DaCOD 代码运行和使用手册

## 完整操作指南 - 从零开始到成功运行

本手册提供最详细的、一步一步的操作指南，确保您能够成功运行 DaCOD 项目。

---

## 目录

1. [系统要求检查](#系统要求检查)
2. [环境安装详细步骤](#环境安装详细步骤)
3. [下载所需文件](#下载所需文件)
4. [项目配置](#项目配置)
5. [数据准备详解](#数据准备详解)
6. [训练模型完整流程](#训练模型完整流程)
7. [测试模型完整流程](#测试模型完整流程)
8. [结果分析和可视化](#结果分析和可视化)
9. [问题排查指南](#问题排查指南)
10. [自定义数据处理](#自定义数据处理)

---

## 系统要求检查

### 第一步：检查您的硬件

```bash
# 1. 检查 GPU 是否可用
nvidia-smi

# 您应该看到类似输出：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 470.xx       Driver Version: 470.xx       CUDA Version: 11.4    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA RTX 3090    Off  | 00000000:01:00.0 Off |                  N/A |
# | 30%   45C    P8    25W / 350W |      0MiB / 24268MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

**最低要求**：
- GPU 显存：≥ 11GB (推荐 24GB)
- 系统内存：≥ 16GB
- 硬盘空间：≥ 50GB (用于数据集和模型)

### 第二步：检查 CUDA 版本

```bash
nvcc --version

# 输出示例：
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 11.3, V11.3.109
```

**支持的 CUDA 版本**：10.2 / 11.1 / 11.3 / 11.6

---

## 环境安装详细步骤

### 方法一：使用 Conda（推荐）

#### 步骤 1：安装 Miniconda/Anaconda

如果还没有安装 Conda：

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 按照提示完成安装，然后重启终端或运行：
source ~/.bashrc
```

#### 步骤 2：创建虚拟环境

```bash
# 创建名为 dacod 的环境，Python 3.8
conda create -n dacod python=3.8 -y

# 激活环境
conda activate dacod

# 确认环境已激活（命令行前会显示 (dacod)）
# (dacod) user@machine:~$
```

#### 步骤 3：安装 PyTorch

**根据您的 CUDA 版本选择相应的安装命令：**

**CUDA 11.3:**
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**CUDA 11.1:**
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

**CUDA 10.2:**
```bash
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

**验证安装：**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

# 期望输出：
# PyTorch version: 1.10.0+cu113
# CUDA available: True
# CUDA version: 11.3
```

#### 步骤 4：安装其他依赖

```bash
# 安装基础依赖
pip install pillow==9.0.0
pip install numpy==1.21.0
pip install tqdm==4.62.0
pip install tensorboard==2.8.0
pip install matplotlib==3.5.0
pip install opencv-python==4.5.5.62

# 验证安装
python -c "import PIL; import numpy; import tqdm; import tensorboard; print('All dependencies installed successfully!')"
```

### 方法二：使用 virtualenv

```bash
# 安装 virtualenv
pip install virtualenv

# 创建虚拟环境
virtualenv dacod_env

# 激活环境
source dacod_env/bin/activate

# 安装依赖（参考方法一的步骤 3 和 4）
```

---

## 下载所需文件

### 1. 克隆项目代码

```bash
# 克隆仓库
git clone https://github.com/rywc2005/DaCOD.git

# 进入项目目录
cd DaCOD

# 查看项目结构
ls -la
```

### 2. 下载预训练权重

#### 创建存放目录

```bash
# 在项目根目录下
mkdir -p backbone
mkdir -p checkpoints/Depth_cod
```

#### 下载 Swin Transformer 权重

**选项 A - 百度网盘：**
1. 访问：https://pan.baidu.com/s/1smqwGSiHr_Hw1uSGnWC70Q
2. 提取码：ksv5
3. 下载文件：`swin_large_patch4_window7_224_22k.pth`
4. 移动到项目的 `backbone/` 目录

**选项 B - Google Drive：**
1. 访问：https://drive.google.com/drive/folders/144jEAZz4ZAzWXCuYb3Zm7_mGspSJYgeo
2. 下载 `swin_large_patch4_window7_224_22k.pth`
3. 移动到项目的 `backbone/` 目录

```bash
# 移动文件（假设下载到 ~/Downloads）
mv ~/Downloads/swin_large_patch4_window7_224_22k.pth ./backbone/

# 验证文件
ls -lh backbone/swin_large_patch4_window7_224_22k.pth
# 应该显示文件大小约 755MB
```

#### 下载 ResNet50 权重

**选项 A - 百度网盘：**
1. 访问：https://pan.baidu.com/s/1FVLyuAqxnFwxusKpkVM1dg
2. 提取码：qxju
3. 下载文件：`resnet50-19c8e357.pth`

**选项 B - 直接下载：**
```bash
cd backbone
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
cd ..
```

```bash
# 验证文件
ls -lh backbone/resnet50-19c8e357.pth
# 应该显示文件大小约 98MB
```

#### 下载训练好的模型（用于测试）

**百度网盘：**
1. 访问：https://pan.baidu.com/s/1MIz-oyKxlit1jnxpYDuc1A
2. 提取码：sj8w
3. 下载文件：`55.pth`
4. 移动到 `checkpoints/Depth_cod/` 目录

```bash
# 移动文件
mv ~/Downloads/55.pth ./checkpoints/Depth_cod/

# 验证所有权重文件
ls -lh backbone/
ls -lh checkpoints/Depth_cod/
```

### 3. 下载数据集

#### 下载训练和测试数据

**百度网盘：**
1. 访问：https://pan.baidu.com/s/1dV9F-dcTauvHcQ-C90piZg
2. 提取码：hqxc
3. 下载完整数据集压缩包

**Google Drive：**
1. 访问：https://drive.google.com/drive/folders/144jEAZz4ZAzWXCuYb3Zm7_mGspSJYgeo
2. 下载数据集

#### 解压数据集

```bash
# 假设下载的是 datasets.zip
unzip datasets.zip

# 或如果是 tar.gz
tar -xzvf datasets.tar.gz

# 移动到项目目录
mv datasets ./

# 验证数据集结构
tree datasets -L 3
# 或
ls -R datasets/ | head -50
```

**期望的目录结构：**
```
datasets/
├── train/
│   └── cod10k_depth_train/
│       ├── rgb/
│       ├── depth/
│       └── gt/
└── test/
    ├── CAMO_depth/
    │   ├── rgb/
    │   ├── depth/
    │   └── gt/
    ├── cod10k_depth_test/
    │   ├── rgb/
    │   ├── depth/
    │   └── gt/
    └── NC4K/
        ├── rgb/
        ├── depth/
        └── gt/
```

---

## 项目配置

### 1. 检查配置文件

```bash
# 查看配置文件
cat utils/config.py
```

**确认路径正确：**
```python
backbone_path = './backbone/resnet50-19c8e357.pth'
datasets_root = './datasets'
depth_cod_training_root = os.path.join(datasets_root,'train/cod10k_depth_train')
# ... 其他路径
```

### 2. 验证文件完整性

创建一个简单的验证脚本：

```bash
cat > verify_setup.py << 'EOF'
import os
import torch

print("=" * 60)
print("DaCOD 环境验证")
print("=" * 60)

# 1. 检查 PyTorch 和 CUDA
print("\n1. PyTorch 环境:")
print(f"   PyTorch 版本: {torch.__version__}")
print(f"   CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   GPU 数量: {torch.cuda.device_count()}")
    print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")

# 2. 检查预训练权重
print("\n2. 预训练权重:")
swin_path = './backbone/swin_large_patch4_window7_224_22k.pth'
resnet_path = './backbone/resnet50-19c8e357.pth'
model_path = './checkpoints/Depth_cod/55.pth'

for name, path in [('Swin', swin_path), ('ResNet50', resnet_path), ('训练模型', model_path)]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✓ {name}: {size_mb:.1f} MB")
    else:
        print(f"   ✗ {name}: 未找到")

# 3. 检查数据集
print("\n3. 数据集:")
train_rgb = './datasets/train/cod10k_depth_train/rgb'
test_camo = './datasets/test/CAMO_depth/rgb'

if os.path.exists(train_rgb):
    num_train = len([f for f in os.listdir(train_rgb) if f.endswith('.jpg')])
    print(f"   ✓ 训练集: {num_train} 张图像")
else:
    print(f"   ✗ 训练集: 未找到")

if os.path.exists(test_camo):
    num_test = len([f for f in os.listdir(test_camo) if f.endswith('.jpg')])
    print(f"   ✓ 测试集(CAMO): {num_test} 张图像")
else:
    print(f"   ✗ 测试集: 未找到")

# 4. 检查依赖包
print("\n4. Python 依赖:")
required_packages = ['PIL', 'numpy', 'tqdm', 'tensorboard']
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ✗ {pkg}: 未安装")

print("\n" + "=" * 60)
print("验证完成！")
print("=" * 60)
EOF

# 运行验证
python verify_setup.py
```

---

## 数据准备详解

### 数据集统计信息

运行以下脚本了解数据集详情：

```bash
cat > dataset_info.py << 'EOF'
import os
from PIL import Image

def analyze_dataset(root_path, name):
    rgb_path = os.path.join(root_path, 'rgb')
    depth_path = os.path.join(root_path, 'depth')
    gt_path = os.path.join(root_path, 'gt')
    
    print(f"\n{name}:")
    
    if not os.path.exists(rgb_path):
        print("  数据集不存在！")
        return
    
    # 统计数量
    rgb_files = [f for f in os.listdir(rgb_path) if f.endswith('.jpg')]
    num_images = len(rgb_files)
    print(f"  图像数量: {num_images}")
    
    # 检查第一张图像的尺寸
    if num_images > 0:
        sample_img = Image.open(os.path.join(rgb_path, rgb_files[0]))
        print(f"  示例图像尺寸: {sample_img.size}")
    
    # 检查文件完整性
    missing = 0
    for img_name in rgb_files[:10]:  # 检查前10个
        base_name = os.path.splitext(img_name)[0]
        depth_file = os.path.join(depth_path, base_name + '.png')
        gt_file = os.path.join(gt_path, base_name + '.png')
        if not os.path.exists(depth_file) or not os.path.exists(gt_file):
            missing += 1
    
    if missing > 0:
        print(f"  ⚠️ 前10个样本中有 {missing} 个缺少深度图或标注")
    else:
        print(f"  ✓ 数据完整")

# 分析所有数据集
print("=" * 60)
print("数据集分析")
print("=" * 60)

analyze_dataset('./datasets/train/cod10k_depth_train', '训练集 (COD10K-Train)')
analyze_dataset('./datasets/test/CAMO_depth', '测试集 (CAMO)')
analyze_dataset('./datasets/test/cod10k_depth_test', '测试集 (COD10K-Test)')
analyze_dataset('./datasets/test/NC4K', '测试集 (NC4K)')

print("\n" + "=" * 60)
EOF

python dataset_info.py
```

### 数据预览

创建脚本预览数据：

```bash
cat > preview_data.py << 'EOF'
import os
import matplotlib.pyplot as plt
from PIL import Image

# 加载一个样本
rgb_path = './datasets/train/cod10k_depth_train/rgb'
depth_path = './datasets/train/cod10k_depth_train/depth'
gt_path = './datasets/train/cod10k_depth_train/gt'

# 获取第一个文件
files = [f for f in os.listdir(rgb_path) if f.endswith('.jpg')]
if len(files) == 0:
    print("未找到训练图像！")
    exit(1)

sample_file = files[0]
base_name = os.path.splitext(sample_file)[0]

# 加载图像
rgb = Image.open(os.path.join(rgb_path, sample_file))
depth = Image.open(os.path.join(depth_path, base_name + '.png'))
gt = Image.open(os.path.join(gt_path, base_name + '.png'))

# 显示
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgb)
axes[0].set_title('RGB Image')
axes[0].axis('off')

axes[1].imshow(depth)
axes[1].set_title('Depth Map')
axes[1].axis('off')

axes[2].imshow(gt, cmap='gray')
axes[2].set_title('Ground Truth')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('data_preview.png', dpi=150, bbox_inches='tight')
print(f"数据预览已保存到: data_preview.png")
print(f"样本文件: {sample_file}")
print(f"RGB 尺寸: {rgb.size}")
print(f"深度图尺寸: {depth.size}")
print(f"标注尺寸: {gt.size}")
EOF

python preview_data.py
```

---

## 训练模型完整流程

### 准备工作

1. **确认 GPU 可用：**
```bash
python -c "import torch; print('GPU 可用:', torch.cuda.is_available())"
```

2. **创建日志和检查点目录：**
```bash
mkdir -p checkpoints/Depth_cod
mkdir -p /root/tf-logs/new2/Depth_cod/log
# 如果 /root 目录无权限，可以修改 train.py 中的路径
```

3. **修改训练参数（可选）：**

编辑 `train.py` 文件：

```bash
# 打开编辑器
nano train.py
# 或
vim train.py
```

找到 `args` 字典，根据需要修改：

```python
args = {
    'epoch_num': 60,                # 总训练轮数
    'train_batch_size': 10,         # 批次大小（显存不足可改为 5 或 8）
    'lr': 1e-3,                     # 学习率
    'lr_decay': 0.9,                # 学习率衰减
    'weight_decay': 5e-4,           # 权重衰减
    'momentum': 0.9,                # 动量
    'scale': 448,                   # 输入图像尺寸
    'poly_train': True,             # 使用 poly 学习率
    'optimizer': 'SGD',             # 优化器 ('SGD' 或 'Adam')
}
```

**如果显存不足：**
```python
args['train_batch_size'] = 5  # 减小批次
```

**快速测试训练（只训练几轮）：**
```python
args['epoch_num'] = 2  # 只训练 2 轮用于测试
```

### 开始训练

#### 方法 1：直接训练

```bash
# 激活环境
conda activate dacod

# 开始训练
python train.py
```

#### 方法 2：后台训练（推荐用于长时间训练）

```bash
# 使用 nohup 后台运行
nohup python train.py > training.log 2>&1 &

# 获取进程 ID
echo $!

# 查看训练日志
tail -f training.log

# 实时监控 GPU 使用情况（另开一个终端）
watch -n 1 nvidia-smi
```

#### 方法 3：使用 screen（防止 SSH 断线）

```bash
# 创建 screen 会话
screen -S dacod_train

# 在 screen 中运行训练
conda activate dacod
python train.py

# 分离 screen: Ctrl+A 然后按 D

# 重新连接
screen -r dacod_train

# 列出所有 screen
screen -ls
```

### 训练过程监控

#### 1. 命令行输出

训练时会看到如下输出：
```
Train set: 3040
Epoch  1/60
[  1,     10, 0.002000, 1.2345, 0.2345, 0.2456, 0.2567, 0.2678, 0.2299]: 100%|██████████| 304/304
[  1,     20, 0.001998, 1.1234, 0.2123, 0.2234, 0.2345, 0.2456, 0.2076]: 100%|██████████| 304/304
...
```

**输出说明：**
- `[1, 10, ...]`: [轮次, 迭代次数, 学习率, 总损失, loss1, loss2, loss3, loss4, loss5]
- 进度条显示当前批次处理进度

#### 2. TensorBoard 可视化

```bash
# 在新终端启动 TensorBoard
tensorboard --logdir=/root/tf-logs/new2/Depth_cod/log --port=6006

# 浏览器访问
# http://localhost:6006
# 或 http://your-server-ip:6006
```

**如果端口被占用：**
```bash
tensorboard --logdir=/root/tf-logs/new2/Depth_cod/log --port=6007
```

#### 3. 检查保存的模型

```bash
# 查看已保存的检查点
ls -lh checkpoints/Depth_cod/

# 训练过程中会保存：
# 45.pth, 50.pth, 55.pth, 60.pth 等
```

### 从检查点恢复训练

如果训练中断，可以从保存的检查点恢复：

**步骤 1：** 编辑 `train.py`

```python
args = {
    # ... 其他参数
    'last_epoch': 50,           # 从第 50 轮继续
    'snapshot': '50',           # 加载 50.pth
}
```

**步骤 2：** 重新开始训练

```bash
python train.py
```

程序会加载 `checkpoints/Depth_cod/50.pth` 并从第 51 轮继续训练。

### 训练完成

训练完成后会输出：

```
Total Training Time: 11:32:15
Depth_cod
Optimization Have Done!
```

最终模型保存在：`checkpoints/Depth_cod/60.pth`

---

## 测试模型完整流程

### 使用预训练模型测试

#### 步骤 1：准备测试环境

```bash
# 确认预训练模型存在
ls -lh checkpoints/Depth_cod/55.pth

# 创建结果目录
mkdir -p results/Depth_cod
```

#### 步骤 2：检查测试配置

查看 `infer.py` 文件：

```bash
cat infer.py | grep -A 5 "to_test"
```

应该看到：
```python
to_test = OrderedDict([
    ('CAMO_depth', camo_path),
    ('cod10k_depth_test', cod10k_path),
    ('NC4K', NC4K_path)
])
```

#### 步骤 3：运行测试

```bash
# 激活环境
conda activate dacod

# 运行测试
python infer.py
```

**输出示例：**
```
1.10.0+cu113
Load 55.pth succeed!
Processing CAMO_depth: 100%|████████████████| 250/250 [00:30<00:00,  8.23it/s]
Depth_cod
CAMO_depth's average Time Is : 0.034 s
CAMO_depth's average Time Is : 29.4 fps

Processing cod10k_depth_test: 100%|████| 2026/2026 [02:15<00:00, 14.95it/s]
Depth_cod
cod10k_depth_test's average Time Is : 0.035 s
cod10k_depth_test's average Time Is : 28.6 fps

Processing NC4K: 100%|████████████████████| 4121/4121 [04:40<00:00, 14.70it/s]
Depth_cod
NC4K's average Time Is : 0.036 s
NC4K's average Time Is : 27.8 fps

Total Testing Time: 0:07:25
```

#### 步骤 4：查看结果

```bash
# 查看生成的预测结果
ls -lh results/Depth_cod/CAMO_depth/ | head -10

# 统计生成的预测图数量
find results/Depth_cod/ -name "*.png" | wc -l
```

### 测试自己训练的模型

如果要测试自己训练的模型：

**步骤 1：** 修改 `infer.py`

```python
# 找到这一行（大约第 57 行）
net.load_state_dict(torch.load('checkpoints/Depth_cod/55.pth'))

# 改为
net.load_state_dict(torch.load('checkpoints/Depth_cod/60.pth'))  # 使用你的模型
```

**步骤 2：** 运行测试

```bash
python infer.py
```

### 只测试特定数据集

修改 `infer.py` 中的 `to_test`：

```python
# 只测试 CAMO 数据集
to_test = OrderedDict([
    ('CAMO_depth', camo_path)
])
```

---

## 结果分析和可视化

### 1. 快速可视化单个结果

```bash
cat > visualize_single.py << 'EOF'
import matplotlib.pyplot as plt
from PIL import Image
import sys

# 使用方法: python visualize_single.py image_name
# 例如: python visualize_single.py camourflage_00001

if len(sys.argv) > 1:
    img_name = sys.argv[1]
else:
    # 默认使用第一张图
    import os
    rgb_dir = './datasets/test/CAMO_depth/rgb'
    files = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
    if len(files) == 0:
        print("未找到测试图像！")
        exit(1)
    img_name = os.path.splitext(files[0])[0]

print(f"可视化: {img_name}")

# 加载图像
try:
    rgb = Image.open(f'./datasets/test/CAMO_depth/rgb/{img_name}.jpg')
    depth = Image.open(f'./datasets/test/CAMO_depth/depth/{img_name}.png')
    gt = Image.open(f'./datasets/test/CAMO_depth/gt/{img_name}.png')
    pred = Image.open(f'./results/Depth_cod/CAMO_depth/{img_name}.png')
    
    # 创建图像
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(depth)
    axes[0, 1].set_title('Depth Map', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gt, cmap='gray')
    axes[1, 0].set_title('Ground Truth', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred, cmap='gray')
    axes[1, 1].set_title('Prediction', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_name = f'{img_name}_visualization.png'
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_name}")
    
except FileNotFoundError as e:
    print(f"错误: {e}")
    print("请确保已运行测试脚本生成预测结果")
EOF

# 运行可视化
python visualize_single.py
```

### 2. 批量可视化

```bash
cat > visualize_batch.py << 'EOF'
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

# 从测试集随机选择 9 个样本
rgb_dir = './datasets/test/CAMO_depth/rgb'
files = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
samples = random.sample(files, min(9, len(files)))

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Prediction Results (Random Samples)', fontsize=16)

for idx, img_file in enumerate(samples):
    img_name = os.path.splitext(img_file)[0]
    
    try:
        rgb = Image.open(f'./datasets/test/CAMO_depth/rgb/{img_name}.jpg')
        pred = Image.open(f'./results/Depth_cod/CAMO_depth/{img_name}.png')
        
        # 在 RGB 上叠加预测结果
        from PIL import ImageDraw, ImageFont
        rgb_copy = rgb.copy()
        
        row = idx // 3
        col = idx % 3
        
        # 显示原图和预测
        axes[row, col].imshow(rgb)
        axes[row, col].imshow(pred, alpha=0.5, cmap='hot')
        axes[row, col].set_title(f'{img_name[:15]}...', fontsize=10)
        axes[row, col].axis('off')
        
    except Exception as e:
        print(f"处理 {img_name} 时出错: {e}")

plt.tight_layout()
plt.savefig('batch_visualization.png', dpi=150, bbox_inches='tight')
print("批量可视化已保存到: batch_visualization.png")
EOF

python visualize_batch.py
```

### 3. 评估指标计算

安装评估工具：

```bash
pip install pysodmetrics
```

创建评估脚本：

```bash
cat > evaluate.py << 'EOF'
import os
import numpy as np
from PIL import Image
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

def evaluate_dataset(pred_dir, gt_dir, dataset_name):
    print(f"\n评估 {dataset_name}:")
    print("-" * 50)
    
    # 初始化评估指标
    mae_metric = MAE()
    em_metric = Emeasure()
    fm_metric = Fmeasure()
    sm_metric = Smeasure()
    wfm_metric = WeightedFmeasure()
    
    # 获取所有预测文件
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    
    for pred_file in pred_files:
        # 加载预测和真值
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, pred_file)
        
        if not os.path.exists(gt_path):
            continue
        
        pred = np.array(Image.open(pred_path).convert('L'))
        gt = np.array(Image.open(gt_path).convert('L'))
        
        # 归一化到 [0, 1]
        pred = pred / 255.0
        gt = gt / 255.0
        
        # 更新指标
        mae_metric.step(pred, gt)
        em_metric.step(pred, gt)
        fm_metric.step(pred, gt)
        sm_metric.step(pred, gt)
        wfm_metric.step(pred, gt)
    
    # 获取结果
    mae = mae_metric.get_results()['mae']
    em = em_metric.get_results()['em']
    fm = fm_metric.get_results()['fm']
    sm = sm_metric.get_results()['sm']
    wfm = wfm_metric.get_results()['wfm']
    
    print(f"MAE: {mae:.4f}")
    print(f"E-measure: {em['curve'].mean():.4f}")
    print(f"F-measure: {fm['curve'].mean():.4f}")
    print(f"S-measure: {sm:.4f}")
    print(f"Weighted F-measure: {wfm:.4f}")
    
    return {
        'mae': mae,
        'em': em['curve'].mean(),
        'fm': fm['curve'].mean(),
        'sm': sm,
        'wfm': wfm
    }

# 评估所有数据集
datasets = [
    ('CAMO', './results/Depth_cod/CAMO_depth', './datasets/test/CAMO_depth/gt'),
    ('COD10K', './results/Depth_cod/cod10k_depth_test', './datasets/test/cod10k_depth_test/gt'),
    ('NC4K', './results/Depth_cod/NC4K', './datasets/test/NC4K/gt')
]

results = {}
for name, pred_dir, gt_dir in datasets:
    if os.path.exists(pred_dir) and os.path.exists(gt_dir):
        results[name] = evaluate_dataset(pred_dir, gt_dir, name)

# 打印总结
print("\n" + "=" * 50)
print("评估总结")
print("=" * 50)
print(f"{'数据集':<15} {'MAE':<10} {'Fβ':<10} {'Sm':<10} {'Em':<10}")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:<15} {metrics['mae']:<10.4f} {metrics['fm']:<10.4f} "
          f"{metrics['sm']:<10.4f} {metrics['em']:<10.4f}")
EOF

python evaluate.py
```

---

## 问题排查指南

### 问题 1：CUDA out of memory（显存不足）

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案：**

1. **减小批次大小**（最简单）：
```python
# 在 train.py 中
args['train_batch_size'] = 5  # 从 10 改为 5
```

2. **减小输入尺寸**：
```python
args['scale'] = 352  # 从 448 改为 352
```

3. **清理 GPU 缓存**：
```python
import torch
torch.cuda.empty_cache()
```

### 问题 2：找不到模块

**症状：**
```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案：**
```bash
# 确认环境已激活
conda activate dacod

# 重新安装缺失的包
pip install xxx
```

### 问题 3：预训练权重加载失败

**症状：**
```
FileNotFoundError: [Errno 2] No such file or directory: './backbone/xxx.pth'
```

**解决方案：**
```bash
# 检查文件是否存在
ls -lh backbone/

# 检查文件名是否匹配
# 确保文件名完全一致，包括大小写
```

### 问题 4：数据集路径错误

**症状：**
```
FileNotFoundError: [Errno 2] No such file or directory: './datasets/train/...'
```

**解决方案：**
```bash
# 检查数据集目录结构
tree datasets -L 3

# 确认路径配置
cat utils/config.py
```

### 问题 5：训练过程中断

**恢复方法：**

1. 找到最后保存的检查点：
```bash
ls -lt checkpoints/Depth_cod/
```

2. 修改 `train.py`：
```python
args['last_epoch'] = 50      # 最后完成的轮次
args['snapshot'] = '50'       # 对应的检查点文件
```

3. 重新开始训练：
```bash
python train.py
```

### 问题 6：TensorBoard 无法访问

**解决方案：**

1. **检查端口是否被占用**：
```bash
lsof -i :6006
```

2. **更换端口**：
```bash
tensorboard --logdir=/root/tf-logs/new2/Depth_cod/log --port=6007
```

3. **SSH 端口转发**（远程服务器）：
```bash
# 在本地机器运行
ssh -L 6006:localhost:6006 user@remote-server
# 然后在浏览器访问 http://localhost:6006
```

### 问题 7：推理速度慢

**优化方法：**

1. **确认使用 GPU**：
```python
# 在 infer.py 开头添加
import torch
print(f"Using device: {torch.cuda.get_device_name(0)}")
```

2. **减小图像尺寸**：
```python
# 在 infer.py 中
args['scale'] = 352  # 从 448 减小
```

3. **使用半精度推理**：
```python
net = net.half()  # 在模型加载后
```

---

## 自定义数据处理

### 使用自己的数据集进行测试

#### 步骤 1：准备数据

创建目录结构：
```bash
mkdir -p my_dataset/rgb
mkdir -p my_dataset/depth
mkdir -p my_dataset/gt  # 如果有标注
```

#### 步骤 2：生成深度图

如果没有深度图，使用 MiDaS 生成：

```bash
# 克隆 MiDaS
git clone https://github.com/isl-org/MiDaS.git
cd MiDaS

# 下载模型
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt

# 批量生成深度图
python run.py --model_type dpt_beit_large_512 \
              --input_path ../my_dataset/rgb \
              --output_path ../my_dataset/depth
cd ..
```

#### 步骤 3：修改配置

编辑 `utils/config.py`：
```python
# 添加你的数据集路径
my_dataset_path = './my_dataset'
```

编辑 `infer.py`：
```python
to_test = OrderedDict([
    ('my_dataset', my_dataset_path)
])
```

#### 步骤 4：运行测试

```bash
python infer.py
```

结果保存在：`results/Depth_cod/my_dataset/`

### 在自己的数据上训练

#### 步骤 1：准备训练数据

确保数据结构：
```
my_train_data/
├── rgb/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── depth/
│   ├── img001.png
│   ├── img002.png
│   └── ...
└── gt/
    ├── img001.png
    ├── img002.png
    └── ...
```

**注意：**
- RGB 图像必须是 `.jpg` 格式
- 深度图和标注必须是 `.png` 格式
- 文件名（不含扩展名）必须完全一致

#### 步骤 2：修改配置

编辑 `utils/config.py`：
```python
depth_cod_training_root = './my_train_data'
```

#### 步骤 3：开始训练

```bash
python train.py
```

---

## 附录：完整命令速查表

### 环境相关
```bash
# 激活环境
conda activate dacod

# 验证环境
python verify_setup.py

# 查看 GPU 状态
nvidia-smi
watch -n 1 nvidia-smi
```

### 训练相关
```bash
# 开始训练
python train.py

# 后台训练
nohup python train.py > training.log 2>&1 &

# 查看训练日志
tail -f training.log

# 启动 TensorBoard
tensorboard --logdir=/root/tf-logs/new2/Depth_cod/log --port=6006
```

### 测试相关
```bash
# 运行测试
python infer.py

# 查看结果
ls results/Depth_cod/

# 可视化结果
python visualize_single.py [image_name]
python visualize_batch.py
```

### 评估相关
```bash
# 安装评估工具
pip install pysodmetrics

# 运行评估
python evaluate.py
```

### 数据相关
```bash
# 查看数据集信息
python dataset_info.py

# 预览数据
python preview_data.py

# 检查数据完整性
find datasets -name "*.jpg" | wc -l
find datasets -name "*.png" | wc -l
```

---

## 常见使用场景

### 场景 1：快速测试（不训练）

```bash
# 1. 下载预训练模型 55.pth
# 2. 准备测试数据
# 3. 运行测试
conda activate dacod
python infer.py
```

### 场景 2：从头训练新模型

```bash
# 1. 准备完整训练数据
# 2. 下载预训练骨干网络
# 3. 开始训练
conda activate dacod
python train.py
```

### 场景 3：在自定义数据上测试

```bash
# 1. 准备数据（RGB + 深度图）
# 2. 修改 infer.py 配置
# 3. 运行测试
python infer.py
```

### 场景 4：评估模型性能

```bash
# 1. 运行测试生成预测
python infer.py

# 2. 运行评估脚本
python evaluate.py
```

---

## 获取帮助

如果遇到问题：

1. **查看日志文件**：检查 `training.log` 或命令行输出
2. **运行验证脚本**：`python verify_setup.py`
3. **检查 GitHub Issues**：https://github.com/rywc2005/DaCOD/issues
4. **查看论文**：了解方法细节

---

**文档版本**: v1.0  
**最后更新**: 2024年  
**适用于**: DaCOD 项目

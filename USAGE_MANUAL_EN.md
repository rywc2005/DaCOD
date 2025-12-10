# DaCOD Code Running and Usage Manual

## Complete Step-by-Step Guide - From Scratch to Successful Execution

This manual provides the most detailed, step-by-step instructions to ensure you can successfully run the DaCOD project.

---

## Table of Contents

1. [System Requirements Check](#system-requirements-check)
2. [Detailed Environment Installation](#detailed-environment-installation)
3. [Download Required Files](#download-required-files)
4. [Project Configuration](#project-configuration)
5. [Data Preparation Guide](#data-preparation-guide)
6. [Complete Training Workflow](#complete-training-workflow)
7. [Complete Testing Workflow](#complete-testing-workflow)
8. [Results Analysis and Visualization](#results-analysis-and-visualization)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Custom Data Processing](#custom-data-processing)

---

## System Requirements Check

### Step 1: Check Your Hardware

```bash
# 1. Check if GPU is available
nvidia-smi

# You should see output like:
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

**Minimum Requirements**:
- GPU VRAM: ≥ 11GB (24GB recommended)
- System RAM: ≥ 16GB
- Disk Space: ≥ 50GB (for datasets and models)

### Step 2: Check CUDA Version

```bash
nvcc --version

# Example output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 11.3, V11.3.109
```

**Supported CUDA Versions**: 10.2 / 11.1 / 11.3 / 11.6

---

## Detailed Environment Installation

### Method 1: Using Conda (Recommended)

#### Step 1: Install Miniconda/Anaconda

If you don't have Conda installed:

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts to complete installation, then restart terminal or run:
source ~/.bashrc
```

#### Step 2: Create Virtual Environment

```bash
# Create environment named dacod with Python 3.8
conda create -n dacod python=3.8 -y

# Activate environment
conda activate dacod

# Confirm environment is activated (you'll see (dacod) in prompt)
# (dacod) user@machine:~$
```

#### Step 3: Install PyTorch

**Choose installation command based on your CUDA version:**

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

**Verify Installation:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

# Expected output:
# PyTorch version: 1.10.0+cu113
# CUDA available: True
# CUDA version: 11.3
```

#### Step 4: Install Other Dependencies

```bash
# Install basic dependencies
pip install pillow==9.0.0
pip install numpy==1.21.0
pip install tqdm==4.62.0
pip install tensorboard==2.8.0
pip install matplotlib==3.5.0
pip install opencv-python==4.5.5.62

# Verify installation
python -c "import PIL; import numpy; import tqdm; import tensorboard; print('All dependencies installed successfully!')"
```

### Method 2: Using virtualenv

```bash
# Install virtualenv
pip install virtualenv

# Create virtual environment
virtualenv dacod_env

# Activate environment
source dacod_env/bin/activate

# Install dependencies (refer to Method 1 steps 3 and 4)
```

---

## Download Required Files

### 1. Clone Project Code

```bash
# Clone repository
git clone https://github.com/rywc2005/DaCOD.git

# Enter project directory
cd DaCOD

# View project structure
ls -la
```

### 2. Download Pre-trained Weights

#### Create Storage Directories

```bash
# In project root directory
mkdir -p backbone
mkdir -p checkpoints/Depth_cod
```

#### Download Swin Transformer Weights

**Option A - Baidu Pan:**
1. Visit: https://pan.baidu.com/s/1smqwGSiHr_Hw1uSGnWC70Q
2. Code: ksv5
3. Download file: `swin_large_patch4_window7_224_22k.pth`
4. Move to project's `backbone/` directory

**Option B - Google Drive:**
1. Visit: https://drive.google.com/drive/folders/144jEAZz4ZAzWXCuYb3Zm7_mGspSJYgeo
2. Download `swin_large_patch4_window7_224_22k.pth`
3. Move to project's `backbone/` directory

```bash
# Move file (assuming downloaded to ~/Downloads)
mv ~/Downloads/swin_large_patch4_window7_224_22k.pth ./backbone/

# Verify file
ls -lh backbone/swin_large_patch4_window7_224_22k.pth
# Should show file size ~755MB
```

#### Download ResNet50 Weights

**Option A - Baidu Pan:**
1. Visit: https://pan.baidu.com/s/1FVLyuAqxnFwxusKpkVM1dg
2. Code: qxju
3. Download file: `resnet50-19c8e357.pth`

**Option B - Direct Download:**
```bash
cd backbone
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
cd ..
```

```bash
# Verify file
ls -lh backbone/resnet50-19c8e357.pth
# Should show file size ~98MB
```

#### Download Trained Model (for testing)

**Baidu Pan:**
1. Visit: https://pan.baidu.com/s/1MIz-oyKxlit1jnxpYDuc1A
2. Code: sj8w
3. Download file: `55.pth`
4. Move to `checkpoints/Depth_cod/` directory

```bash
# Move file
mv ~/Downloads/55.pth ./checkpoints/Depth_cod/

# Verify all weight files
ls -lh backbone/
ls -lh checkpoints/Depth_cod/
```

### 3. Download Datasets

#### Download Training and Test Data

**Baidu Pan:**
1. Visit: https://pan.baidu.com/s/1dV9F-dcTauvHcQ-C90piZg
2. Code: hqxc
3. Download complete dataset archive

**Google Drive:**
1. Visit: https://drive.google.com/drive/folders/144jEAZz4ZAzWXCuYb3Zm7_mGspSJYgeo
2. Download datasets

#### Extract Dataset

```bash
# Assuming downloaded datasets.zip
unzip datasets.zip

# Or if tar.gz
tar -xzvf datasets.tar.gz

# Move to project directory
mv datasets ./

# Verify dataset structure
tree datasets -L 3
# Or
ls -R datasets/ | head -50
```

**Expected Directory Structure:**
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

## Project Configuration

### 1. Check Configuration File

```bash
# View configuration file
cat utils/config.py
```

**Confirm paths are correct:**
```python
backbone_path = './backbone/resnet50-19c8e357.pth'
datasets_root = './datasets'
depth_cod_training_root = os.path.join(datasets_root,'train/cod10k_depth_train')
# ... other paths
```

### 2. Verify File Integrity

Create a simple verification script:

```bash
cat > verify_setup.py << 'EOF'
import os
import torch

print("=" * 60)
print("DaCOD Environment Verification")
print("=" * 60)

# 1. Check PyTorch and CUDA
print("\n1. PyTorch Environment:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")

# 2. Check pre-trained weights
print("\n2. Pre-trained Weights:")
swin_path = './backbone/swin_large_patch4_window7_224_22k.pth'
resnet_path = './backbone/resnet50-19c8e357.pth'
model_path = './checkpoints/Depth_cod/55.pth'

for name, path in [('Swin', swin_path), ('ResNet50', resnet_path), ('Trained Model', model_path)]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✓ {name}: {size_mb:.1f} MB")
    else:
        print(f"   ✗ {name}: Not found")

# 3. Check datasets
print("\n3. Datasets:")
train_rgb = './datasets/train/cod10k_depth_train/rgb'
test_camo = './datasets/test/CAMO_depth/rgb'

if os.path.exists(train_rgb):
    num_train = len([f for f in os.listdir(train_rgb) if f.endswith('.jpg')])
    print(f"   ✓ Training set: {num_train} images")
else:
    print(f"   ✗ Training set: Not found")

if os.path.exists(test_camo):
    num_test = len([f for f in os.listdir(test_camo) if f.endswith('.jpg')])
    print(f"   ✓ Test set (CAMO): {num_test} images")
else:
    print(f"   ✗ Test set: Not found")

# 4. Check dependencies
print("\n4. Python Dependencies:")
required_packages = ['PIL', 'numpy', 'tqdm', 'tensorboard']
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ✗ {pkg}: Not installed")

print("\n" + "=" * 60)
print("Verification complete!")
print("=" * 60)
EOF

# Run verification
python verify_setup.py
```

---

## Data Preparation Guide

### Dataset Statistics

Run the following script to understand dataset details:

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
        print("  Dataset does not exist!")
        return
    
    # Count files
    rgb_files = [f for f in os.listdir(rgb_path) if f.endswith('.jpg')]
    num_images = len(rgb_files)
    print(f"  Number of images: {num_images}")
    
    # Check first image size
    if num_images > 0:
        sample_img = Image.open(os.path.join(rgb_path, rgb_files[0]))
        print(f"  Sample image size: {sample_img.size}")
    
    # Check file integrity
    missing = 0
    for img_name in rgb_files[:10]:  # Check first 10
        base_name = os.path.splitext(img_name)[0]
        depth_file = os.path.join(depth_path, base_name + '.png')
        gt_file = os.path.join(gt_path, base_name + '.png')
        if not os.path.exists(depth_file) or not os.path.exists(gt_file):
            missing += 1
    
    if missing > 0:
        print(f"  ⚠️ {missing} out of first 10 samples missing depth or GT")
    else:
        print(f"  ✓ Data integrity check passed")

# Analyze all datasets
print("=" * 60)
print("Dataset Analysis")
print("=" * 60)

analyze_dataset('./datasets/train/cod10k_depth_train', 'Training Set (COD10K-Train)')
analyze_dataset('./datasets/test/CAMO_depth', 'Test Set (CAMO)')
analyze_dataset('./datasets/test/cod10k_depth_test', 'Test Set (COD10K-Test)')
analyze_dataset('./datasets/test/NC4K', 'Test Set (NC4K)')

print("\n" + "=" * 60)
EOF

python dataset_info.py
```

### Data Preview

Create script to preview data:

```bash
cat > preview_data.py << 'EOF'
import os
import matplotlib.pyplot as plt
from PIL import Image

# Load a sample
rgb_path = './datasets/train/cod10k_depth_train/rgb'
depth_path = './datasets/train/cod10k_depth_train/depth'
gt_path = './datasets/train/cod10k_depth_train/gt'

# Get first file
files = [f for f in os.listdir(rgb_path) if f.endswith('.jpg')]
if len(files) == 0:
    print("No training images found!")
    exit(1)

sample_file = files[0]
base_name = os.path.splitext(sample_file)[0]

# Load images
rgb = Image.open(os.path.join(rgb_path, sample_file))
depth = Image.open(os.path.join(depth_path, base_name + '.png'))
gt = Image.open(os.path.join(gt_path, base_name + '.png'))

# Display
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
print(f"Data preview saved to: data_preview.png")
print(f"Sample file: {sample_file}")
print(f"RGB size: {rgb.size}")
print(f"Depth size: {depth.size}")
print(f"GT size: {gt.size}")
EOF

python preview_data.py
```

---

## Complete Training Workflow

### Preparation

1. **Confirm GPU is available:**
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

2. **Create log and checkpoint directories:**
```bash
mkdir -p checkpoints/Depth_cod
mkdir -p /root/tf-logs/new2/Depth_cod/log
# If /root directory has no permission, modify paths in train.py
```

3. **Modify training parameters (optional):**

Edit `train.py` file:

```bash
# Open editor
nano train.py
# Or
vim train.py
```

Find `args` dictionary and modify as needed:

```python
args = {
    'epoch_num': 60,                # Total training epochs
    'train_batch_size': 10,         # Batch size (reduce to 5 or 8 if OOM)
    'lr': 1e-3,                     # Learning rate
    'lr_decay': 0.9,                # Learning rate decay
    'weight_decay': 5e-4,           # Weight decay
    'momentum': 0.9,                # Momentum
    'scale': 448,                   # Input image size
    'poly_train': True,             # Use poly learning rate
    'optimizer': 'SGD',             # Optimizer ('SGD' or 'Adam')
}
```

**If out of memory:**
```python
args['train_batch_size'] = 5  # Reduce batch size
```

**Quick test training (few epochs):**
```python
args['epoch_num'] = 2  # Train only 2 epochs for testing
```

### Start Training

#### Method 1: Direct Training

```bash
# Activate environment
conda activate dacod

# Start training
python train.py
```

#### Method 2: Background Training (recommended for long training)

```bash
# Run in background with nohup
nohup python train.py > training.log 2>&1 &

# Get process ID
echo $!

# View training log
tail -f training.log

# Monitor GPU usage (in another terminal)
watch -n 1 nvidia-smi
```

#### Method 3: Using screen (prevent SSH disconnection)

```bash
# Create screen session
screen -S dacod_train

# Run training in screen
conda activate dacod
python train.py

# Detach screen: Ctrl+A then press D

# Reattach
screen -r dacod_train

# List all screens
screen -ls
```

### Training Monitoring

#### 1. Command Line Output

During training you'll see output like:
```
Train set: 3040
Epoch  1/60
[  1,     10, 0.002000, 1.2345, 0.2345, 0.2456, 0.2567, 0.2678, 0.2299]: 100%|██████████| 304/304
[  1,     20, 0.001998, 1.1234, 0.2123, 0.2234, 0.2345, 0.2456, 0.2076]: 100%|██████████| 304/304
...
```

**Output explanation:**
- `[1, 10, ...]`: [epoch, iteration, learning rate, total loss, loss1, loss2, loss3, loss4, loss5]
- Progress bar shows current batch processing progress

#### 2. TensorBoard Visualization

```bash
# Start TensorBoard in new terminal
tensorboard --logdir=/root/tf-logs/new2/Depth_cod/log --port=6006

# Access in browser
# http://localhost:6006
# Or http://your-server-ip:6006
```

**If port is occupied:**
```bash
tensorboard --logdir=/root/tf-logs/new2/Depth_cod/log --port=6007
```

#### 3. Check Saved Models

```bash
# View saved checkpoints
ls -lh checkpoints/Depth_cod/

# During training, saves:
# 45.pth, 50.pth, 55.pth, 60.pth, etc.
```

### Resume Training from Checkpoint

If training is interrupted, resume from saved checkpoint:

**Step 1:** Edit `train.py`

```python
args = {
    # ... other parameters
    'last_epoch': 50,           # Continue from epoch 50
    'snapshot': '50',           # Load 50.pth
}
```

**Step 2:** Restart training

```bash
python train.py
```

Program will load `checkpoints/Depth_cod/50.pth` and continue from epoch 51.

### Training Complete

When training completes, you'll see:

```
Total Training Time: 11:32:15
Depth_cod
Optimization Have Done!
```

Final model saved at: `checkpoints/Depth_cod/60.pth`

---

## Complete Testing Workflow

### Testing with Pre-trained Model

#### Step 1: Prepare Test Environment

```bash
# Confirm pre-trained model exists
ls -lh checkpoints/Depth_cod/55.pth

# Create results directory
mkdir -p results/Depth_cod
```

#### Step 2: Check Test Configuration

View `infer.py` file:

```bash
cat infer.py | grep -A 5 "to_test"
```

Should see:
```python
to_test = OrderedDict([
    ('CAMO_depth', camo_path),
    ('cod10k_depth_test', cod10k_path),
    ('NC4K', NC4K_path)
])
```

#### Step 3: Run Testing

```bash
# Activate environment
conda activate dacod

# Run testing
python infer.py
```

**Example output:**
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

#### Step 4: View Results

```bash
# View generated predictions
ls -lh results/Depth_cod/CAMO_depth/ | head -10

# Count generated prediction images
find results/Depth_cod/ -name "*.png" | wc -l
```

### Testing Your Own Trained Model

To test your own trained model:

**Step 1:** Modify `infer.py`

```python
# Find this line (around line 57)
net.load_state_dict(torch.load('checkpoints/Depth_cod/55.pth'))

# Change to
net.load_state_dict(torch.load('checkpoints/Depth_cod/60.pth'))  # Use your model
```

**Step 2:** Run testing

```bash
python infer.py
```

### Test Only Specific Dataset

Modify `to_test` in `infer.py`:

```python
# Test only CAMO dataset
to_test = OrderedDict([
    ('CAMO_depth', camo_path)
])
```

---

## Results Analysis and Visualization

### 1. Quick Visualize Single Result

```bash
cat > visualize_single.py << 'EOF'
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Usage: python visualize_single.py image_name
# Example: python visualize_single.py camourflage_00001

if len(sys.argv) > 1:
    img_name = sys.argv[1]
else:
    # Default to first image
    import os
    rgb_dir = './datasets/test/CAMO_depth/rgb'
    files = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]
    if len(files) == 0:
        print("No test images found!")
        exit(1)
    img_name = os.path.splitext(files[0])[0]

print(f"Visualizing: {img_name}")

# Load images
try:
    rgb = Image.open(f'./datasets/test/CAMO_depth/rgb/{img_name}.jpg')
    depth = Image.open(f'./datasets/test/CAMO_depth/depth/{img_name}.png')
    gt = Image.open(f'./datasets/test/CAMO_depth/gt/{img_name}.png')
    pred = Image.open(f'./results/Depth_cod/CAMO_depth/{img_name}.png')
    
    # Create figure
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
    print(f"Visualization saved to: {output_name}")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure you've run the test script to generate predictions")
EOF

# Run visualization
python visualize_single.py
```

### 2. Batch Visualization

```bash
cat > visualize_batch.py << 'EOF'
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

# Randomly select 9 samples from test set
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
        
        # Overlay prediction on RGB
        from PIL import ImageDraw, ImageFont
        rgb_copy = rgb.copy()
        
        row = idx // 3
        col = idx % 3
        
        # Display original and prediction
        axes[row, col].imshow(rgb)
        axes[row, col].imshow(pred, alpha=0.5, cmap='hot')
        axes[row, col].set_title(f'{img_name[:15]}...', fontsize=10)
        axes[row, col].axis('off')
        
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

plt.tight_layout()
plt.savefig('batch_visualization.png', dpi=150, bbox_inches='tight')
print("Batch visualization saved to: batch_visualization.png")
EOF

python visualize_batch.py
```

### 3. Compute Evaluation Metrics

Install evaluation tool:

```bash
pip install pysodmetrics
```

Create evaluation script:

```bash
cat > evaluate.py << 'EOF'
import os
import numpy as np
from PIL import Image
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

def evaluate_dataset(pred_dir, gt_dir, dataset_name):
    print(f"\nEvaluating {dataset_name}:")
    print("-" * 50)
    
    # Initialize metrics
    mae_metric = MAE()
    em_metric = Emeasure()
    fm_metric = Fmeasure()
    sm_metric = Smeasure()
    wfm_metric = WeightedFmeasure()
    
    # Get all prediction files
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    
    for pred_file in pred_files:
        # Load prediction and ground truth
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, pred_file)
        
        if not os.path.exists(gt_path):
            continue
        
        pred = np.array(Image.open(pred_path).convert('L'))
        gt = np.array(Image.open(gt_path).convert('L'))
        
        # Normalize to [0, 1]
        pred = pred / 255.0
        gt = gt / 255.0
        
        # Update metrics
        mae_metric.step(pred, gt)
        em_metric.step(pred, gt)
        fm_metric.step(pred, gt)
        sm_metric.step(pred, gt)
        wfm_metric.step(pred, gt)
    
    # Get results
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

# Evaluate all datasets
datasets = [
    ('CAMO', './results/Depth_cod/CAMO_depth', './datasets/test/CAMO_depth/gt'),
    ('COD10K', './results/Depth_cod/cod10k_depth_test', './datasets/test/cod10k_depth_test/gt'),
    ('NC4K', './results/Depth_cod/NC4K', './datasets/test/NC4K/gt')
]

results = {}
for name, pred_dir, gt_dir in datasets:
    if os.path.exists(pred_dir) and os.path.exists(gt_dir):
        results[name] = evaluate_dataset(pred_dir, gt_dir, name)

# Print summary
print("\n" + "=" * 50)
print("Evaluation Summary")
print("=" * 50)
print(f"{'Dataset':<15} {'MAE':<10} {'Fβ':<10} {'Sm':<10} {'Em':<10}")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:<15} {metrics['mae']:<10.4f} {metrics['fm']:<10.4f} "
          f"{metrics['sm']:<10.4f} {metrics['em']:<10.4f}")
EOF

python evaluate.py
```

---

## Troubleshooting Guide

### Issue 1: CUDA out of memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce batch size** (simplest):
```python
# In train.py
args['train_batch_size'] = 5  # Change from 10 to 5
```

2. **Reduce input size**:
```python
args['scale'] = 352  # Change from 448 to 352
```

3. **Clear GPU cache**:
```python
import torch
torch.cuda.empty_cache()
```

### Issue 2: Module not found

**Symptom:**
```
ModuleNotFoundError: No module named 'xxx'
```

**Solution:**
```bash
# Confirm environment is activated
conda activate dacod

# Reinstall missing package
pip install xxx
```

### Issue 3: Pre-trained weight loading failed

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: './backbone/xxx.pth'
```

**Solution:**
```bash
# Check if file exists
ls -lh backbone/

# Ensure file name matches exactly, including case
```

### Issue 4: Dataset path error

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: './datasets/train/...'
```

**Solution:**
```bash
# Check dataset directory structure
tree datasets -L 3

# Confirm path configuration
cat utils/config.py
```

### Issue 5: Training interrupted

**Recovery method:**

1. Find last saved checkpoint:
```bash
ls -lt checkpoints/Depth_cod/
```

2. Modify `train.py`:
```python
args['last_epoch'] = 50      # Last completed epoch
args['snapshot'] = '50'       # Corresponding checkpoint file
```

3. Restart training:
```bash
python train.py
```

### Issue 6: TensorBoard not accessible

**Solutions:**

1. **Check if port is occupied**:
```bash
lsof -i :6006
```

2. **Change port**:
```bash
tensorboard --logdir=/root/tf-logs/new2/Depth_cod/log --port=6007
```

3. **SSH port forwarding** (remote server):
```bash
# Run on local machine
ssh -L 6006:localhost:6006 user@remote-server
# Then access in browser: http://localhost:6006
```

### Issue 7: Slow inference speed

**Optimization methods:**

1. **Confirm using GPU**:
```python
# Add at beginning of infer.py
import torch
print(f"Using device: {torch.cuda.get_device_name(0)}")
```

2. **Reduce image size**:
```python
# In infer.py
args['scale'] = 352  # Reduce from 448
```

3. **Use half precision inference**:
```python
net = net.half()  # After model loading
```

---

## Custom Data Processing

### Testing with Your Own Dataset

#### Step 1: Prepare Data

Create directory structure:
```bash
mkdir -p my_dataset/rgb
mkdir -p my_dataset/depth
mkdir -p my_dataset/gt  # If annotations available
```

#### Step 2: Generate Depth Maps

If you don't have depth maps, use MiDaS to generate:

```bash
# Clone MiDaS
git clone https://github.com/isl-org/MiDaS.git
cd MiDaS

# Download model
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt

# Batch generate depth maps
python run.py --model_type dpt_beit_large_512 \
              --input_path ../my_dataset/rgb \
              --output_path ../my_dataset/depth
cd ..
```

#### Step 3: Modify Configuration

Edit `utils/config.py`:
```python
# Add your dataset path
my_dataset_path = './my_dataset'
```

Edit `infer.py`:
```python
to_test = OrderedDict([
    ('my_dataset', my_dataset_path)
])
```

#### Step 4: Run Testing

```bash
python infer.py
```

Results saved at: `results/Depth_cod/my_dataset/`

### Training on Your Own Data

#### Step 1: Prepare Training Data

Ensure data structure:
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

**Note:**
- RGB images must be `.jpg` format
- Depth maps and annotations must be `.png` format
- File names (without extension) must match exactly

#### Step 2: Modify Configuration

Edit `utils/config.py`:
```python
depth_cod_training_root = './my_train_data'
```

#### Step 3: Start Training

```bash
python train.py
```

---

## Appendix: Quick Command Reference

### Environment Related
```bash
# Activate environment
conda activate dacod

# Verify environment
python verify_setup.py

# View GPU status
nvidia-smi
watch -n 1 nvidia-smi
```

### Training Related
```bash
# Start training
python train.py

# Background training
nohup python train.py > training.log 2>&1 &

# View training log
tail -f training.log

# Start TensorBoard
tensorboard --logdir=/root/tf-logs/new2/Depth_cod/log --port=6006
```

### Testing Related
```bash
# Run testing
python infer.py

# View results
ls results/Depth_cod/

# Visualize results
python visualize_single.py [image_name]
python visualize_batch.py
```

### Evaluation Related
```bash
# Install evaluation tool
pip install pysodmetrics

# Run evaluation
python evaluate.py
```

### Data Related
```bash
# View dataset info
python dataset_info.py

# Preview data
python preview_data.py

# Check data integrity
find datasets -name "*.jpg" | wc -l
find datasets -name "*.png" | wc -l
```

---

## Common Use Scenarios

### Scenario 1: Quick Test (No Training)

```bash
# 1. Download pre-trained model 55.pth
# 2. Prepare test data
# 3. Run testing
conda activate dacod
python infer.py
```

### Scenario 2: Train New Model from Scratch

```bash
# 1. Prepare complete training data
# 2. Download pre-trained backbone networks
# 3. Start training
conda activate dacod
python train.py
```

### Scenario 3: Test on Custom Data

```bash
# 1. Prepare data (RGB + depth maps)
# 2. Modify infer.py configuration
# 3. Run testing
python infer.py
```

### Scenario 4: Evaluate Model Performance

```bash
# 1. Run testing to generate predictions
python infer.py

# 2. Run evaluation script
python evaluate.py
```

---

## Getting Help

If you encounter issues:

1. **Check log files**: Examine `training.log` or command line output
2. **Run verification script**: `python verify_setup.py`
3. **Check GitHub Issues**: https://github.com/rywc2005/DaCOD/issues
4. **Read the paper**: Understand method details

---

**Document Version**: v1.0  
**Last Updated**: 2024  
**For**: DaCOD Project

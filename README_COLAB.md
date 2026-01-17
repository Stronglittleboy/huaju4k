# 华剧4K视频增强 - 免费云端部署指南

## 🎯 推荐平台对比

| 平台 | GPU | 免费时长 | 存储 | 稳定性 | 推荐度 |
|------|-----|----------|------|--------|--------|
| **Google Colab** | T4 (15GB) | 12小时/次 | 15GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Kaggle** | P100/T4 | 30小时/周 | 20GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Lightning AI** | A10G | 有限 | 5GB | ⭐⭐⭐ | ⭐⭐⭐ |

## 🚀 Google Colab 快速开始

### 方法1: 使用预制Notebook

1. **打开Colab**: https://colab.research.google.com/
2. **上传notebook**: 上传 `HuaJu4K_Colab.ipynb`
3. **设置GPU**: Runtime → Change runtime type → GPU (T4)
4. **运行**: 按顺序执行所有代码块

### 方法2: 手动部署

```bash
# 1. 在Colab中运行
!git clone https://github.com/Stronglittleboy/huaju4k.git
%cd huaju4k
!python deploy_to_colab.py
```

## 📱 使用流程

### 步骤1: 环境准备
```python
# 检查GPU
!nvidia-smi

# 克隆项目
!git clone https://github.com/Stronglittleboy/huaju4k.git
%cd huaju4k

# 自动设置环境
!python deploy_to_colab.py
```

### 步骤2: 上传视频
```python
from google.colab import files
uploaded = files.upload()

# 文件会自动移动到 colab_input/ 目录
```

### 步骤3: 开始处理
```python
# 运行简化的处理器
!python colab_processor.py
```

### 步骤4: 下载结果
```python
from google.colab import files
import os

# 下载所有处理结果
for filename in os.listdir('colab_output'):
    if filename.endswith('.mp4'):
        files.download(f'colab_output/{filename}')
```

## ⚙️ 配置优化

### 针对免费GPU的优化设置

```python
# 推荐配置 (colab_processor.py 中)
config = {
    "target_resolution": [1920, 1080],  # 降低到1080p
    "quality": "medium",                # 中等质量
    "tile_size": 256,                  # 小tile减少内存使用
    "batch_size": 2,                   # 小batch避免OOM
    "use_gpu": True                    # 启用GPU加速
}
```

### 内存管理技巧

```python
# 处理大文件时的分段策略
def process_large_video(video_path, segment_duration=300):  # 5分钟段
    """将大视频分段处理"""
    # 1. 分割视频
    # 2. 逐段处理
    # 3. 合并结果
    pass
```

## 🔧 故障排除

### 常见问题

1. **GPU内存不足**
   ```python
   # 减小batch_size和tile_size
   config["batch_size"] = 1
   config["tile_size"] = 128
   ```

2. **运行时断开**
   ```python
   # 添加保活代码
   import time
   from IPython.display import Javascript
   
   def keep_alive():
       display(Javascript('''
           function ClickConnect(){
               console.log("Working");
               document.querySelector("colab-toolbar-button#connect").click()
           }
           setInterval(ClickConnect,60000)
       '''))
   
   keep_alive()
   ```

3. **文件上传失败**
   ```python
   # 使用Google Drive挂载
   from google.colab import drive
   drive.mount('/content/drive')
   
   # 从Drive读取文件
   input_path = '/content/drive/MyDrive/videos/input.mp4'
   ```

## 📊 性能预期

### 处理速度参考 (T4 GPU)

| 视频长度 | 分辨率 | 预计时间 | 内存使用 |
|----------|--------|----------|----------|
| 1分钟 | 1080p | 5-10分钟 | 8-12GB |
| 5分钟 | 1080p | 25-50分钟 | 10-14GB |
| 10分钟 | 720p | 30-60分钟 | 6-10GB |

### 质量设置对比

| 设置 | 处理速度 | 质量提升 | 内存使用 |
|------|----------|----------|----------|
| Fast | 快 | 中等 | 低 |
| Medium | 中等 | 好 | 中等 |
| High | 慢 | 最佳 | 高 |

## 🎯 最佳实践

### 1. 文件管理
- 视频文件 < 500MB (免费版限制)
- 使用Google Drive存储大文件
- 及时清理临时文件

### 2. 处理策略
- 长视频分段处理
- 优先处理关键片段
- 批量处理多个短视频

### 3. 资源优化
- 监控GPU内存使用
- 适当降低处理参数
- 使用CPU作为备选方案

## 🔄 其他平台部署

### Kaggle Notebooks
```python
# Kaggle特定设置
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# 上传数据集
api = KaggleApi()
api.authenticate()
```

### Lightning AI Studio
```python
# Lightning AI 部署
!pip install lightning-ai
import lightning as L

# 创建Studio应用
app = L.LightningApp(VideoEnhancementApp())
```

## 📞 技术支持

遇到问题？
1. 检查 [Issues](https://github.com/Stronglittleboy/huaju4k/issues)
2. 查看项目文档
3. 提交新的Issue

---

**免费云端GPU让4K视频增强触手可及！** 🎬✨
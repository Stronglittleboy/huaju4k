# huaju4k Video Enhancement Tool

一个专为戏剧视频优化的4K视频增强工具，集成AI模型和智能资源管理。

## 快速开始

### 1. 环境设置

```bash
# 自动设置虚拟环境和依赖
./setup_env.sh

# 或手动设置
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 激活环境

```bash
# 激活虚拟环境
source venv/bin/activate

# 或使用快捷脚本（setup_env.sh运行后会创建）
./activate_env.sh
```

### 3. 运行项目

```bash
# 查看帮助
python -m huaju4k --help

# 基本使用示例
python -m huaju4k enhance input_video.mp4 --output output_4k.mp4 --preset theater_medium
```

## 项目结构

```
huaju4k/
├── huaju4k/                 # 主要代码
│   ├── core/               # 核心处理模块
│   │   ├── ai_model_manager.py    # AI模型管理
│   │   ├── tile_processor.py      # 瓦片处理
│   │   ├── ai_integration.py      # AI集成系统
│   │   └── memory_manager.py      # 内存管理
│   ├── models/             # 数据模型
│   ├── utils/              # 工具函数
│   └── configs/            # 配置管理
├── requirements*.txt       # 依赖文件
├── setup_env.sh           # 环境设置脚本
└── ENVIRONMENT_SETUP.md   # 详细环境设置指南
```

## 功能特性

- **AI视频增强**: 使用Real-ESRGAN等模型进行4K升级
- **戏剧音频优化**: 专门针对戏剧场景的音频增强
- **智能内存管理**: 自适应瓦片处理，防止内存溢出
- **GPU/CPU自动切换**: 根据系统资源自动选择处理方式
- **断点续传**: 支持处理中断后的恢复
- **批量处理**: 支持多文件批量处理

## 依赖管理

项目提供了多个requirements文件：

- `requirements.txt`: 完整功能（推荐）
- `requirements-minimal.txt`: 基本功能
- `requirements-ai.txt`: 仅AI模型依赖
- `requirements-dev.txt`: 开发工具

## 开发

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest

# 代码格式化
black huaju4k/
isort huaju4k/

# 类型检查
mypy huaju4k/
```

## 系统要求

- Python 3.8+
- Ubuntu 18.04+ / WSL2
- 4GB+ RAM（推荐8GB+）
- NVIDIA GPU（可选，用于AI加速）

详细环境设置请参考 [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

## 许可证

MIT License
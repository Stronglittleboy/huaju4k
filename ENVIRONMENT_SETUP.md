# huaju4k 环境设置指南

## Ubuntu/WSL 环境设置

### 快速开始

1. **自动设置环境**（推荐）：
```bash
./setup_env.sh
```

2. **手动设置环境**：
```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 选择安装模式（选择其一）
pip install -r requirements-minimal.txt    # 最小安装
pip install -r requirements.txt            # 完整安装
pip install -r requirements-ai.txt         # 仅AI模块
pip install -r requirements-dev.txt        # 开发工具
```

### 依赖文件说明

- **requirements.txt**: 完整依赖，包含所有功能
- **requirements-minimal.txt**: 最小依赖，仅基本功能
- **requirements-ai.txt**: AI模型相关依赖
- **requirements-dev.txt**: 开发和测试工具

### 日常使用

```bash
# 激活环境
source venv/bin/activate
# 或使用快捷脚本
./activate_env.sh

# 运行项目
python -m huaju4k --help

# 运行测试
pytest

# 退出环境
deactivate
```

### 环境验证

激活环境后运行：
```bash
python -c "
import sys
sys.path.append('.')
from huaju4k.core.ai_model_manager import AIModelManager
print('✅ 环境配置正确')
"
```

### 故障排除

1. **Python版本**: 确保使用Python 3.8+
2. **权限问题**: 确保setup_env.sh有执行权限
3. **依赖冲突**: 删除venv文件夹重新创建
4. **GPU支持**: 安装CUDA和对应的PyTorch版本

### GPU支持（可选）

如果需要GPU加速，确保安装：
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应的PyTorch版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
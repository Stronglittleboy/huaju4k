# huaju4k 使用示例

## 环境设置示例

### 1. 首次设置

```bash
# 克隆项目（如果还没有）
git clone <repository-url>
cd huaju4k

# 运行自动环境设置
./setup_env.sh

# 按提示选择安装模式：
# 1) 最小安装 (基本功能) - 适合测试
# 2) 完整安装 (包含AI模型) - 推荐用于生产
# 3) 开发安装 (包含开发工具) - 适合开发者
# 4) 自定义安装 - 高级用户
```

### 2. 日常使用

```bash
# 激活环境
source venv/bin/activate
# 或
./activate_env.sh

# 验证安装
python -c "
import sys
sys.path.append('.')
from huaju4k.core.ai_model_manager import AIModelManager
print('✅ 环境正常')
"

# 退出环境
deactivate
```

### 3. 不同安装模式的区别

#### 最小安装 (requirements-minimal.txt)
- 仅包含基本功能
- 不包含AI模型依赖
- 适合：测试、CI/CD、资源受限环境
- 大小：~200MB

#### 完整安装 (requirements.txt)
- 包含所有功能
- 包含AI模型依赖（PyTorch、Real-ESRGAN等）
- 适合：生产环境、完整功能使用
- 大小：~2-3GB

#### 开发安装 (requirements-dev.txt)
- 包含测试工具（pytest、hypothesis）
- 包含代码质量工具（black、mypy、flake8）
- 包含文档工具
- 适合：开发者、贡献者

### 4. 故障排除

#### Python版本问题
```bash
# 检查Python版本
python3 --version

# 如果版本过低，安装新版本
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev
```

#### 权限问题
```bash
# 给脚本执行权限
chmod +x setup_env.sh
chmod +x activate_env.sh
```

#### 依赖安装失败
```bash
# 清理并重新安装
rm -rf venv
./setup_env.sh

# 或手动安装
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-minimal.txt
```

#### GPU相关问题
```bash
# 检查CUDA
nvidia-smi

# 安装GPU版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 5. 开发工作流

```bash
# 1. 激活环境
source venv/bin/activate

# 2. 安装开发依赖
pip install -r requirements-dev.txt

# 3. 运行测试
pytest

# 4. 代码格式化
black huaju4k/
isort huaju4k/

# 5. 类型检查
mypy huaju4k/

# 6. 提交前检查
pre-commit run --all-files
```

### 6. 性能优化建议

#### 内存优化
```bash
# 使用最小安装减少内存占用
pip install -r requirements-minimal.txt

# 监控内存使用
python -c "
from huaju4k.core.memory_manager import ConservativeMemoryManager
mm = ConservativeMemoryManager()
status = mm.monitor_and_adjust()
print(f'可用内存: {status.available_memory_mb}MB')
"
```

#### 存储优化
```bash
# 清理pip缓存
pip cache purge

# 清理临时文件
rm -rf workspace/temp/*
rm -rf workspace/enhanced_backup/*
```

### 7. 常见使用场景

#### 场景1：快速测试
```bash
# 最小安装，快速验证功能
./setup_env.sh  # 选择选项1
source venv/bin/activate
python -m huaju4k --help
```

#### 场景2：生产使用
```bash
# 完整安装，包含所有AI功能
./setup_env.sh  # 选择选项2
source venv/bin/activate
python -m huaju4k enhance video.mp4 --preset theater_medium
```

#### 场景3：开发贡献
```bash
# 开发安装，包含所有工具
./setup_env.sh  # 选择选项3
source venv/bin/activate
pytest
black huaju4k/
```

这样的环境管理方式解决了您提到的两个问题：
1. ✅ 提供了虚拟环境管理脚本
2. ✅ 创建了详细的requirements.txt依赖管理系统
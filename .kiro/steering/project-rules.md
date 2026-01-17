# 视频增强GUI项目规则

## 开发环境规则

### WSL/Ubuntu环境配置
- **重要规则**: 所有构建和运行操作都必须在WSL/Ubuntu环境中执行
- **当前环境**: 已在WSL2 Ubuntu环境中 (Linux DESKTOP-MQ8QRL9)
- **WSL账号**: `whh`
- **项目路径**: `/mnt/d/workProject/huaju4k/video-enhancement-gui`
- **Rust工具链**: 需要更新到最新版本以支持edition2024特性

### 开发工具链要求
- **Node.js**: 用于前端开发和构建
- **Rust**: 最新版本 (支持edition2024)，通过rustup安装
- **Cargo**: Rust包管理器，需要最新版本
- **Python**: 用于视频处理后端集成
## 技术栈要求

### 前端框架
- **Vue.js 3**: 使用Composition API和TypeScript
- **UI组件库**: 推荐使用Element Plus或Ant Design Vue
- **状态管理**: 使用Pinia进行状态管理
- **路由**: Vue Router 4用于页面导航

### 桌面应用框架
- **Tauri**: 使用Rust后端和Web前端的混合架构
- **打包目标**: Windows x64可执行文件
- **权限配置**: 文件系统访问、进程执行权限

### 开发工具
- **构建工具**: Vite用于快速开发和构建
- **代码质量**: ESLint + Prettier用于代码规范
- **类型检查**: TypeScript严格模式

## 视频处理集成规则

### OpenCV GPU配置和性能特征

#### 已验证的系统配置
- **GPU**: NVIDIA GTX 1650 (4GB VRAM)
- **系统**: Windows + WSL Ubuntu
- **OpenCV版本**: 4.13.0-pre
- **CUDA设备数**: 1个

#### OpenCV CUDA功能状态

##### ✅ 可用功能
- GPU内存分配和管理
- 数据上传/下载 (CPU ↔ GPU)
- 基本图像处理函数:
  - `cv2.cuda.resize()` - 图像缩放
  - `cv2.cuda.cvtColor()` - 颜色空间转换
  - `cv2.cuda.threshold()` - 阈值处理
  - `cv2.cuda.bilateralFilter()` - 双边滤波
  - `cv2.cuda.warpAffine()` - 仿射变换
  - `cv2.cuda.remap()` - 重映射

##### ❌ 不可用功能
- `cv2.cuda.GaussianBlur()` - 高斯模糊
- `cv2.cuda.Canny()` - Canny边缘检测
- `cv2.cuda.morphologyEx()` - 形态学操作
- `cv2.cuda.blur()` - 普通模糊

#### 性能特征和使用策略

##### ⚠️ GPU性能限制
- **简单操作**: GPU比CPU慢 (0.02x加速比)
- **复杂操作**: GPU仍比CPU慢 (0.78x加速比)
- **主要瓶颈**: 
  - 数据传输开销 (CPU ↔ GPU)
  - GPU初始化开销
  - 对于小图像/简单操作，CPU已经很快

##### 💡 最佳使用场景
- 大批量图像处理 (减少初始化开销)
- 复杂算法 (如AI模型推理)
- 支持CUDA的第三方库 (如Real-ESRGAN)

### OpenCV GPU配置遵循
基于现有的OpenCV GPU验证结果：

#### ✅ 推荐使用的功能
- CPU优化作为主要处理方式
- GPU仅用于支持CUDA的第三方库（如Real-ESRGAN）
- 混合处理模式：
  - 帧提取：CPU (FFmpeg)
  - 基本预处理：CPU (resize, 颜色转换)
  - AI升级：GPU (如果AI模型支持CUDA)
  - 后处理：CPU (简单操作)

#### ❌ 避免使用的功能
- 简单操作的GPU加速（性能反而下降）
- 频繁的CPU-GPU数据传输
- 小图像的GPU处理

### 编程指导原则

#### 🎯 推荐策略
```python
# 优先使用CPU进行简单操作
cpu_result = cv2.resize(image, (width, height))
cpu_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 仅在以下情况考虑GPU:
# 1. 大批量处理
# 2. 复杂算法
# 3. 第三方CUDA库
```

#### 🔧 混合处理模式
- **帧提取**: CPU (FFmpeg)
- **基本预处理**: CPU (resize, 颜色转换)
- **AI升级**: GPU (如果AI模型支持CUDA)
- **后处理**: CPU (简单操作)

#### 📊 性能优化建议
1. **避免频繁的CPU-GPU数据传输**
2. **批处理多个操作减少初始化开销**
3. **对于单张图像的简单操作，直接使用CPU**
4. **保留GPU功能作为特定场景的备选方案**

#### 错误处理模板
```python
def safe_gpu_operation(image, operation):
    """安全的GPU操作，失败时回退到CPU"""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            # GPU操作
            result_gpu = operation(gpu_img)
            return result_gpu.download()
        else:
            # CPU回退
            return cpu_operation(image)
    except Exception:
        # 出错时使用CPU
        return cpu_operation(image)
```

### 处理策略
```javascript
// 推荐的处理流程
const processingStrategy = {
  frameExtraction: 'CPU_FFMPEG',
  preprocessing: 'CPU_OPENCV',
  aiUpscaling: 'GPU_IF_AVAILABLE',
  postprocessing: 'CPU_OPENCV',
  audioProcessing: 'CPU_LIBROSA'
}
```

## 架构设计原则

### 模块化设计
- **核心处理模块**: 独立的视频/音频处理逻辑
- **UI组件**: 可复用的Vue组件
- **状态管理**: 集中式的任务和配置状态
- **服务层**: API调用和文件操作抽象

### 任务管理架构
```typescript
interface Task {
  id: string
  name: string
  inputFile: string
  outputPath: string
  parameters: ProcessingParameters
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  createdAt: Date
  completedAt?: Date
  error?: string
}
```

### 参数配置结构
```typescript
interface ProcessingParameters {
  video: {
    aiModel: 'real-esrgan' | 'esrgan' | 'waifu2x'
    targetResolution: [number, number]
    quality: 'fast' | 'medium' | 'high'
    tileSize: number
    batchSize: number
  }
  audio: {
    theaterPreset: 'small' | 'medium' | 'large' | 'custom'
    noiseReduction: number // 0.0-1.0
    dialogueEnhancement: number // 0.0-1.0
    preserveNaturalness: boolean
    reverbSettings: ReverbSettings
  }
  performance: {
    useGPU: boolean
    cpuThreads: number
    memoryLimit: number
  }
}
```

## 用户界面设计规范

### 布局结构
- **侧边栏**: 主要功能导航（任务管理、参数配置、历史记录）
- **主内容区**: 当前功能的详细界面
- **状态栏**: 系统状态和进度显示
- **工具栏**: 常用操作快捷按钮

### 组件设计原则
- **响应式设计**: 适配不同屏幕尺寸
- **无障碍访问**: 支持键盘导航和屏幕阅读器
- **国际化支持**: 使用Vue I18n进行多语言支持
- **主题系统**: 支持明暗主题切换

### 交互设计
- **拖拽上传**: 支持拖拽文件到应用程序
- **实时预览**: 参数调整时的实时效果预览
- **进度反馈**: 详细的处理进度和状态信息
- **错误处理**: 友好的错误提示和恢复建议

## 性能优化规则

### 内存管理
- **大文件处理**: 分段加载和处理
- **缓存策略**: 智能缓存中间结果
- **垃圾回收**: 及时释放不需要的资源
- **内存监控**: 实时监控内存使用情况

### 处理优化
- **多线程处理**: 利用Web Workers进行后台处理
- **批量操作**: 优化批量文件处理流程
- **断点续传**: 支持大文件的断点续传
- **资源调度**: 智能分配CPU和GPU资源

## 错误处理和日志

### 错误分类
- **用户错误**: 文件格式不支持、参数无效等
- **系统错误**: 内存不足、磁盘空间不够等
- **处理错误**: AI模型加载失败、编码错误等
- **网络错误**: 模型下载失败、更新检查失败等

### 日志记录
- **操作日志**: 记录所有用户操作
- **处理日志**: 详细的处理步骤和参数
- **错误日志**: 完整的错误信息和堆栈
- **性能日志**: 处理时间和资源使用情况

### 恢复机制
- **自动保存**: 定期保存任务状态
- **断点续传**: 从中断点恢复处理
- **配置备份**: 自动备份用户配置
- **数据恢复**: 意外关闭后的数据恢复

## 文件和目录结构

### 项目结构
```
video-enhancement-gui/
├── src-tauri/          # Tauri后端代码
│   ├── src/
│   ├── Cargo.toml
│   └── tauri.conf.json
├── src/                # Vue前端代码
│   ├── components/     # 可复用组件
│   ├── views/          # 页面组件
│   ├── stores/         # Pinia状态管理
│   ├── services/       # API服务
│   ├── utils/          # 工具函数
│   └── types/          # TypeScript类型定义
├── public/             # 静态资源
└── dist/               # 构建输出
```

### 工作目录
- **临时文件**: `%TEMP%/video-enhancement-gui/`
- **配置文件**: `%APPDATA%/video-enhancement-gui/config/`
- **日志文件**: `%APPDATA%/video-enhancement-gui/logs/`
- **预设文件**: `%APPDATA%/video-enhancement-gui/presets/`

## 测试策略

### 单元测试
- **组件测试**: Vue组件的单元测试
- **工具函数测试**: 纯函数的单元测试
- **状态管理测试**: Pinia store的测试

### 集成测试
- **API集成**: 前后端接口测试
- **文件处理**: 文件操作的集成测试
- **用户流程**: 完整用户操作流程测试

### 性能测试
- **内存使用**: 大文件处理的内存测试
- **处理速度**: 不同参数下的处理性能
- **并发处理**: 多任务并发的性能测试

## 部署和分发

### 构建配置
- **开发环境**: 热重载和调试支持
- **生产环境**: 代码压缩和优化
- **打包配置**: Tauri打包为Windows可执行文件

### 安装程序
- **MSI安装包**: 标准Windows安装程序
- **便携版本**: 免安装的便携版本
- **自动更新**: 内置的自动更新机制

### 版本管理
- **语义化版本**: 遵循SemVer版本规范
- **更新日志**: 详细的版本更新记录
- **兼容性**: 向后兼容的配置文件格式

## 安全和隐私

### 数据安全
- **本地处理**: 所有处理在本地进行
- **文件权限**: 最小化文件访问权限
- **临时文件**: 及时清理临时文件
- **配置加密**: 敏感配置信息加密存储

### 用户隐私
- **无数据收集**: 不收集用户个人信息
- **本地存储**: 所有数据本地存储
- **透明度**: 明确的隐私政策说明

最后更新: 2025-01-02 (合并OpenCV GPU配置规则)
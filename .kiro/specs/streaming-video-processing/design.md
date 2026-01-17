# 流式视频处理系统设计文档

## 概述

本设计实现了一个工程级流式视频处理系统，彻底解决磁盘空间限制问题。系统采用"永不落盘中间帧"的核心原则，所有帧处理都在内存中完成，确保在20GB磁盘空间限制下处理任意长度的4K视频。

## 架构设计

### 核心数据流

```
input.mp4
    ↓ (VideoCapture解码)
[frame: np.ndarray] ← 仅存在于内存
    ↓ (转换为Tensor)
[input_tensor: torch.Tensor] ← GPU内存
    ↓ (AI增强/mask/时序)
[enhanced_tensor: torch.Tensor] ← GPU内存
    ↓ (转换回numpy)
[output_frame: np.ndarray] ← 内存
    ↓ (VideoWriter编码)
output.mp4
```

**关键原则**:
- 帧从来没有.png形态
- 帧从来没有"目录"存储
- 一帧处理完就消失

### 系统组件

#### 1. StreamingVideoProcessor

核心流式处理器，替代现有的基于文件的处理方式。

```python
class StreamingVideoProcessor:
    """
    工程级流式视频处理器
    
    特点:
    - 永不落盘中间帧
    - 恒定磁盘占用
    - 显存优化管理
    - 支持断点续传
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.checkpoint_manager = CheckpointManager()
        self.temporal_buffer = TemporalBuffer(max_frames=3)
        
    def process_streaming(self, input_path: str, output_path: str) -> StreamingResult:
        """主流式处理入口"""
        pass
```

#### 2. MemoryFrameBuffer

内存帧缓冲管理器，确保高效的内存使用。

```python
class MemoryFrameBuffer:
    """
    内存帧缓冲器
    
    职责:
    - 管理帧在内存中的生命周期
    - 自动释放已处理帧
    - 监控内存使用量
    """
    
    def __init__(self, max_buffer_size_mb: int = 1024):
        self.max_buffer_size = max_buffer_size_mb
        self.current_frames = {}
        self.memory_usage = 0
        
    def add_frame(self, frame_id: int, frame: np.ndarray) -> bool:
        """添加帧到缓冲区，自动检查内存限制"""
        pass
        
    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """获取帧数据"""
        pass
        
    def release_frame(self, frame_id: int) -> None:
        """立即释放帧内存"""
        pass
```

#### 3. GPUMemoryManager

显存优化管理器，确保6GB显存限制下稳定运行。

```python
class GPUMemoryManager:
    """
    GPU显存管理器
    
    策略:
    - 模型一次性加载，全程保持
    - 每帧处理后立即清理张量
    - 锯齿型显存使用模式
    """
    
    def __init__(self, max_vram_mb: int = 6144):
        self.max_vram = max_vram_mb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
    def load_model_once(self, model_path: str) -> torch.nn.Module:
        """一次性加载模型到显存"""
        pass
        
    def process_frame_tensor(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """处理帧张量，自动管理显存"""
        pass
        
    def cleanup_frame_tensors(self, *tensors) -> None:
        """清理帧处理产生的张量"""
        pass
```

#### 4. TemporalBuffer

时序缓冲器，在内存中维护时序一致性。

```python
class TemporalBuffer:
    """
    时序帧缓冲器
    
    功能:
    - 维护有限的历史帧用于时序稳定
    - 滑动窗口机制
    - 内存中的光流计算
    """
    
    def __init__(self, max_frames: int = 3):
        self.max_frames = max_frames
        self.frame_history = deque(maxlen=max_frames)
        
    def add_frame(self, frame_tensor: torch.Tensor) -> None:
        """添加帧到时序缓冲区"""
        pass
        
    def get_previous_frame(self) -> Optional[torch.Tensor]:
        """获取前一帧用于时序处理"""
        pass
        
    def apply_temporal_stabilization(self, current_frame: torch.Tensor) -> torch.Tensor:
        """应用时序稳定，使用内存中的历史帧"""
        pass
```

#### 5. CheckpointManager

轻量级检查点管理器，支持断点续传。

```python
class CheckpointManager:
    """
    检查点管理器
    
    特点:
    - 仅保存进度信息，不保存帧数据
    - 轻量级JSON格式
    - 快速恢复机制
    """
    
    def save_checkpoint(self, frame_index: int, processing_params: dict) -> None:
        """保存检查点"""
        pass
        
    def load_checkpoint(self) -> Optional[dict]:
        """加载检查点"""
        pass
        
    def clear_checkpoint(self) -> None:
        """清理检查点文件"""
        pass
```

## 核心处理流程

### 主循环设计

基于你提供的工程级参考，核心主循环如下：

```python
def streaming_main_loop(self, input_path: str, output_path: str) -> bool:
    """
    工程级主循环 - 永不落盘中间帧
    """
    # 1. 初始化（只做一次）
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w*4, h*4))  # 4K输出
    
    # 2. 模型加载（显存一次性占住）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = self.gpu_memory_manager.load_model_once(self.config.model_path)
    
    # 3. 核心主循环
    frame_index = 0
    last_frame_tensor = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_index += 1
        
        # ① OpenCV帧 → Torch张量（内存态）
        input_tensor = self._frame_to_tensor(frame, device)
        
        # ② AI增强（显存核心区）
        with torch.no_grad():
            enhanced_tensor = model(input_tensor)
        
        # ③ 时序稳定（用上一帧，而不是磁盘）
        if last_frame_tensor is not None:
            enhanced_tensor = self.temporal_buffer.apply_temporal_stabilization(
                enhanced_tensor, last_frame_tensor
            )
        
        # ④ 张量 → OpenCV帧（立刻可写）
        output_frame = self._tensor_to_frame(enhanced_tensor)
        
        # ⑤ 写入视频（唯一落盘点）
        writer.write(output_frame)
        
        # ⑥ 清理 & 滑动窗口
        last_frame_tensor = enhanced_tensor.detach()
        del input_tensor
        del enhanced_tensor
        
        # 显存清理
        if frame_index % 10 == 0:  # 每10帧清理一次
            torch.cuda.empty_cache()
    
    # 4. 收尾
    cap.release()
    writer.release()
    
    return True
```

### 内存管理策略

#### 帧生命周期管理

```python
def _manage_frame_lifecycle(self, frame_index: int, frame: np.ndarray) -> None:
    """
    帧生命周期管理
    
    生命周期:
    1. 从VideoCapture读取 → 内存numpy数组
    2. 转换为GPU张量 → 显存
    3. AI处理 → 显存
    4. 转换回CPU → 内存numpy数组
    5. 写入VideoWriter → 立即释放
    """
    # 帧在内存中的最大存活时间：单次循环
    # 绝不允许帧数据持久化到磁盘
```

#### 显存使用模式

```python
def _gpu_memory_pattern(self) -> None:
    """
    显存使用模式：锯齿型但不增长
    
    模式:
    - 模型加载：固定占用 ~3GB
    - 帧处理：波动占用 ~1-2GB
    - 总计：不超过6GB
    """
    # 每帧处理后显存使用回到基线
    # 通过torch.no_grad()和显式del确保
```

## 磁盘空间控制

### 磁盘占用计算

```python
def calculate_disk_usage(self, input_path: str, output_path: str) -> dict:
    """
    计算磁盘占用
    
    组成:
    - input.mp4: 原始大小
    - output.mp4: 约为原始大小的2-3倍（4K）
    - 系统缓存: < 1GB
    - 检查点文件: < 10MB
    
    总计: 约为原始文件的3-4倍，与视频长度无关
    """
    input_size = Path(input_path).stat().st_size
    estimated_output_size = input_size * 3  # 4K压缩后估算
    system_cache = 1024 * 1024 * 1024  # 1GB
    checkpoint_size = 10 * 1024 * 1024  # 10MB
    
    total_estimated = input_size + estimated_output_size + system_cache + checkpoint_size
    
    return {
        "input_size_gb": input_size / (1024**3),
        "estimated_output_gb": estimated_output_size / (1024**3),
        "total_estimated_gb": total_estimated / (1024**3),
        "safe_for_20gb": total_estimated < 18 * (1024**3)  # 留2GB缓冲
    }
```

### 空间预检查

```python
def pre_check_disk_space(self, input_path: str, available_space_gb: int) -> bool:
    """
    处理前磁盘空间检查
    
    检查项:
    1. 输入文件大小
    2. 预估输出文件大小
    3. 系统缓存需求
    4. 安全缓冲区
    """
    usage = self.calculate_disk_usage(input_path, "")
    return usage["safe_for_20gb"] and usage["total_estimated_gb"] < available_space_gb * 0.9
```

## 错误处理和恢复

### 断点续传机制

```python
def resume_from_checkpoint(self, checkpoint_path: str) -> dict:
    """
    从检查点恢复处理
    
    恢复信息:
    - 上次处理到的帧索引
    - 处理参数
    - 模型配置
    
    不包含:
    - 帧数据（永远不保存）
    - 中间结果
    """
    checkpoint = self.checkpoint_manager.load_checkpoint()
    if checkpoint:
        return {
            "resume_from_frame": checkpoint["frame_index"],
            "processing_params": checkpoint["params"],
            "can_resume": True
        }
    return {"can_resume": False}
```

### 内存不足处理

```python
def handle_memory_shortage(self, current_batch_size: int) -> int:
    """
    内存不足时的处理策略
    
    策略:
    1. 减少批处理大小
    2. 强制垃圾回收
    3. 清理缓存
    4. 如果仍不足，报错退出
    """
    # 自动降级处理
    new_batch_size = max(1, current_batch_size // 2)
    
    # 强制清理
    torch.cuda.empty_cache()
    gc.collect()
    
    return new_batch_size
```

## 性能监控

### 实时监控指标

```python
class StreamingMonitor:
    """
    流式处理监控器
    
    监控指标:
    - 内存使用量（系统+显存）
    - 处理速度（FPS）
    - 磁盘I/O（确认仅有视频读写）
    - 预计完成时间
    """
    
    def get_realtime_stats(self) -> dict:
        return {
            "memory_usage_mb": self._get_memory_usage(),
            "vram_usage_mb": self._get_vram_usage(),
            "processing_fps": self._calculate_fps(),
            "disk_io_ops": self._monitor_disk_io(),
            "eta_seconds": self._estimate_completion_time()
        }
```

## 系统集成

### 与现有架构的兼容性

```python
class StreamingVideoEnhancementProcessor(VideoProcessor):
    """
    流式处理器，实现现有VideoProcessor接口
    
    集成方式:
    - 作为可选的处理模式
    - 保持现有配置格式
    - 兼容进度跟踪组件
    """
    
    def process(self, input_path: str, output_path: str = None, 
                preset: str = "theater_medium", quality: str = "balanced") -> ProcessResult:
        """
        实现现有接口，内部使用流式处理
        """
        # 检查是否启用流式模式
        if self.config.get("streaming_mode", False):
            return self._process_streaming(input_path, output_path, preset, quality)
        else:
            # 回退到原有处理方式
            return super().process(input_path, output_path, preset, quality)
```

## 正确性属性

*属性是一个特征或行为，应该在系统的所有有效执行中保持为真——本质上是关于系统应该做什么的正式声明。属性作为人类可读规范和机器可验证正确性保证之间的桥梁。*

### 属性 1: 磁盘空间恒定性
*对于任意* 输入视频长度，系统的磁盘占用应该保持在输入文件大小的4倍以内，与视频时长无关
**验证: 需求 1.2, 5.1**

### 属性 2: 内存流式处理
*对于任意* 视频帧，帧数据应该仅存在于内存中，从不以文件形式保存到磁盘
**验证: 需求 1.3, 2.1**

### 属性 3: 显存稳定性
*对于任意* 处理会话，显存使用应该呈现锯齿型波动但峰值不超过6GB限制
**验证: 需求 3.2, 3.5**

### 属性 4: 时序一致性
*对于任意* 连续帧对，时序处理应该使用内存中的历史帧数据，不依赖磁盘文件
**验证: 需求 4.2, 4.4**

### 属性 5: 资源清理完整性
*对于任意* 处理完成的帧，相关的内存和显存资源应该在下一帧处理前被完全释放
**验证: 需求 2.5, 3.3**

### 属性 6: 断点续传轻量性
*对于任意* 检查点保存操作，检查点文件应该仅包含进度信息，不包含帧数据，大小不超过10MB
**验证: 需求 6.2, 6.3**

## 错误处理策略

### 内存不足错误处理
- 自动降低批处理大小
- 强制垃圾回收
- 清理GPU缓存
- 如果仍不足，优雅退出并保存检查点

### 磁盘空间不足预防
- 处理前预检查可用空间
- 实时监控磁盘使用
- 达到阈值时提前停止

### 模型加载失败处理
- 检查显存可用性
- 尝试CPU回退模式
- 提供详细错误信息

## 测试策略

### 单元测试
- 内存管理组件测试
- 显存优化测试
- 检查点机制测试

### 集成测试
- 完整流式处理流程测试
- 长视频处理稳定性测试
- 断点续传功能测试

### 性能测试
- 不同长度视频的磁盘占用测试
- 显存使用模式验证
- 处理速度基准测试

### 属性测试
- 磁盘空间恒定性验证
- 内存流式处理验证
- 显存稳定性验证
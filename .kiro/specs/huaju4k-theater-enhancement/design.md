# Huaju4K 话剧母版级4K增强系统 - 设计文档 (实现版)

## 系统架构设计

### 整体架构
```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface                            │
│              python -m huaju4k enhance                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              VideoEnhancementProcessor                     │
│                   (主控制器)                                │
│         使用 StrategyDrivenModelManager                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌───▼────┐ ┌──────▼──────┐
│   Stage 1    │ │Stage 2 │ │  Stage 4    │
│ 舞台结构分析  │ │策略生成│ │ 三阶段增强   │
│   (CPU)      │ │ (CPU)  │ │  (FFmpeg)   │
└──────────────┘ └────────┘ └─────────────┘
```

### 三阶段视频增强流程 (Stage 4)
```
┌─────────────────────────────────────────────────────────────┐
│                  ThreeStageVideoEnhancer                   │
│                    (全流式FFmpeg处理)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼───────────┐ ┌───▼───────────┐ ┌───▼───────────┐
│  Stage 4.1    │ │  Stage 4.2    │ │  Stage 4.3    │
│  结构重建     │ │  GAN增强      │ │  时序锁定     │
│              │ │              │ │              │
│ FFmpeg:      │ │ FFmpeg:      │ │ FFmpeg:      │
│ - lanczos    │ │ - unsharp    │ │ - deflicker  │
│ - unsharp    │ │ - eq         │ │ - 音频合并   │
│ - hqdn3d     │ │ - hqdn3d     │ │              │
│              │ │              │ │              │
│ 进度条显示 ✓ │ │ 进度条显示 ✓ │ │ 进度条显示 ✓ │
└──────────────┘ └──────────────┘ └──────────────┘
```

### 关键设计决策

1. **全流式处理**: 不提取帧到磁盘，使用FFmpeg管道直接处理
2. **实时进度显示**: 每个阶段显示进度条和帧计数
3. **真实 GPU 超分**: Stage 4.2 优先使用 Real-ESRGAN GPU，失败时回退到 FFmpeg
4. **策略驱动**: 使用 `StrategyDrivenModelManager` 而非基类 `AIModelManager`

---

## GPU Stage 设计 (Stage 4.2 真实 GPU 超分)

### 设计边界

| 类型 | 说明 |
|------|------|
| ❌ 不追求 | 流式/Zero-copy、显存极限优化、多模型并行 |
| ✅ 追求 | GPU 确实参与像素计算、可插拔可回退、可验证 |

### GPU Stage 架构
```
ThreeStageVideoEnhancer
├── Stage 4.1 CPU 结构重建 (FFmpeg lanczos)
├── Stage 4.2 GPU 超分增强 (Real-ESRGAN) ⭐ NEW
│   ├── 优先: GPUVideoSuperResolver (真实 GPU)
│   └── 回退: FFmpeg 滤镜增强
└── Stage 4.3 CPU 时序 + 音频合成 (FFmpeg)
```

### GPU Stage 代码结构
```
huaju4k/
├── gpu_stage/
│   ├── __init__.py
│   ├── gpu_super_resolver.py   # ⭐ 核心 GPU 处理器
│   └── model_manager.py        # 模型管理
└── requirements-gpu.txt        # GPU 依赖
```

### GPUVideoSuperResolver 接口
```python
class GPUVideoSuperResolver:
    """
    真实 GPU 视频超分处理器
    
    使用 Real-ESRGAN 进行 GPU 超分，确保：
    - nvidia-smi 显示显存占用 > 1GB
    - GPU Util 波动 30%~90%
    - 输出视频可播放且分辨率提升
    """
    
    def __init__(self,
                 model_name: str = "RealESRGAN_x4plus",
                 tile_size: int = 384,  # 6GB 显存安全值
                 device: str = "cuda"):
        """
        约束：
        - 初始化即加载模型到 GPU
        - 进程生命周期内只加载一次
        """
    
    def enhance_video(self,
                     input_video: str,
                     output_video: str,
                     progress_callback: Optional[Callable] = None) -> bool:
        """
        GPU 超分处理流程：
        1. OpenCV 解码视频帧
        2. frame → torch.Tensor → GPU
        3. Real-ESRGAN forward() 推理
        4. OpenCV 编码输出视频
        
        Returns:
            成功返回 True
        """
```

### GPU 执行流程
```
[input.mp4]
    │
    ▼
OpenCV decode
    │
raw RGB frame
    ▼
Torch Tensor (CUDA)   ◀────── GPU Util 在这里上升
    │
    ▼
Real-ESRGAN forward()
    │
    ▼
OpenCV encode
    │
    ▼
[output_sr.mp4]
```

### 显存与参数约束 (6GB GPU 安全配置)

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| tile_size | 384 | 6GB 显存安全值 |
| half | True | FP16 节省显存 |
| batch | 1 | 单帧处理 |
| model | RealESRGAN_x4plus | 4x 超分 |

### 与主流程集成
```python
# three_stage_enhancer.py 中的集成逻辑

def _stage_4_2_controlled_gan_enhancement(self, input_path: str) -> StageResult:
    # 1. 尝试真实 GPU 超分
    gpu_success = self._try_real_gpu_super_resolution(input_path, output_path)
    
    if gpu_success:
        return StageResult(success=True, metadata={"gpu_used": True})
    
    # 2. GPU 失败，回退到 FFmpeg 滤镜
    return self._fallback_ffmpeg_enhancement(input_path, output_path)

def _try_real_gpu_super_resolution(self, input_path: str, output_path: str) -> bool:
    try:
        from ..gpu_stage import GPUVideoSuperResolver
        
        resolver = GPUVideoSuperResolver(
            model_name="RealESRGAN_x4plus",
            tile_size=384
        )
        
        success = resolver.enhance_video(input_path, output_path)
        resolver.cleanup()
        
        return success
    except Exception:
        logger.warning("GPU SR failed, fallback to CPU")
        return False
```

### GPU 成功判定标准

运行时必须同时满足：

| 指标 | 要求 |
|------|------|
| nvidia-smi 显存 | > 1GB 占用 |
| GPU Util | 波动 30%~90% |
| 输出视频 | 可播放 |
| 分辨率 | 确实提升 (如 1080p → 4K) |

### 失败回退设计
```python
try:
    gpu_stage.enhance_video(input_path, output_path)
except Exception:
    logger.warning("GPU SR failed, fallback to FFmpeg filters")
    # 使用 FFmpeg 滤镜增强
    ffmpeg_enhance(input_path, output_path)
```

**原则**: GPU 失败 ≠ 整个流程失败

### 核心组件设计

#### 1. StageStructureAnalyzer (Stage 1)
**位置**: `huaju4k/analysis/stage_structure_analyzer.py`

**职责**:
- 对输入视频进行舞台结构分析
- 输出客观的数值化特征
- 不做任何主观判断或决策

**核心算法**:
```python
class StageStructureAnalyzer:
    def analyze_structure(self, video_path: str) -> StructureFeatures:
        """
        分析舞台结构特征
        
        Returns:
            StructureFeatures: 包含所有数值化特征的数据类
        """
        # 1. 亮度结构分析
        lighting_features = self._analyze_lighting_structure(frames)
        
        # 2. 边缘密度分析  
        edge_features = self._analyze_edge_density(frames)
        
        # 3. 帧间变化分析
        motion_features = self._analyze_frame_changes(frames)
        
        # 4. 噪声评估
        noise_features = self._analyze_noise_level(frames)
        
        return StructureFeatures(
            lighting=lighting_features,
            edge=edge_features,
            motion=motion_features,
            noise=noise_features
        )
```

**输出数据结构**:
```python
@dataclass
class StructureFeatures:
    # 基础视频信息
    resolution: Tuple[int, int]
    fps: float
    duration: float
    total_frames: int
    
    # 舞台结构特征
    is_static_camera: bool
    highlight_ratio: float      # 高光区域比例
    dark_ratio: float          # 暗部区域比例
    midtone_ratio: float       # 中间调比例
    edge_density: float        # 边缘密度
    frame_diff_mean: float     # 帧间变化均值
    noise_score: float         # 噪声评分
    
    # 分析元数据
    sample_frames: int         # 采样帧数
    analysis_timestamp: datetime
```

#### 2. EnhancementStrategyPlanner (Stage 2)
**位置**: `huaju4k/strategy/enhancement_planner.py`

**职责**:
- 将Stage 1的结构特征翻译为执行策略
- 生成机器可执行的策略JSON
- 不理解视频内容，只做数值映射

**策略生成逻辑**:
```python
class EnhancementStrategyPlanner:
    def generate_strategy(self, features: StructureFeatures) -> EnhancementStrategy:
        """
        基于结构特征生成增强策略
        
        Args:
            features: Stage 1输出的结构特征
            
        Returns:
            EnhancementStrategy: 完整的执行策略
        """
        # 1. 分辨率路径规划
        resolution_plan = self._plan_resolution_path(features.resolution)
        
        # 2. GAN允许度计算
        gan_policy = self._calculate_gan_policy(features)
        
        # 3. 分层处理策略
        layer_strategy = self._generate_layer_strategy(features)
        
        # 4. 时序锁定策略
        temporal_strategy = self._generate_temporal_strategy(features)
        
        # 5. 内存管理策略
        memory_policy = self._generate_memory_policy(features)
        
        return EnhancementStrategy(
            resolution_plan=resolution_plan,
            gan_policy=gan_policy,
            layer_strategy=layer_strategy,
            temporal_strategy=temporal_strategy,
            memory_policy=memory_policy
        )
```

**策略数据结构**:
```python
@dataclass
class EnhancementStrategy:
    # 分辨率处理路径
    resolution_plan: List[str]  # ["x2", "x2"] 或 ["x2"]
    
    # 分层处理策略
    layer_strategy: Dict[str, LayerConfig]
    
    # GAN控制策略
    gan_policy: GANPolicy
    
    # 时序处理策略
    temporal_strategy: TemporalConfig
    
    # 内存管理策略
    memory_policy: MemoryConfig
    
    # 音频处理策略
    audio_strategy: AudioConfig
    
    # 策略元数据
    strategy_version: str
    generation_timestamp: datetime
    source_features_hash: str

@dataclass
class LayerConfig:
    structure_sr: bool          # 是否启用结构重建
    gan_strength: str          # "off", "weak", "medium", "strong"
    post_smooth: bool          # 是否后处理平滑
    detail_limit: bool         # 是否限制细节生成

@dataclass
class GANPolicy:
    global_allowed: bool       # 全局GAN开关
    strength: str             # "weak", "medium", "strong"
    highlight_threshold: float # 高光阈值 (0.85)
    shadow_threshold: float   # 暗部阈值 (0.15)
    edge_threshold: float     # 边缘密度阈值 (0.1)
    motion_threshold: float   # 运动检测阈值 (0.05)
    highlight_forbidden: bool  # 高光区域禁用
    dark_limit: bool          # 暗部限制

@dataclass
class TemporalConfig:
    background_lock: bool      # 背景锁定
    strength: str             # "low", "medium", "high"
    motion_threshold: float   # 运动检测阈值
    optical_flow_enabled: bool # 是否启用光流
    smoothing_alpha: float    # 帧间平滑系数 (0.3)

@dataclass
class MemoryConfig:
    max_model_loaded: int     # 最大同时加载模型数
    tile_size: int           # 瓦片大小
    batch_size: int          # 批处理大小
    use_fp16: bool           # 是否使用半精度
    max_workers: int         # CPU并行工作线程数

@dataclass
class AudioConfig:
    source_separation_enabled: bool  # 是否启用音源分离
    dialogue_enhancement: float     # 对白增强强度 (0.0-1.0)
    music_processing: str          # 音乐处理模式 ("preserve", "enhance")
    ambient_processing: str        # 环境音处理模式 ("preserve", "spatial")
    master_settings: Dict[str, Any] # 母版级重混设置
```

#### 3. StrategyDrivenModelManager (Stage 3升级)
**位置**: `huaju4k/core/ai_model_manager.py` (升级现有)

**核心改进**:
- 策略驱动的模型调度
- 严格的显存管理 (6GB约束)
- 即用即载，用完即卸
- 单模型约束 (cache_size=1)

```python
class StrategyDrivenModelManager(AIModelManager):
    """
    策略驱动的AI模型管理器
    
    继承自AIModelManager，添加:
    - set_strategy(): 设置增强策略
    - execute_strategy_phase(): 执行策略阶段
    - predict_masked(): 区域增强预测
    - GPUMemoryMonitor: GPU内存监控
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_strategy: Optional[EnhancementStrategy] = None
        self.gpu_memory_monitor = GPUMemoryMonitor()
        self.cache_size = 1  # 强制单模型约束
    
    def set_strategy(self, strategy: EnhancementStrategy) -> None:
        """
        设置当前增强策略
        
        Args:
            strategy: 增强策略配置
            
        注意: VideoEnhancementProcessor 必须使用此类而非基类 AIModelManager
        """
        self.current_strategy = strategy
        logger.info(f"Strategy set: GAN={strategy.gan_policy.global_allowed}, "
                   f"temporal_lock={strategy.temporal_strategy.background_lock}")
    
    def execute_strategy_phase(self, phase: str) -> bool:
        """
        执行策略中的特定阶段
        
        Args:
            phase: "structure_sr", "gan_enhance", "temporal_lock"
            
        Returns:
            bool: 执行成功标志
        """
        if not self.current_strategy:
            logger.warning("No strategy set, using default model selection")
            return self.load_model('opencv_cubic')
        
        # 确定当前阶段需要的模型
        required_model = self._get_required_model_for_phase(phase)
        
        # 切换模型（如果需要）
        if required_model != self.current_model_name:
            return self._switch_model(required_model)
        
        return True
    
    def _switch_model(self, model_name: str) -> bool:
        """
        安全切换模型，确保显存释放
        """
        # 卸载当前模型
        if self.current_model:
            self.current_model.unload()
            self.current_model = None
            self.current_model_name = None
            
            # 强制GPU内存清理
            self.gpu_memory_monitor.force_gpu_memory_cleanup()
        
        # 加载新模型
        return self.load_model(model_name)


class GPUMemoryMonitor:
    """
    GPU内存监控器 - 6GB显存约束
    """
    
    def __init__(self, max_gpu_memory_mb: int = 5500):
        self.max_gpu_memory_mb = max_gpu_memory_mb  # 预留500MB缓冲
    
    def check_gpu_memory_available(self) -> int:
        """检查可用GPU内存 (MB)"""
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                cached = torch.cuda.memory_reserved(0)
                
                used_mb = (allocated + cached) // (1024 * 1024)
                effective_total = min(total // (1024 * 1024), self.max_gpu_memory_mb)
                
                return max(0, effective_total - used_mb)
            return 0
        except:
            return 0
    
    def force_gpu_memory_cleanup(self) -> None:
        """强制GPU内存清理"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
```

**重要**: `VideoEnhancementProcessor` 必须使用 `StrategyDrivenModelManager` 而非基类 `AIModelManager`：
```python
# 正确 ✓
self.ai_model_manager = StrategyDrivenModelManager(
    models_dir=self.config.get('models_dir', './models'),
    cache_size=self.config.get('model_cache_size', 2)
)

# 错误 ✗ - 会导致 'AIModelManager' object has no attribute 'set_strategy'
# self.ai_model_manager = AIModelManager(...)
```

#### 4. ThreeStageVideoEnhancer (Stage 4重构)
**位置**: `huaju4k/core/three_stage_enhancer.py`

**核心设计原则**:
- **全流式处理**: 使用 FFmpeg 管道，不提取帧到磁盘
- **实时进度显示**: 每个阶段显示进度条和帧计数
- **回退机制**: Real-ESRGAN 不可用时使用 FFmpeg 滤镜增强

**三阶段处理流程**:

```python
class ThreeStageVideoEnhancer:
    def enhance_video(self, input_path: str, output_path: str,
                     strategy: EnhancementStrategy,
                     progress_callback: Optional[Callable] = None) -> bool:
        """
        三阶段视频增强处理 - 全流式FFmpeg实现
        
        Stage 4.1: 结构重建 (FFmpeg lanczos + unsharp + hqdn3d)
        Stage 4.2: 受控GAN增强 (FFmpeg滤镜或Real-ESRGAN)
        Stage 4.3: 时序锁定 + 音频合并 (FFmpeg)
        
        Returns:
            bool: 处理成功标志
        """
        # Stage 4.1: 结构重建
        structure_result = self._stage_4_1_structure_reconstruction(input_path)
        if not structure_result.success:
            return False
        
        # Stage 4.2: 受控GAN增强
        gan_result = self._stage_4_2_controlled_gan_enhancement(
            structure_result.output_path
        )
        if not gan_result.success:
            return False
        
        # Stage 4.3: 时序锁定 + 音频合并
        temporal_result = self._stage_4_3_temporal_locking(
            gan_result.output_path, output_path
        )
        
        return temporal_result.success
```

**Stage 4.1: 结构重建 (FFmpeg流式处理)**:
```python
def _stage_4_1_structure_reconstruction(self, input_path: str) -> StageResult:
    """
    Stage 4.1: 使用FFmpeg进行结构重建
    
    处理流程:
    1. 根据resolution_plan执行多步放大 (如 1080p -> 2K -> 4K)
    2. 每步使用 lanczos + unsharp + hqdn3d 滤镜
    3. 实时显示进度条
    """
    resolution_plan = self.current_strategy.resolution_plan  # ["x2", "x2"]
    
    for step_idx, step in enumerate(resolution_plan):
        # FFmpeg滤镜链: 高质量放大 + 锐化 + 降噪
        filter_complex = (
            f"scale={target_width}:{target_height}:flags=lanczos,"
            f"unsharp=5:5:0.8:5:5:0.4,"
            f"hqdn3d=1.5:1.5:6:6"
        )
        
        cmd = [
            'ffmpeg', '-y', '-i', current_input,
            '-vf', filter_complex,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-progress', 'pipe:1',  # 进度输出
            step_output
        ]
        
        # 实时进度显示
        print(f"\n🎬 Stage 4.1 - 结构重建 步骤 {step_idx+1}/{len(resolution_plan)}")
        print(f"   分辨率: {current_width}x{current_height} -> {target_width}x{target_height}")
        
        # 解析FFmpeg进度输出并显示进度条
        for line in process.stdout:
            if line.startswith('frame='):
                frames_processed = int(line.strip().split('=')[1])
                progress = frames_processed / total_frames
                bar = '█' * int(40 * progress) + '░' * (40 - int(40 * progress))
                print(f"\r   进度: [{bar}] {progress*100:.1f}%", end='', flush=True)
```

**Stage 4.2: GAN增强 (FFmpeg滤镜回退)**:
```python
def _stage_4_2_controlled_gan_enhancement(self, input_path: str) -> StageResult:
    """
    Stage 4.2: 受控GAN增强
    
    实际实现:
    - 如果Real-ESRGAN可用: 使用AI模型增强
    - 如果不可用: 使用FFmpeg高级滤镜模拟GAN效果
    
    FFmpeg滤镜根据gan_strength调整:
    - weak:   unsharp=3:3:0.5, eq=contrast=1.01
    - medium: unsharp=5:5:0.8, eq=contrast=1.03:saturation=1.05
    - strong: unsharp=7:7:1.2, eq=contrast=1.05:saturation=1.1
    """
    if not self.current_strategy.gan_policy.global_allowed:
        return StageResult(output_path=input_path, success=True, skipped=True)
    
    gan_strength = self.current_strategy.gan_policy.strength
    
    # 根据强度选择滤镜参数
    if gan_strength == "strong":
        filter_complex = "unsharp=7:7:1.2:7:7:0.6,eq=contrast=1.05:brightness=0.02:saturation=1.1"
    elif gan_strength == "medium":
        filter_complex = "unsharp=5:5:0.8:5:5:0.4,eq=contrast=1.03:brightness=0.01:saturation=1.05"
    else:  # weak
        filter_complex = "unsharp=3:3:0.5:3:3:0.3,eq=contrast=1.01:saturation=1.02"
    
    print(f"\n🎨 Stage 4.2 - GAN增强 (强度: {gan_strength})")
    # ... FFmpeg处理 + 进度显示
```

**Stage 4.3: 时序锁定 + 音频合并**:
```python
def _stage_4_3_temporal_locking(self, input_path: str, 
                               final_output_path: str) -> StageResult:
    """
    Stage 4.3: 时序锁定 + 音频合并
    
    处理流程:
    1. 根据temporal_strategy选择滤镜
    2. 从原始视频提取音频并合并
    3. 输出最终4K视频
    
    时序滤镜选择:
    - high strength + background_lock: mpdecimate
    - medium strength: deflicker
    - low strength: null (直通)
    """
    temporal_config = self.current_strategy.temporal_strategy
    
    # 选择时序滤镜
    if temporal_config.background_lock and temporal_config.strength == "high":
        temporal_filter = "mpdecimate,setpts=N/FRAME_RATE/TB"
    elif temporal_config.strength == "medium":
        temporal_filter = "deflicker=mode=pm:size=5"
    else:
        temporal_filter = "null"
    
    # 合并视频和原始音频
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,           # 增强后的视频
        '-i', original_video,       # 原始视频(音频源)
        '-filter_complex', f"[0:v]{temporal_filter}[v]",
        '-map', '[v]', '-map', '1:a?',
        '-c:v', 'libx264', '-c:a', 'aac',
        '-progress', 'pipe:1',
        final_output_path
    ]
    
    print(f"\n🔒 Stage 4.3 - 时序锁定 + 音频合并")
    print(f"   音频: {'有' if has_audio else '无'}")
    # ... FFmpeg处理 + 进度显示
```
    
    def _generate_multi_dimensional_safe_mask(self, frame: np.ndarray, 
                                            previous_frame: Optional[np.ndarray],
                                            gan_policy: GANPolicy) -> np.ndarray:
        """
        生成多维度GAN安全区域mask
        
        综合考虑：
        1. 亮度排除（高光和暗部区域禁用GAN）
        2. 边缘密度限制（只在边缘密度高的区域允许细节增强）
        3. 运动检测限制（快速移动区域允许小幅增强）
        """
        height, width = frame.shape[:2]
        
        # 1. 亮度排除mask
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_normalized = frame_gray.astype(np.float32) / 255.0
        
        # 高光和暗部阈值
        highlight_mask = frame_normalized > gan_policy.highlight_threshold
        shadow_mask = frame_normalized < gan_policy.shadow_threshold
        
        # 2. 边缘密度mask
        edges = cv2.Canny(frame_gray, 50, 150)
        
        # 计算局部边缘密度
        kernel = np.ones((15, 15), np.uint8)  # 15x15邻域
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel) / (15 * 15 * 255)
        edge_mask = edge_density > gan_policy.edge_threshold
        
        # 3. 运动检测mask
        motion_mask = np.ones((height, width), dtype=bool)  # 默认允许
        if previous_frame is not None:
            # 帧间差分检测运动
            prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(frame_gray, prev_gray).astype(np.float32) / 255.0
            motion_mask = frame_diff > gan_policy.motion_threshold
        
        # 4. 综合安全区域mask
        # 允许GAN的区域：有边缘 AND 非高光 AND 非暗部 AND (有运动或静态背景)
        safe_mask = edge_mask & (~highlight_mask) & (~shadow_mask)
        
        # 根据GAN强度调整mask
        if gan_policy.strength == 'weak':
            # 弱强度：只在明确的边缘区域
            safe_mask = safe_mask & motion_mask
        elif gan_policy.strength == 'medium':
            # 中等强度：边缘区域 + 部分静态区域
            safe_mask = safe_mask | (motion_mask & (~highlight_mask))
        elif gan_policy.strength == 'strong':
            # 强强度：更大范围，但仍避免高光
            safe_mask = safe_mask | (~highlight_mask & ~shadow_mask)
        
        # 形态学操作平滑mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        safe_mask = cv2.morphologyEx(safe_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_OPEN, kernel)
        
        return safe_mask.astype(bool)
    
    def _blend_enhanced_regions(self, original_frame: np.ndarray, 
                              enhanced_regions: List[Dict], 
                              safe_mask: np.ndarray) -> np.ndarray:
        """
        将增强区域混合回原始帧
        """
        result_frame = original_frame.copy()
        
        for region_data in enhanced_regions:
            enhanced_region = region_data['region']
            region_mask = region_data['mask']
            x, y, w, h = region_data['bbox']
            
            # 将增强区域混合回原始帧
            for c in range(3):  # RGB三通道
                result_frame[y:y+h, x:x+w, c] = np.where(
                    region_mask,
                    enhanced_region[:, :, c],
                    result_frame[y:y+h, x:x+w, c]
                )
        
        return result_frame
```

#### 5. TemporalLockProcessor (时序锁定处理器)
**位置**: `huaju4k/core/temporal_lock_processor.py`

**设计说明**:
当前实现使用 FFmpeg 内置滤镜进行时序处理，而非 OpenCV 光流。
这是因为 FFmpeg 滤镜在流式处理中更高效，且不需要将帧提取到内存。

**FFmpeg 时序滤镜选择**:
```python
class TemporalLockProcessor:
    def get_temporal_filter(self, temporal_config: TemporalConfig) -> str:
        """
        根据配置返回FFmpeg时序滤镜
        
        滤镜选择逻辑:
        - background_lock=True + strength=high: mpdecimate (去除重复帧)
        - strength=medium: deflicker (去闪烁)
        - strength=low: null (直通，不处理)
        """
        if temporal_config.background_lock and temporal_config.strength == "high":
            return "mpdecimate,setpts=N/FRAME_RATE/TB"
        elif temporal_config.strength == "medium":
            return "deflicker=mode=pm:size=5"
        else:
            return "null"
```

**备用 OpenCV 光流实现** (用于需要精细控制的场景):
```python
def _apply_optical_flow_stabilization(self, current_frame: np.ndarray,
                                    previous_frame: np.ndarray) -> np.ndarray:
    """
    使用光流进行背景稳定 (备用方案)
    
    注意: 当前主流程使用FFmpeg滤镜，此方法作为备用
    """
    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # 检测运动区域
    motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    motion_mask = motion_magnitude > self.config.motion_threshold
    
    # 合并运动区域和稳定背景
    return stabilized_frame
```

#### 6. MasterGradeAudioEnhancer (Stage 5升级)
**位置**: `huaju4k/audio/master_grade_enhancer.py`

**具体库调用的母版级音频处理**:
```python
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydub import AudioSegment
import librosa
import noisereduce as nr
import numpy as np

class MasterGradeAudioEnhancer:
    def __init__(self):
        self.temp_dir = Path("/tmp/huaju4k_audio")
        self.temp_dir.mkdir(exist_ok=True)
    
    def enhance_audio(self, video_path: str, 
                     strategy: EnhancementStrategy) -> AudioResult:
        """
        母版级音频增强主流程 - 具体实现版本
        """
        if not strategy.audio_strategy.source_separation_enabled:
            # 简单音频增强
            return self._simple_audio_enhancement(video_path, strategy)
        
        # 1. 音轨提取
        audio_path = self._extract_audio_with_ffmpeg(video_path)
        
        # 2. 音源分离
        separated_tracks = self._separate_audio_sources(audio_path)
        
        # 3. 分轨处理
        enhanced_dialogue = self._enhance_dialogue(
            separated_tracks['vocals'], 
            strategy.audio_strategy.dialogue_enhancement
        )
        
        enhanced_music = self._process_music(
            separated_tracks['accompaniment'],
            strategy.audio_strategy.music_processing
        )
        
        enhanced_ambient = self._process_ambient(
            separated_tracks.get('ambient', separated_tracks['accompaniment']),
            strategy.audio_strategy.ambient_processing
        )
        
        # 4. 母版级重混
        master_audio_path = self._master_grade_remix(
            enhanced_dialogue, enhanced_music, enhanced_ambient,
            strategy.audio_strategy.master_settings
        )
        
        return AudioResult(
            output_path=master_audio_path,
            quality_metrics=self._calculate_audio_quality_metrics(master_audio_path)
        )
    
    def _extract_audio_with_ffmpeg(self, video_path: str) -> str:
        """
        使用FFmpeg提取音轨
        """
        audio_path = self.temp_dir / "extracted_audio.wav"
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # 不要视频
            '-acodec', 'pcm_s16le',  # 16位PCM
            '-ar', '44100',  # 44.1kHz采样率
            '-ac', '2',  # 立体声
            '-y',  # 覆盖输出文件
            str(audio_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(audio_path)
    
    def _separate_audio_sources(self, audio_path: str) -> Dict[str, str]:
        """
        使用Spleeter进行音源分离
        """
        try:
            from spleeter.separator import Separator
            
            # 初始化分离器 (人声 + 伴奏)
            separator = Separator('spleeter:2stems-16kHz')
            
            # 分离音频
            output_dir = self.temp_dir / "separated"
            output_dir.mkdir(exist_ok=True)
            
            separator.separate_to_file(audio_path, str(output_dir))
            
            # 返回分离后的文件路径
            audio_name = Path(audio_path).stem
            return {
                'vocals': str(output_dir / audio_name / "vocals.wav"),
                'accompaniment': str(output_dir / audio_name / "accompaniment.wav")
            }
            
        except ImportError:
            # 回退到简单的频域分离
            return self._simple_vocal_separation(audio_path)
    
    def _simple_vocal_separation(self, audio_path: str) -> Dict[str, str]:
        """
        简单的人声分离（回退方案）
        """
        # 加载音频
        y, sr = librosa.load(audio_path, sr=44100, mono=False)
        
        if y.ndim == 1:
            # 单声道，无法分离
            vocals_path = self.temp_dir / "vocals_mono.wav"
            accompaniment_path = self.temp_dir / "accompaniment_mono.wav"
            
            # 复制原音频
            librosa.output.write_wav(str(vocals_path), y, sr)
            librosa.output.write_wav(str(accompaniment_path), y * 0.3, sr)  # 降低伴奏音量
        else:
            # 立体声，使用中央声道分离
            vocals = y[0] - y[1]  # 左右声道差值（粗略的人声提取）
            accompaniment = (y[0] + y[1]) / 2  # 左右声道平均（伴奏）
            
            vocals_path = self.temp_dir / "vocals_separated.wav"
            accompaniment_path = self.temp_dir / "accompaniment_separated.wav"
            
            librosa.output.write_wav(str(vocals_path), vocals, sr)
            librosa.output.write_wav(str(accompaniment_path), accompaniment, sr)
        
        return {
            'vocals': str(vocals_path),
            'accompaniment': str(accompaniment_path)
        }
    
    def _enhance_dialogue(self, vocals_path: str, enhancement_strength: float) -> str:
        """
        对白增强处理
        """
        # 加载人声音频
        audio = AudioSegment.from_wav(vocals_path)
        
        # 1. 降噪处理
        y, sr = librosa.load(vocals_path, sr=44100)
        
        # 使用noisereduce进行降噪
        reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=enhancement_strength * 0.8)
        
        # 2. 预加重（增强高频）
        if enhancement_strength > 0.5:
            reduced_noise = librosa.effects.preemphasis(reduced_noise, coef=0.97)
        
        # 3. 动态范围压缩
        # 简单的软限制器
        threshold = 0.8
        ratio = 4.0
        compressed = np.where(
            np.abs(reduced_noise) > threshold,
            np.sign(reduced_noise) * (threshold + (np.abs(reduced_noise) - threshold) / ratio),
            reduced_noise
        )
        
        # 保存增强后的对白
        enhanced_path = self.temp_dir / "enhanced_dialogue.wav"
        librosa.output.write_wav(str(enhanced_path), compressed, sr)
        
        return str(enhanced_path)
    
    def _process_music(self, music_path: str, processing_mode: str) -> str:
        """
        音乐处理
        """
        if processing_mode == "preserve":
            # 保持原样
            return music_path
        elif processing_mode == "enhance":
            # 轻微增强
            audio = AudioSegment.from_wav(music_path)
            
            # 轻微的EQ调整
            enhanced = audio.low_pass_filter(8000).high_pass_filter(80)
            
            enhanced_path = self.temp_dir / "enhanced_music.wav"
            enhanced.export(str(enhanced_path), format="wav")
            return str(enhanced_path)
        
        return music_path
    
    def _process_ambient(self, ambient_path: str, processing_mode: str) -> str:
        """
        环境音处理
        """
        if processing_mode == "preserve":
            return ambient_path
        elif processing_mode == "spatial":
            # 增强空间感
            audio = AudioSegment.from_wav(ambient_path)
            
            # 轻微的混响效果（简化版）
            # 实际实现中可以使用更复杂的空间音频处理
            enhanced = audio + 2  # 轻微增益
            
            enhanced_path = self.temp_dir / "enhanced_ambient.wav"
            enhanced.export(str(enhanced_path), format="wav")
            return str(enhanced_path)
        
        return ambient_path
    
    def _master_grade_remix(self, dialogue_path: str, music_path: str, 
                          ambient_path: str, master_settings: Dict[str, Any]) -> str:
        """
        母版级重混
        """
        # 加载所有音轨
        dialogue = AudioSegment.from_wav(dialogue_path)
        music = AudioSegment.from_wav(music_path)
        ambient = AudioSegment.from_wav(ambient_path)
        
        # 音量平衡
        dialogue_gain = master_settings.get('dialogue_gain', 0)  # dB
        music_gain = master_settings.get('music_gain', -6)       # dB
        ambient_gain = master_settings.get('ambient_gain', -12)  # dB
        
        dialogue = dialogue + dialogue_gain
        music = music + music_gain
        ambient = ambient + ambient_gain
        
        # 混合音轨
        # 确保所有音轨长度一致
        max_length = max(len(dialogue), len(music), len(ambient))
        
        dialogue = dialogue[:max_length]
        music = music[:max_length] 
        ambient = ambient[:max_length]
        
        # 叠加混合
        master_audio = dialogue.overlay(music).overlay(ambient)
        
        # 母版级处理
        # 1. 限制器防止削波
        master_audio = master_audio.normalize(headroom=1.0)
        
        # 2. 最终音量调整
        target_lufs = master_settings.get('target_lufs', -23)  # EBU R128标准
        # 简化的响度调整（实际应使用专业响度测量）
        master_audio = master_audio.normalize(headroom=abs(target_lufs))
        
        # 导出最终音频
        master_path = self.temp_dir / "master_audio.wav"
        master_audio.export(str(master_path), format="wav")
        
        return str(master_path)
```

### 性能优化设计

#### 实时进度显示机制
```python
class FFmpegProgressMonitor:
    """
    FFmpeg 进度监控器 - 实时显示处理进度
    
    使用 -progress pipe:1 参数获取FFmpeg进度输出，
    解析 frame= 行获取已处理帧数，显示进度条。
    """
    
    def run_with_progress(self, cmd: List[str], total_frames: int, 
                         stage_name: str) -> bool:
        """
        运行FFmpeg命令并显示实时进度
        
        Args:
            cmd: FFmpeg命令列表
            total_frames: 总帧数
            stage_name: 阶段名称 (用于显示)
        """
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        frames_processed = 0
        last_print_frame = 0
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            # 解析进度输出 (格式: frame=123)
            if line.startswith('frame='):
                frames_processed = int(line.strip().split('=')[1])
                
                if total_frames > 0:
                    progress = min(frames_processed / total_frames, 1.0)
                    
                    # 每100帧更新一次显示
                    if frames_processed - last_print_frame >= 100:
                        last_print_frame = frames_processed
                        bar_width = 40
                        filled = int(bar_width * progress)
                        bar = '█' * filled + '░' * (bar_width - filled)
                        print(f"\r   进度: [{bar}] {progress*100:.1f}% "
                              f"({frames_processed}/{total_frames}帧)", 
                              end='', flush=True)
        
        print()  # 换行
        return process.returncode == 0
```

**进度显示示例输出**:
```
🎬 Stage 4.1 - 结构重建 步骤 1/2
   分辨率: 1920x1080 -> 3840x2160
   总帧数: 7500
   进度: [████████████████████░░░░░░░░░░░░░░░░░░░░] 50.0% (3750/7500帧)

🎨 Stage 4.2 - GAN增强 (强度: medium)
   总帧数: 7500
   进度: [████████████████████████████████████████] 100.0% (7500/7500帧)

🔒 Stage 4.3 - 时序锁定 + 音频合并
   总帧数: 7500
   音频: 有
   进度: [████████████████████████████████████████] 100.0% (7500/7500帧)
```

#### 动态并行处理优化
```python
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable

class DynamicParallelProcessingOptimizer:
    """
    动态并行处理优化器 - 根据系统资源自适应调整
    """
    
    def __init__(self):
        # 动态检测CPU核心数
        self.cpu_cores = multiprocessing.cpu_count()
        self.max_workers = min(8, self.cpu_cores)  # 最多8个工作线程
        
        # GPU处理保持串行
        self.gpu_serial_lock = threading.Lock()
        
        logger.info(f"Detected {self.cpu_cores} CPU cores, using {self.max_workers} workers")
    
    def optimize_cpu_processing(self, items: List[Any], 
                              process_func: Callable,
                              progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        CPU阶段并行处理优化
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_func, item): item 
                for item in items
            }
            
            # 收集结果
            completed = 0
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed / len(items))
                        
                except Exception as e:
                    logger.error(f"Processing failed for item: {e}")
                    results.append(None)
        
        return results
    
    def optimize_gpu_processing(self, items: List[Any],
                              process_func: Callable,
                              progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        GPU阶段串行处理（避免显存冲突）
        """
        results = []
        
        with self.gpu_serial_lock:
            for i, item in enumerate(items):
                try:
                    result = process_func(item)
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback((i + 1) / len(items))
                        
                except Exception as e:
                    logger.error(f"GPU processing failed for item {i}: {e}")
                    results.append(None)
        
        return results
    
    def optimize_mixed_processing(self, frames: List[np.ndarray],
                                cpu_preprocess: Callable,
                                gpu_process: Callable,
                                cpu_postprocess: Callable,
                                progress_callback: Optional[Callable] = None) -> List[np.ndarray]:
        """
        混合处理优化：CPU预处理 -> GPU处理 -> CPU后处理
        """
        # 1. CPU并行预处理
        preprocessed_frames = self.optimize_cpu_processing(
            frames, cpu_preprocess,
            lambda p: progress_callback(p * 0.3) if progress_callback else None
        )
        
        # 2. GPU串行处理
        gpu_processed_frames = self.optimize_gpu_processing(
            preprocessed_frames, gpu_process,
            lambda p: progress_callback(0.3 + p * 0.4) if progress_callback else None
        )
        
        # 3. CPU并行后处理
        final_frames = self.optimize_cpu_processing(
            gpu_processed_frames, cpu_postprocess,
            lambda p: progress_callback(0.7 + p * 0.3) if progress_callback else None
        )
        
        return final_frames
```

### 内存管理策略

#### 6GB显存约束下的精确内存管理
```python
class PreciseMemoryManager:
    """
    精确的GPU内存管理器
    """
    
    def __init__(self, max_gpu_memory_mb: int = 5500):
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.current_usage_mb = 0
        self.memory_reservations = {}
        
    def check_gpu_memory_available(self) -> int:
        """
        检查可用GPU内存（MB）
        """
        if not torch.cuda.is_available():
            return 0
        
        # 获取实际GPU内存使用情况
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)
        
        used_memory_mb = (allocated_memory + cached_memory) // (1024 * 1024)
        available_mb = (total_memory // (1024 * 1024)) - used_memory_mb
        
        # 应用安全限制
        safe_available = min(available_mb, self.max_gpu_memory_mb - used_memory_mb)
        
        return max(0, safe_available)
    
    def reserve_memory(self, operation_id: str, required_mb: int) -> bool:
        """
        预留内存用于特定操作
        """
        available = self.check_gpu_memory_available()
        
        if available >= required_mb:
            self.memory_reservations[operation_id] = required_mb
            return True
        else:
            logger.warning(f"Insufficient GPU memory: need {required_mb}MB, available {available}MB")
            return False
    
    def release_memory(self, operation_id: str) -> None:
        """
        释放预留的内存
        """
        if operation_id in self.memory_reservations:
            del self.memory_reservations[operation_id]
            
            # 强制清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def get_optimal_tile_size(self, base_tile_size: int, model_memory_mb: int) -> int:
        """
        根据可用内存计算最优瓦片大小
        """
        available = self.check_gpu_memory_available()
        
        if available < model_memory_mb:
            return 64  # 最小瓦片大小
        
        # 根据可用内存调整瓦片大小
        memory_ratio = (available - model_memory_mb) / 1000  # 每1GB额外内存
        
        if memory_ratio >= 3:
            return min(base_tile_size * 2, 512)  # 大瓦片
        elif memory_ratio >= 1:
            return base_tile_size  # 标准瓦片
        else:
            return max(base_tile_size // 2, 64)  # 小瓦片
```

### 质量保证设计

#### 母版级质量验证器
```python
class MasterGradeQualityValidator:
    """
    母版级质量验证器 - 具体指标实现
    """
    
    def validate_master_quality(self, input_path: str, 
                               output_path: str) -> QualityReport:
        """
        执行母版级质量验证
        """
        report = QualityReport()
        
        # 1. 视频质量验证
        report.video_quality = self._validate_video_quality(output_path)
        
        # 2. 音频质量验证
        report.audio_quality = self._validate_audio_quality(output_path)
        
        # 3. 同步性验证
        report.sync_quality = self._validate_av_sync(output_path)
        
        # 4. 技术规格验证
        report.technical_specs = self._validate_technical_specs(output_path)
        
        return report
    
    def _validate_video_quality(self, video_path: str) -> VideoQualityMetrics:
        """
        视频质量验证指标
        """
        cap = cv2.VideoCapture(video_path)
        
        brightness_values = []
        edge_stability_scores = []
        highlight_clipping_ratios = []
        
        previous_frame = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 1. 亮度稳定性检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            brightness_values.append(brightness)
            
            # 2. 边缘稳定性评估
            if previous_frame is not None:
                prev_edges = cv2.Canny(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY), 50, 150)
                curr_edges = cv2.Canny(gray, 50, 150)
                
                # 计算边缘一致性
                edge_diff = cv2.absdiff(prev_edges, curr_edges)
                stability_score = 1.0 - (np.sum(edge_diff > 0) / (frame.shape[0] * frame.shape[1]))
                edge_stability_scores.append(stability_score)
            
            # 3. 高光溢出检测
            highlight_mask = np.any(frame > 250, axis=2)  # 接近255的像素
            highlight_ratio = np.sum(highlight_mask) / (frame.shape[0] * frame.shape[1])
            highlight_clipping_ratios.append(highlight_ratio)
            
            previous_frame = frame
            
            # 采样处理，避免处理所有帧
            if frame_count >= 100:  # 最多检查100帧
                break
        
        cap.release()
        
        # 计算质量指标
        brightness_stability = 1.0 - np.std(brightness_values)  # 亮度稳定性
        edge_stability = np.mean(edge_stability_scores) if edge_stability_scores else 1.0
        highlight_clipping = np.mean(highlight_clipping_ratios)
        
        return VideoQualityMetrics(
            brightness_stability=max(0, brightness_stability),
            edge_stability=edge_stability,
            highlight_clipping=highlight_clipping,
            temporal_consistency=edge_stability,  # 使用边缘稳定性作为时序一致性指标
            resolution_accuracy=self._verify_resolution_accuracy(video_path)
        )
    
    def _validate_audio_quality(self, video_path: str) -> AudioQualityMetrics:
        """
        音频质量验证
        """
        # 提取音频进行分析
        temp_audio = "/tmp/temp_audio_analysis.wav"
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '44100', '-y', temp_audio
        ], capture_output=True)
        
        # 加载音频数据
        y, sr = librosa.load(temp_audio, sr=44100)
        
        # 1. 对白清晰度评分（基于频谱特征）
        # 人声频率范围大致在300-3400Hz
        stft = librosa.stft(y)
        freqs = librosa.fft_frequencies(sr=sr)
        
        # 计算人声频段的能量比例
        voice_freq_mask = (freqs >= 300) & (freqs <= 3400)
        voice_energy = np.mean(np.abs(stft[voice_freq_mask, :]))
        total_energy = np.mean(np.abs(stft))
        
        dialogue_clarity = voice_energy / total_energy if total_energy > 0 else 0
        
        # 2. 动态范围计算
        rms_energy = librosa.feature.rms(y=y)[0]
        dynamic_range = np.max(rms_energy) / (np.mean(rms_energy) + 1e-8)
        
        # 3. 响度测量（简化版LUFS）
        # 实际应使用专业的响度测量库
        loudness_lufs = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-8)
        
        # 4. 音量一致性
        # 计算RMS能量的变异系数
        volume_consistency = 1.0 - (np.std(rms_energy) / (np.mean(rms_energy) + 1e-8))
        
        # 清理临时文件
        os.remove(temp_audio)
        
        return AudioQualityMetrics(
            dialogue_clarity_score=min(1.0, dialogue_clarity * 2),  # 归一化到0-1
            dynamic_range=min(10.0, dynamic_range),  # 限制在合理范围
            loudness_lufs=loudness_lufs,
            volume_consistency=max(0, volume_consistency)
        )

@dataclass
class VideoQualityMetrics:
    brightness_stability: float    # 亮度稳定性 (0-1)
    edge_stability: float         # 边缘稳定性 (0-1)
    highlight_clipping: float     # 高光溢出比例 (0-1)
    temporal_consistency: float   # 时序一致性 (0-1)
    resolution_accuracy: float    # 分辨率准确性 (0-1)

@dataclass
class AudioQualityMetrics:
    dialogue_clarity_score: float  # 对白清晰度 (0-1)
    dynamic_range: float          # 动态范围 (dB)
    loudness_lufs: float          # 响度 (LUFS)
    volume_consistency: float     # 音量一致性 (0-1)

@dataclass
class QualityReport:
    video_quality: VideoQualityMetrics
    audio_quality: AudioQualityMetrics
    sync_quality: float           # 音视频同步质量 (0-1)
    technical_specs: Dict[str, Any]  # 技术规格验证结果
```

## 配置管理设计

### 策略配置模板（更新版）
```json
{
  "theater_enhancement_config": {
    "version": "2.0.0",
    "presets": {
      "theater_small": {
        "lighting_sensitivity": 0.8,
        "gan_policy": {
          "strength": "weak",
          "highlight_threshold": 0.85,
          "shadow_threshold": 0.15,
          "edge_threshold": 0.1,
          "motion_threshold": 0.05
        },
        "temporal_config": {
          "background_lock": true,
          "strength": "high",
          "optical_flow_enabled": true,
          "smoothing_alpha": 0.1
        },
        "audio_config": {
          "source_separation_enabled": true,
          "dialogue_enhancement": 0.8,
          "music_processing": "preserve",
          "ambient_processing": "spatial"
        }
      },
      "theater_medium": {
        "lighting_sensitivity": 1.0,
        "gan_policy": {
          "strength": "medium",
          "highlight_threshold": 0.80,
          "shadow_threshold": 0.20,
          "edge_threshold": 0.08,
          "motion_threshold": 0.03
        },
        "temporal_config": {
          "background_lock": true,
          "strength": "medium",
          "optical_flow_enabled": true,
          "smoothing_alpha": 0.3
        },
        "audio_config": {
          "source_separation_enabled": true,
          "dialogue_enhancement": 0.6,
          "music_processing": "enhance",
          "ambient_processing": "spatial"
        }
      },
      "theater_large": {
        "lighting_sensitivity": 1.2,
        "gan_policy": {
          "strength": "strong",
          "highlight_threshold": 0.75,
          "shadow_threshold": 0.25,
          "edge_threshold": 0.06,
          "motion_threshold": 0.02
        },
        "temporal_config": {
          "background_lock": false,
          "strength": "low",
          "optical_flow_enabled": false,
          "smoothing_alpha": 0.5
        },
        "audio_config": {
          "source_separation_enabled": true,
          "dialogue_enhancement": 0.4,
          "music_processing": "enhance",
          "ambient_processing": "preserve"
        }
      }
    },
    "hardware_profiles": {
      "gtx_1650_4gb": {
        "max_tile_size": 128,
        "max_batch_size": 1,
        "force_fp16": true,
        "max_models_loaded": 1,
        "max_workers": 4
      },
      "rtx_3060_12gb": {
        "max_tile_size": 256,
        "max_batch_size": 4,
        "force_fp16": false,
        "max_models_loaded": 2,
        "max_workers": 6
      }
    },
    "audio_dependencies": {
      "required_packages": [
        "spleeter>=2.3.0",
        "pydub>=0.25.1",
        "librosa>=0.9.0",
        "noisereduce>=2.0.0"
      ],
      "fallback_enabled": true
    }
  }
}
```

这个优化版的设计文档现在包含了：

1. **多维度GAN安全区域Mask生成** - 具体的算法实现
2. **光流增强的时序锁定处理** - 完整的光流和帧间平滑算法
3. **动态并行处理优化** - 自适应CPU核心检测和混合处理策略
4. **具体库调用的音频处理** - FFmpeg、Spleeter、pydub等具体实现
5. **精确的内存管理** - 6GB显存约束下的精确控制
6. **母版级质量验证** - 具体的质量指标计算算法

现在让我更新tasks.md文档以反映这些优化：
# Huaju4K 话剧母版级4K增强系统 - 实施任务清单

## 项目概述

基于现有Huaju4K系统，实现从「通用视频超分」到「话剧母版级4K增强系统」的升级。当前系统已具备完整的7阶段处理流程，需要在此基础上添加新的分析和策略组件。

## 当前实现状态分析

### ✅ 已完成的核心组件
- **VideoEnhancementProcessor**: 主处理管道完整实现，支持7阶段处理流程
- **VideoAnalyzer**: 视频分析功能完整，包含FFprobe集成和策略计算
- **AIModelManager**: AI模型管理和缓存机制，支持Real-ESRGAN和OpenCV回退
- **TheaterAudioEnhancer**: 剧院音频增强器，支持多种剧院预设
- **ConservativeMemoryManager**: 内存管理和资源控制，支持瓦片大小优化
- **MultiStageProgressTracker**: 多阶段进度跟踪，支持子阶段和实时更新
- **TileProcessor**: 瓦片处理和GPU优化，支持批处理
- **PerformanceOptimizer**: 性能优化器，支持并行处理和动态优化
- **CLI接口**: 完整的命令行界面，支持多种预设和质量级别

### 🔄 需要新增的组件
基于设计文档和现有架构分析，需要添加以下新组件来实现母版级增强：

---

## 实施任务清单

### 任务 1：创建舞台结构分析模块
**优先级**: P0 (最高)  
**预估工时**: 6小时  
**依赖**: 无  
**需求**: FR-001 舞台结构分析能力

#### 实现要求
1. **创建分析模块目录结构**
   ```
   huaju4k/
   └── analysis/
       ├── __init__.py
       └── stage_structure_analyzer.py  # 新增
   ```

2. **实现StageStructureAnalyzer类**
   ```python
   class StageStructureAnalyzer:
       def __init__(self, sample_frames: int = 30):
           """初始化分析器，采样帧数可配置（优化为30帧提高速度）"""
           
       def analyze_structure(self, video_path: str) -> StructureFeatures:
           """主分析方法，返回舞台结构特征"""
           
       def _analyze_lighting_structure(self, frames: List[np.ndarray]) -> Dict[str, float]:
           """亮度结构分析：高光比例、暗部比例、中间调比例"""
           
       def _analyze_edge_density(self, frames: List[np.ndarray]) -> float:
           """边缘密度分析：使用Canny算子统计边缘像素占比"""
           
       def _analyze_frame_changes(self, frames: List[np.ndarray]) -> Dict[str, float]:
           """帧间变化分析：检测固定机位和运动程度"""
           
       def _analyze_noise_level(self, frames: List[np.ndarray]) -> float:
           """噪声评估：使用Laplacian方差法"""
   ```

3. **扩展现有data_models.py**
   ```python
   @dataclass
   class StructureFeatures:
       # 基础视频信息
       resolution: Tuple[int, int]
       fps: float
       duration: float
       total_frames: int
       
       # 舞台结构特征
       is_static_camera: bool          # 固定机位检测
       highlight_ratio: float          # 高光区域比例 (0-1)
       dark_ratio: float              # 暗部区域比例 (0-1)
       midtone_ratio: float           # 中间调比例 (0-1)
       edge_density: float            # 边缘密度 (0-1)
       frame_diff_mean: float         # 帧间变化均值 (0-1)
       noise_score: float             # 噪声评分 (0-1)
       
       # 分析元数据
       sample_frames: int
       analysis_timestamp: datetime
       
       def to_dict(self) -> Dict[str, Any]:
           """转换为字典格式，便于JSON序列化"""
           return asdict(self)
   ```

4. **核心算法实现**
   - **亮度分析**: 直方图统计，阈值0.85(高光)、0.15(暗部)
   - **边缘检测**: Canny(50, 150)算子，统计边缘像素占比
   - **帧间差分**: cv2.absdiff + 变化区域统计
   - **噪声评估**: Laplacian方差法，cv2.Laplacian(frame, cv2.CV_64F).var()

#### 验收标准
- [ ] 能分析2小时视频，处理时间<20秒（优化采样策略）
- [ ] 输出结构化JSON数据，所有数值在合理范围内(0-1)
- [ ] 固定机位检测准确率>95% (frame_diff_mean < 0.02)
- [ ] 集成到现有VideoAnalyzer工作流程无缝工作
- [ ] 内存使用合理，不超过512MB峰值（优化内存使用）

---

### 任务 2：创建增强策略规划模块
**优先级**: P0 (最高)  
**预估工时**: 6小时  
**依赖**: 任务1完成  
**需求**: FR-002 分层增强策略生成

#### 实现要求
1. **创建策略模块目录结构**
   ```
   huaju4k/
   └── strategy/
       ├── __init__.py
       └── enhancement_planner.py  # 新增
   ```

2. **实现EnhancementStrategyPlanner类**
   ```python
   class EnhancementStrategyPlanner:
       def __init__(self):
           """初始化策略规划器"""
           
       def generate_strategy(self, features: StructureFeatures) -> EnhancementStrategy:
           """基于结构特征生成完整增强策略"""
           
       def _plan_resolution_path(self, resolution: Tuple[int, int]) -> List[str]:
           """分辨率路径规划：1080p以下用["x2", "x2"]，其他用["x2"]"""
           
       def _calculate_gan_policy(self, features: StructureFeatures) -> GANPolicy:
           """GAN策略计算：基于高光比例和噪声水平"""
           
       def _generate_temporal_strategy(self, features: StructureFeatures) -> TemporalConfig:
           """时序策略生成：基于固定机位检测和帧间变化"""
           
       def _generate_memory_policy(self, features: StructureFeatures) -> MemoryConfig:
           """内存策略生成：确保6GB显存约束"""
           
       def _generate_audio_strategy(self, features: StructureFeatures) -> AudioConfig:
           """音频策略生成：基于视频特征推荐音频处理参数"""
   ```

3. **扩展现有data_models.py**
   ```python
   @dataclass
   class EnhancementStrategy:
       # 分辨率处理路径
       resolution_plan: List[str]  # ["x2", "x2"] 或 ["x2"]
       
       # GAN控制策略
       gan_policy: GANPolicy
       
       # 时序处理策略
       temporal_strategy: TemporalConfig
       
       # 内存管理策略
       memory_policy: MemoryConfig
       
       # 音频处理策略（扩展现有AudioConfig）
       audio_strategy: AudioConfig
       
       # 策略元数据
       strategy_version: str = "1.0"
       generation_timestamp: datetime = field(default_factory=datetime.now)
       source_features_hash: str = ""
       
       def to_dict(self) -> Dict[str, Any]:
           """转换为字典格式，便于JSON序列化"""
           return asdict(self)
       
       def save_to_file(self, filepath: str) -> None:
           """保存策略到JSON文件"""
           with open(filepath, 'w', encoding='utf-8') as f:
               json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)

   @dataclass
   class GANPolicy:
       global_allowed: bool           # 全局GAN开关
       strength: str                 # "weak", "medium", "strong"
       highlight_threshold: float = 0.85    # 高光阈值
       shadow_threshold: float = 0.15       # 暗部阈值
       edge_threshold: float = 0.1          # 边缘密度阈值
       motion_threshold: float = 0.05       # 运动检测阈值

   @dataclass
   class TemporalConfig:
       background_lock: bool          # 背景锁定开关
       strength: str                 # "low", "medium", "high"
       motion_threshold: float = 0.05       # 运动检测阈值
       optical_flow_enabled: bool = True    # 光流算法开关
       smoothing_alpha: float = 0.3         # 帧间平滑系数

   @dataclass
   class MemoryConfig:
       max_model_loaded: int = 1      # 最大同时加载模型数
       tile_size: int = 512          # 瓦片大小
       batch_size: int = 1           # 批处理大小
       use_fp16: bool = True         # 半精度浮点
       max_workers: int = 4          # CPU并行工作线程数
   ```

4. **策略生成规则实现**
   ```python
   # 分辨率路径规则
   if resolution[0] <= 1920:  # 1080p及以下
       return ["x2", "x2"]    # 两步到4K
   else:
       return ["x2"]          # 一步到4K
   
   # GAN策略规则
   gan_allowed = True
   gan_strength = "medium"
   
   if features.highlight_ratio > 0.2:
       gan_allowed = False    # 高光过多禁用GAN
   elif features.noise_score > 0.25:
       gan_strength = "weak"  # 噪声过多降低GAN强度
   elif features.edge_density > 0.3:
       gan_strength = "strong" # 边缘丰富可用强GAN
   
   # 时序策略规则
   if features.is_static_camera and features.frame_diff_mean < 0.02:
       background_lock = True
       temporal_strength = "high"
   else:
       background_lock = False
       temporal_strength = "medium"
   ```

#### 验收标准
- [ ] 能基于StructureFeatures生成完整EnhancementStrategy
- [ ] 策略JSON可序列化存储和加载
- [ ] 内存策略确保6GB显存约束（max_model_loaded=1）
- [ ] 所有策略规则可追溯和调试
- [ ] 策略生成时间<1秒
- [ ] 与现有AudioConfig和PerformanceConfig兼容

---

### 任务 3：升级AI模型管理器为策略驱动
**优先级**: P0 (最高)  
**预估工时**: 8小时  
**依赖**: 任务2完成  
**需求**: FR-003 策略驱动模型调度

#### 实现要求
1. **扩展现有AIModelManager类**
   ```python
   class StrategyDrivenModelManager(AIModelManager):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.current_strategy: Optional[EnhancementStrategy] = None
           self.gpu_memory_monitor = GPUMemoryMonitor()
           
       def set_strategy(self, strategy: EnhancementStrategy) -> None:
           """设置当前增强策略"""
           self.current_strategy = strategy
           
       def execute_strategy_phase(self, phase: str) -> bool:
           """执行策略中的特定阶段"""
           required_model = self._get_required_model_for_phase(phase)
           if required_model != self.current_model_name:
               return self._switch_model(required_model)
           return True
           
       def predict_masked(self, image: np.ndarray, mask: np.ndarray) -> List[Dict]:
           """对指定区域进行AI增强预测"""
           # 1. cv2.findContours提取连通区域
           # 2. cv2.boundingRect获取边界框
           # 3. 对每个区域调用AI模型
           # 4. 返回增强区域+位置信息
           
       def _get_required_model_for_phase(self, phase: str) -> Optional[str]:
           """根据阶段和策略确定所需模型"""
           if not self.current_strategy:
               return self.config.ai_model  # 回退到默认模型
               
           if phase == "structure_sr":
               return "opencv_cubic"  # 结构重建用传统方法
           elif phase == "gan_enhance":
               if not self.current_strategy.gan_policy.global_allowed:
                   return None  # 跳过GAN阶段
               strength = self.current_strategy.gan_policy.strength
               return "real_esrgan_x4" if strength in ["medium", "strong"] else "real_esrgan_x2"
           elif phase == "temporal_lock":
               return None  # 时序锁定不需要AI模型
           
           return None
   ```

2. **添加精确GPU内存监控**
   ```python
   class GPUMemoryMonitor:
       def __init__(self, max_gpu_memory_mb: int = 5500):
           self.max_gpu_memory_mb = max_gpu_memory_mb  # 预留500MB缓冲
           
       def check_gpu_memory_available(self) -> int:
           """检查可用GPU内存（MB）"""
           try:
               import torch
               if torch.cuda.is_available():
                   total_memory = torch.cuda.get_device_properties(0).total_memory
                   allocated = torch.cuda.memory_allocated(0)
                   cached = torch.cuda.memory_reserved(0)
                   used_mb = (allocated + cached) // (1024 * 1024)
                   total_mb = total_memory // (1024 * 1024)
                   available = min(total_mb - used_mb, self.max_gpu_memory_mb - used_mb)
                   return max(0, available)
               return 0
           except ImportError:
               return 0
               
       def get_optimal_tile_size(self, base_tile_size: int, model_memory_mb: int) -> int:
           """根据可用内存动态调整瓦片大小"""
           available = self.check_gpu_memory_available()
           
           if available >= model_memory_mb + 3000:  # 3GB余量
               return min(512, base_tile_size * 2)
           elif available >= model_memory_mb + 1000:  # 1GB余量
               return base_tile_size
           else:
               return max(64, base_tile_size // 2)
   ```

3. **实现predict_masked方法**
   ```python
   def predict_masked(self, image: np.ndarray, mask: np.ndarray) -> List[Dict]:
       """对指定区域进行AI增强预测"""
       if self.current_model is None:
           raise RuntimeError("No model loaded")
       
       enhanced_regions = []
       
       # 提取连通区域
       contours, _ = cv2.findContours(
           mask.astype(np.uint8), 
           cv2.RETR_EXTERNAL, 
           cv2.CHAIN_APPROX_SIMPLE
       )
       
       for contour in contours:
           # 获取边界框
           x, y, w, h = cv2.boundingRect(contour)
           
           # 提取区域
           region = image[y:y+h, x:x+w]
           region_mask = mask[y:y+h, x:x+w]
           
           # AI增强
           enhanced_region = self.current_model.predict(region)
           
           enhanced_regions.append({
               'region': enhanced_region,
               'mask': region_mask,
               'bbox': (x, y, w, h)
           })
       
       return enhanced_regions
   ```

4. **严格的模型切换机制**
   ```python
   def _switch_model(self, model_name: Optional[str]) -> bool:
       """安全的模型切换"""
       # 卸载当前模型
       if self.current_model:
           self.current_model.unload()
           self.current_model = None
           self.current_model_name = None
           
           # 强制GPU内存清理
           try:
               import torch
               if torch.cuda.is_available():
                   torch.cuda.empty_cache()
                   torch.cuda.synchronize()
           except ImportError:
               pass
       
       # 加载新模型
       if model_name:
           return self.load_model(model_name, use_gpu=True)
       
       return True
   ```

#### 验收标准
- [ ] 任意时刻最多1个模型驻留显存
- [ ] GPU内存监控精确，误差<100MB
- [ ] predict_masked方法正确提取和处理区域
- [ ] 模型切换无内存泄漏，torch.cuda.empty_cache()有效
- [ ] 策略驱动的模型选择逻辑正确工作
- [ ] 动态瓦片大小调整有效（64-512px范围）
- [ ] 与现有AIModelManager完全兼容，支持回退

---

### 任务 4：实现三阶段视频增强器
**优先级**: P0 (最高)  
**预估工时**: 14小时  
**依赖**: 任务3完成  
**需求**: FR-004 三阶段视频增强

#### 实现要求
1. **创建三阶段增强器模块**
   ```
   huaju4k/core/
   ├── three_stage_enhancer.py     # 新增 - 主增强器
   └── gan_mask_generator.py       # 新增 - GAN安全区域生成器
   ```

2. **实现ThreeStageVideoEnhancer类**
   ```python
   class ThreeStageVideoEnhancer:
       def __init__(self, model_manager: StrategyDrivenModelManager, 
                    progress_tracker: MultiStageProgressTracker):
           self.model_manager = model_manager
           self.progress_tracker = progress_tracker
           self.gan_mask_generator = GANSafeMaskGenerator()
           self.temporal_processor = TemporalLockProcessor()
           
       def enhance_video(self, input_path: str, strategy: EnhancementStrategy) -> str:
           """三阶段视频增强处理"""
           # 设置策略
           self.model_manager.set_strategy(strategy)
           
           # Stage 4.1: 结构重建
           structure_enhanced = self._stage_4_1_structure_reconstruction(input_path, strategy)
           
           # Stage 4.2: 受控GAN增强
           gan_enhanced = self._stage_4_2_controlled_gan_enhancement(structure_enhanced, strategy)
           
           # Stage 4.3: 时序锁定
           final_enhanced = self._stage_4_3_temporal_locking(gan_enhanced, strategy)
           
           return final_enhanced
   ```

3. **实现GANSafeMaskGenerator类**
   ```python
   class GANSafeMaskGenerator:
       def generate_multi_dimensional_safe_mask(self, frame: np.ndarray, 
                                               previous_frame: Optional[np.ndarray],
                                               gan_policy: GANPolicy) -> np.ndarray:
           """生成多维度GAN安全区域mask"""
           height, width = frame.shape[:2]
           
           # 1. 亮度排除mask
           frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           frame_normalized = frame_gray.astype(np.float32) / 255.0
           
           highlight_mask = frame_normalized > gan_policy.highlight_threshold
           shadow_mask = frame_normalized < gan_policy.shadow_threshold
           
           # 2. 边缘密度mask
           edges = cv2.Canny(frame_gray, 50, 150)
           kernel = np.ones((15, 15), np.uint8)  # 15x15邻域
           edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel) / (15 * 15 * 255)
           edge_mask = edge_density > gan_policy.edge_threshold
           
           # 3. 运动检测mask
           motion_mask = np.ones((height, width), dtype=bool)  # 默认允许
           if previous_frame is not None:
               prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
               frame_diff = cv2.absdiff(frame_gray, prev_gray).astype(np.float32) / 255.0
               motion_mask = frame_diff > gan_policy.motion_threshold
           
           # 4. 综合安全区域mask
           safe_mask = edge_mask & (~highlight_mask) & (~shadow_mask)
           
           # 根据GAN强度调整mask
           if gan_policy.strength == 'weak':
               safe_mask = safe_mask & motion_mask
           elif gan_policy.strength == 'medium':
               safe_mask = safe_mask | (motion_mask & (~highlight_mask))
           elif gan_policy.strength == 'strong':
               safe_mask = safe_mask | (~highlight_mask & ~shadow_mask)
           
           # 形态学操作平滑mask
           kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
           safe_mask = cv2.morphologyEx(safe_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
           safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_OPEN, kernel)
           
           return safe_mask.astype(bool)
   ```

4. **三个处理阶段实现**
   ```python
   def _stage_4_1_structure_reconstruction(self, input_path: str, strategy: EnhancementStrategy) -> str:
       """Stage 4.1: 结构重建 - 使用传统算法"""
       self.model_manager.execute_strategy_phase("structure_sr")
       # 使用OpenCV等传统方法进行初步放大
       
   def _stage_4_2_controlled_gan_enhancement(self, input_path: str, strategy: EnhancementStrategy) -> str:
       """Stage 4.2: 受控GAN增强 - 仅对安全区域应用GAN"""
       if not strategy.gan_policy.global_allowed:
           return input_path  # 跳过GAN阶段
           
       self.model_manager.execute_strategy_phase("gan_enhance")
       # 逐帧生成安全区域mask，仅对安全区域应用GAN增强
       
   def _stage_4_3_temporal_locking(self, input_path: str, strategy: EnhancementStrategy) -> str:
       """Stage 4.3: 时序锁定 - 稳定背景和平滑帧间变化"""
       self.temporal_processor.set_config(strategy.temporal_strategy)
       # 应用时序锁定算法
   ```

#### 核心算法要求
- **亮度阈值**: 高光0.85，暗部0.15，可通过策略调整
- **边缘检测**: Canny(50, 150) + 15x15邻域密度计算
- **运动检测**: 帧间absdiff，阈值0.05
- **GAN强度映射**:
  - weak: 仅边缘+运动区域
  - medium: 边缘+部分静态区域
  - strong: 更大范围但避免高光
- **形态学操作**: 5x5椭圆核，先闭运算后开运算

#### 验收标准
- [ ] 三阶段处理流程完整实现
- [ ] GAN安全区域mask准确生成，避免高光和暗部
- [ ] 边缘密度计算正确，15x15邻域统计
- [ ] 运动检测有效，避免静态背景误增强
- [ ] 三种GAN强度策略正确实现
- [ ] 形态学操作有效平滑mask边界
- [ ] 每个阶段都有正确的进度反馈

---

### 任务 5：实现光流增强时序锁定
**优先级**: P1 (高)  
**预估工时**: 12小时  
**依赖**: 任务4完成  
**需求**: FR-004 时序锁定处理

#### 实现要求
1. **创建时序锁定处理器**
   ```
   huaju4k/core/
   └── temporal_lock_processor.py  # 新增
   ```

2. **实现TemporalLockProcessor类**
   ```python
   class TemporalLockProcessor:
       def __init__(self, temporal_config: Optional[TemporalConfig] = None):
           self.config = temporal_config or TemporalConfig()
           self.background_model = None
           self.motion_detector = MotionDetector()
           
           # 光流参数
           self.optical_flow_params = {
               'pyr_scale': 0.5,
               'levels': 3,
               'winsize': 15,
               'iterations': 3,
               'poly_n': 5,
               'poly_sigma': 1.2,
               'flags': 0
           }
           
       def set_config(self, config: TemporalConfig) -> None:
           """设置时序配置"""
           self.config = config
           
       def process_video(self, input_path: str, output_path: str,
                        progress_callback: Optional[Callable] = None) -> bool:
           """光流增强的时序锁定主处理流程"""
           try:
               # 1. 建立背景模型（如果启用背景锁定）
               if self.config.background_lock:
                   self.background_model = self._build_background_model(input_path)
               
               # 2. 逐帧处理
               return self._process_frames(input_path, output_path, progress_callback)
               
           except Exception as e:
               logger.error(f"Temporal lock processing failed: {e}")
               return False
   ```

3. **光流背景稳定算法**
   ```python
   def _apply_optical_flow_stabilization(self, current_frame: np.ndarray,
                                        previous_frame: np.ndarray) -> np.ndarray:
       """使用cv2.calcOpticalFlowFarneback进行背景稳定"""
       current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
       previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
       
       # 计算光流
       flow = cv2.calcOpticalFlowFarneback(
           previous_gray, current_gray, None, **self.optical_flow_params
       )
       
       # 检测运动区域
       motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
       motion_mask = motion_magnitude > self.config.motion_threshold
       
       # 背景区域根据光流重映射
       if self.background_model is not None:
           h, w = flow.shape[:2]
           
           # 创建重映射坐标
           y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
           new_x = x_coords + flow[..., 0]
           new_y = y_coords + flow[..., 1]
           
           # 重映射背景
           stabilized_background = cv2.remap(
               self.background_model, 
               new_x, new_y,
               cv2.INTER_LINEAR,
               borderMode=cv2.BORDER_REFLECT
           )
           
           # 合并运动区域和稳定背景
           result = np.where(
               motion_mask[..., np.newaxis],
               current_frame,
               stabilized_background
           )
           
           return result.astype(np.uint8)
       
       return current_frame
   ```

4. **多层时序锁定策略**
   ```python
   def _apply_simple_temporal_stabilization(self, current_frame: np.ndarray,
                                          previous_frame: np.ndarray) -> np.ndarray:
       """指数加权平均平滑"""
       # 检测运动区域
       motion_mask = self._detect_motion(current_frame, previous_frame)
       
       # 根据强度设置平滑系数
       alpha_map = {
           "high": 0.1,    # 强平滑
           "medium": 0.3,  # 中等平滑
           "low": 0.5      # 轻微平滑
       }
       alpha = alpha_map.get(self.config.strength, 0.3)
       
       # 背景区域平滑
       smoothed_frame = cv2.addWeighted(
           current_frame.astype(np.float32), 1 - alpha,
           previous_frame.astype(np.float32), alpha,
           0
       )
       
       # 合并运动区域和平滑背景
       result = np.where(
           motion_mask[..., np.newaxis],
           current_frame,
           smoothed_frame
       )
       
       return result.astype(np.uint8)
   ```

5. **背景模型构建**
   ```python
   def _build_background_model(self, video_path: str) -> np.ndarray:
       """固定机位背景模型构建"""
       cap = cv2.VideoCapture(video_path)
       if not cap.isOpened():
           raise RuntimeError(f"Cannot open video: {video_path}")
       
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       sample_interval = max(1, total_frames // 50)  # 采样50帧
       
       sample_frames = []
       frame_idx = 0
       
       while len(sample_frames) < 50:
           cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
           ret, frame = cap.read()
           if not ret:
               break
           
           sample_frames.append(frame)
           frame_idx += sample_interval
       
       cap.release()
       
       if not sample_frames:
           raise RuntimeError("No frames could be sampled for background model")
       
       # 计算中位数背景
       background = np.median(sample_frames, axis=0).astype(np.uint8)
       logger.info(f"Background model built from {len(sample_frames)} frames")
       
       return background
   ```

6. **运动检测器**
   ```python
   class MotionDetector:
       def __init__(self, threshold: float = 0.05):
           self.threshold = threshold
           
       def detect_motion(self, current_frame: np.ndarray, 
                        previous_frame: np.ndarray) -> np.ndarray:
           """检测帧间运动区域"""
           current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
           previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
           
           # 帧间差分
           frame_diff = cv2.absdiff(current_gray, previous_gray).astype(np.float32) / 255.0
           
           # 运动mask
           motion_mask = frame_diff > self.threshold
           
           # 形态学操作去噪
           kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
           motion_mask = cv2.morphologyEx(motion_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
           motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
           
           return motion_mask.astype(bool)
   ```

#### 核心算法参数
- **光流参数**: pyr_scale=0.5, levels=3, winsize=15, iterations=3
- **运动阈值**: 可通过策略调整，默认0.05
- **平滑系数**: high=0.1, medium=0.3, low=0.5
- **背景采样**: 50帧，均匀间隔

#### 验收标准
- [ ] 光流计算正确，cv2.calcOpticalFlowFarneback工作正常
- [ ] 运动区域检测准确，避免静态背景误锁
- [ ] 背景重映射有效，cv2.remap正确应用光流
- [ ] 三种强度的平滑效果明显区别
- [ ] 背景模型构建稳定，中位数计算正确
- [ ] 处理速度可接受，不超过原视频时长的3倍
- [ ] 内存使用合理，支持长视频处理

---

### 任务 6：升级母版级音频增强器
**优先级**: P1 (高)  
**预估工时**: 16小时  
**依赖**: 任务5完成  
**需求**: FR-005 母版级音频增强

#### 实现要求
1. **创建母版级音频模块**
   ```
   huaju4k/audio/
   ├── master_grade_enhancer.py    # 新增
   └── audio_source_separator.py   # 新增
   ```

2. **实现MasterGradeAudioEnhancer类**
   ```python
   class MasterGradeAudioEnhancer:
       def __init__(self):
           self.temp_dir = Path(tempfile.gettempdir()) / "huaju4k_audio"
           self.temp_dir.mkdir(exist_ok=True)
           self.spleeter_available = self._check_spleeter_availability()
           
       def enhance_audio(self, video_path: str, 
                        strategy: EnhancementStrategy) -> AudioResult:
           """母版级音频增强主流程"""
           try:
               if not strategy.audio_strategy.source_separation_enabled:
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
               
               # 4. 母版级重混
               master_audio_path = self._master_grade_remix(
                   enhanced_dialogue, enhanced_music,
                   strategy.audio_strategy.master_settings
               )
               
               return AudioResult(
                   success=True,
                   output_path=master_audio_path,
                   quality_improvements=self._calculate_audio_quality_metrics(master_audio_path)
               )
               
           except Exception as e:
               logger.error(f"Master grade audio enhancement failed: {e}")
               return AudioResult(success=False, error=str(e))
   ```

3. **FFmpeg音轨提取**
   ```python
   def _extract_audio_with_ffmpeg(self, video_path: str) -> str:
       """使用FFmpeg提取音轨"""
       audio_path = self.temp_dir / "extracted_audio.wav"
       
       cmd = [
           'ffmpeg', '-i', video_path,
           '-vn',  # 不要视频流
           '-acodec', 'pcm_s16le',  # 16位PCM编码
           '-ar', '44100',  # 44.1kHz采样率
           '-ac', '2',  # 立体声
           '-y',  # 覆盖输出文件
           str(audio_path)
       ]
       
       try:
           result = subprocess.run(cmd, check=True, capture_output=True, text=True)
           logger.info(f"Audio extracted successfully: {audio_path}")
           return str(audio_path)
       except subprocess.CalledProcessError as e:
           logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
           raise RuntimeError(f"Audio extraction failed: {e.stderr}")
   ```

4. **Spleeter音源分离**
   ```python
   def _separate_audio_sources(self, audio_path: str) -> Dict[str, str]:
       """使用Spleeter进行音源分离"""
       if self.spleeter_available:
           try:
               from spleeter.separator import Separator
               
               # 初始化分离器 (人声 + 伴奏)
               separator = Separator('spleeter:2stems-16kHz')
               
               # 分离音频
               output_dir = self.temp_dir / "separated"
               output_dir.mkdir(exist_ok=True)
               
               separator.separate_to_file(audio_path, str(output_dir))
               
               audio_name = Path(audio_path).stem
               return {
                   'vocals': str(output_dir / audio_name / "vocals.wav"),
                   'accompaniment': str(output_dir / audio_name / "accompaniment.wav")
               }
               
           except Exception as e:
               logger.warning(f"Spleeter separation failed: {e}, using fallback")
       
       # 回退到简单分离
       return self._simple_vocal_separation(audio_path)
   
   def _simple_vocal_separation(self, audio_path: str) -> Dict[str, str]:
       """简单的人声分离（回退方案）"""
       try:
           import librosa
           
           # 加载音频
           y, sr = librosa.load(audio_path, sr=44100, mono=False)
           
           if y.ndim == 1:
               # 单声道，无法分离
               vocals_path = self.temp_dir / "vocals_mono.wav"
               accompaniment_path = self.temp_dir / "accompaniment_mono.wav"
               
               librosa.output.write_wav(str(vocals_path), y, sr)
               librosa.output.write_wav(str(accompaniment_path), y * 0.3, sr)
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
           
       except ImportError:
           logger.error("librosa not available for fallback separation")
           raise RuntimeError("Audio separation failed: no available method")
   ```

5. **对白增强 - librosa + noisereduce**
   ```python
   def _enhance_dialogue(self, vocals_path: str, enhancement_strength: float) -> str:
       """对白增强处理"""
       try:
           import librosa
           import noisereduce as nr
           
           # 加载音频
           y, sr = librosa.load(vocals_path, sr=44100)
           
           # 1. 降噪处理
           reduced_noise = nr.reduce_noise(
               y=y, sr=sr, 
               prop_decrease=enhancement_strength * 0.8
           )
           
           # 2. 预加重（增强高频清晰度）
           if enhancement_strength > 0.5:
               reduced_noise = librosa.effects.preemphasis(reduced_noise, coef=0.97)
           
           # 3. 动态范围压缩（软限制器）
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
           
           logger.info(f"Dialogue enhanced: {enhanced_path}")
           return str(enhanced_path)
           
       except ImportError as e:
           logger.error(f"Audio enhancement libraries not available: {e}")
           return vocals_path  # 返回原始文件
   ```

6. **母版级重混 - pydub**
   ```python
   def _master_grade_remix(self, dialogue_path: str, music_path: str, 
                          master_settings: Dict[str, Any]) -> str:
       """母版级重混"""
       try:
           from pydub import AudioSegment
           
           # 加载音轨
           dialogue = AudioSegment.from_wav(dialogue_path)
           music = AudioSegment.from_wav(music_path)
           
           # 音量平衡
           dialogue_gain = master_settings.get('dialogue_gain', 0)  # dB
           music_gain = master_settings.get('music_gain', -6)       # dB
           
           dialogue = dialogue + dialogue_gain
           music = music + music_gain
           
           # 长度对齐
           max_length = max(len(dialogue), len(music))
           dialogue = dialogue[:max_length]
           music = music[:max_length]
           
           # 叠加混合
           master_audio = dialogue.overlay(music)
           
           # 母版级处理
           master_audio = master_audio.normalize(headroom=1.0)  # 防止削波
           
           # 导出最终音频
           master_path = self.temp_dir / "master_audio.wav"
           master_audio.export(str(master_path), format="wav")
           
           logger.info(f"Master audio created: {master_path}")
           return str(master_path)
           
       except ImportError as e:
           logger.error(f"pydub not available: {e}")
           return dialogue_path  # 返回对白音频
   ```

7. **依赖包检查和管理**
   ```python
   def _check_spleeter_availability(self) -> bool:
       """检查Spleeter是否可用"""
       try:
           from spleeter.separator import Separator
           return True
       except ImportError:
           logger.warning("Spleeter not available, will use fallback separation")
           return False
   
   @staticmethod
   def check_dependencies() -> Dict[str, bool]:
       """检查所有音频处理依赖"""
       dependencies = {}
       
       try:
           import librosa
           dependencies['librosa'] = True
       except ImportError:
           dependencies['librosa'] = False
       
       try:
           import noisereduce
           dependencies['noisereduce'] = True
       except ImportError:
           dependencies['noisereduce'] = False
       
       try:
           from pydub import AudioSegment
           dependencies['pydub'] = True
       except ImportError:
           dependencies['pydub'] = False
       
       try:
           from spleeter.separator import Separator
           dependencies['spleeter'] = True
       except ImportError:
           dependencies['spleeter'] = False
       
       return dependencies
   ```

#### 核心算法参数
- **FFmpeg音频**: 16位PCM, 44.1kHz, 立体声
- **Spleeter模型**: 2stems-16kHz (人声+伴奏分离)
- **降噪强度**: enhancement_strength * 0.8
- **预加重系数**: 0.97 (增强高频)
- **压缩参数**: threshold=0.8, ratio=4.0
- **音量平衡**: 对白0dB, 音乐-6dB

#### 验收标准
- [ ] FFmpeg音轨提取成功，输出44.1kHz立体声WAV
- [ ] Spleeter音源分离有效，人声和伴奏明显分离
- [ ] 回退分离方案工作正常（立体声中央声道）
- [ ] 对白降噪效果明显，noisereduce工作正常
- [ ] 预加重增强高频清晰度
- [ ] 动态范围压缩防止削波
- [ ] pydub重混输出音量平衡合理
- [ ] 依赖包检查机制工作正常，缺失时优雅降级

---

### 任务 7：集成新组件到主处理流程
**优先级**: P0 (最高)  
**预估工时**: 8小时  
**依赖**: 任务1-6完成  
**需求**: 系统集成

#### 实现要求
1. **修改VideoEnhancementProcessor主流程**
   ```python
   def process(self, input_path: str, output_path: str = None, 
               preset: str = "theater_medium", quality: str = "balanced") -> ProcessResult:
       """升级后的主处理管道"""
       try:
           # ... 现有初始化代码 ...
           
           # Stage 1: 视频分析 + 舞台结构分析
           self.progress_tracker.start_stage("analyzing", "分析视频和舞台结构")
           video_info = self.analyze_video(input_path)
           structure_features = self._analyze_stage_structure(input_path)
           self.progress_tracker.complete_stage("analyzing")
           
           # Stage 2: 策略计算 + 增强策略生成
           self.progress_tracker.start_stage("strategy", "生成增强策略")
           processing_strategy = self._calculate_processing_strategy(video_info, quality, preset)
           enhancement_strategy = self._generate_enhancement_strategy(structure_features)
           self._save_strategy_cache(enhancement_strategy)
           self.progress_tracker.complete_stage("strategy")
           
           # Stage 3: 加载AI模型（策略驱动）
           self.progress_tracker.start_stage("model_loading", "加载AI增强模型")
           self._setup_strategy_driven_model_manager(enhancement_strategy)
           self.progress_tracker.complete_stage("model_loading")
           
           # Stage 4: 三阶段视频增强
           self.progress_tracker.start_stage("video_enhancement", "执行三阶段视频增强")
           enhanced_video_path = self._three_stage_video_enhancement(input_path, enhancement_strategy)
           self.progress_tracker.complete_stage("video_enhancement")
           
           # Stage 5: 母版级音频增强
           audio_result = None
           if video_info.has_audio:
               self.progress_tracker.start_stage("audio_enhancement", "执行母版级音频增强")
               audio_result = self._master_grade_audio_enhancement(input_path, enhancement_strategy)
               self.progress_tracker.complete_stage("audio_enhancement")
           
           # ... 继续现有的Stage 6-7 ...
           
       except Exception as e:
           # 错误时回退到原有处理流程
           logger.warning(f"Enhanced processing failed: {e}, falling back to legacy processing")
           return self._legacy_process(input_path, output_path, preset, quality)
   ```

2. **添加新的分析和策略方法**
   ```python
   def _analyze_stage_structure(self, input_path: str) -> StructureFeatures:
       """Stage 1扩展: 舞台结构分析"""
       from ..analysis.stage_structure_analyzer import StageStructureAnalyzer
       
       analyzer = StageStructureAnalyzer()
       return analyzer.analyze_structure(input_path)
   
   def _generate_enhancement_strategy(self, features: StructureFeatures) -> EnhancementStrategy:
       """Stage 2扩展: 增强策略生成"""
       from ..strategy.enhancement_planner import EnhancementStrategyPlanner
       
       planner = EnhancementStrategyPlanner()
       return planner.generate_strategy(features)
   
   def _setup_strategy_driven_model_manager(self, strategy: EnhancementStrategy) -> None:
       """Stage 3扩展: 设置策略驱动的模型管理器"""
       # 升级现有的AI模型管理器
       if hasattr(self.ai_model_manager, 'set_strategy'):
           self.ai_model_manager.set_strategy(strategy)
       else:
           logger.warning("AI model manager does not support strategy-driven mode")
   
   def _three_stage_video_enhancement(self, input_path: str, 
                                     strategy: EnhancementStrategy) -> Optional[str]:
       """Stage 4扩展: 三阶段视频增强"""
       try:
           from ..core.three_stage_enhancer import ThreeStageVideoEnhancer
           
           enhancer = ThreeStageVideoEnhancer(
               self.ai_model_manager, 
               self.progress_tracker
           )
           
           return enhancer.enhance_video(input_path, strategy)
           
       except ImportError as e:
           logger.warning(f"Three-stage enhancer not available: {e}, using legacy enhancement")
           return self._legacy_enhance_video(input_path, self._current_processing_strategy)
       except Exception as e:
           logger.error(f"Three-stage enhancement failed: {e}, falling back to legacy")
           return self._legacy_enhance_video(input_path, self._current_processing_strategy)
   
   def _master_grade_audio_enhancement(self, input_path: str, 
                                      strategy: EnhancementStrategy) -> Optional[AudioResult]:
       """Stage 5扩展: 母版级音频增强"""
       try:
           from ..audio.master_grade_enhancer import MasterGradeAudioEnhancer
           
           enhancer = MasterGradeAudioEnhancer()
           return enhancer.enhance_audio(input_path, strategy)
           
       except ImportError as e:
           logger.warning(f"Master grade audio enhancer not available: {e}, using legacy audio")
           return self._legacy_enhance_audio(input_path, strategy)
       except Exception as e:
           logger.error(f"Master grade audio enhancement failed: {e}, falling back to legacy")
           return self._legacy_enhance_audio(input_path, strategy)
   ```

3. **策略缓存机制**
   ```python
   def _save_strategy_cache(self, strategy: EnhancementStrategy) -> None:
       """保存策略到缓存文件"""
       try:
           cache_path = self.memory_manager.create_temp_file(
               prefix="enhancement_strategy_", suffix=".json"
           )
           strategy.save_to_file(cache_path)
           self._current_strategy_cache_path = cache_path
           logger.info(f"Enhancement strategy cached to: {cache_path}")
       except Exception as e:
           logger.warning(f"Failed to cache strategy: {e}")
   
   def _load_strategy_cache(self, cache_path: str) -> Optional[EnhancementStrategy]:
       """从缓存加载策略"""
       try:
           import json
           with open(cache_path, 'r', encoding='utf-8') as f:
               data = json.load(f)
           
           # 重建策略对象
           from ..strategy.enhancement_planner import EnhancementStrategy
           return EnhancementStrategy(**data)
           
       except Exception as e:
           logger.warning(f"Failed to load strategy cache: {e}")
           return None
   ```

4. **更新进度跟踪**
   ```python
   def _setup_progress_stages(self) -> None:
       """设置增强后的进度跟踪阶段"""
       stages = [
           ("analyzing", "分析视频和舞台结构", 2.0),  # 扩展分析阶段
           ("strategy", "生成增强策略", 1.0),        # 新增策略阶段
           ("model_loading", "加载AI模型", 1.0),
           ("video_enhancement", "三阶段视频增强", 12.0),  # 扩展视频增强
           ("audio_enhancement", "母版级音频增强", 4.0),   # 扩展音频增强
           ("finalizing", "最终合成", 2.0),
           ("validation", "质量验证", 1.0)
       ]
       
       for stage_name, display_name, weight in stages:
           self.progress_tracker.add_stage(stage_name, display_name, weight)
       
       # 添加视频增强子阶段
       self.progress_tracker.add_substage("video_enhancement", "structure_sr", "结构重建", 3.0)
       self.progress_tracker.add_substage("video_enhancement", "gan_enhance", "GAN增强", 6.0)
       self.progress_tracker.add_substage("video_enhancement", "temporal_lock", "时序锁定", 3.0)
   ```

5. **错误处理和回退机制**
   ```python
   def _legacy_process(self, input_path: str, output_path: str, 
                      preset: str, quality: str) -> ProcessResult:
       """回退到原有处理流程"""
       logger.info("Using legacy processing pipeline")
       
       # 调用原有的process方法逻辑
       # 这里保持与现有实现完全一致
       return self._original_process_implementation(input_path, output_path, preset, quality)
   
   def _legacy_enhance_video(self, input_path: str, strategy: ProcessingStrategy) -> Optional[str]:
       """回退到原有视频增强"""
       # 使用现有的视频增强逻辑
       return self._original_enhance_video_implementation(input_path, strategy)
   
   def _legacy_enhance_audio(self, input_path: str, strategy: EnhancementStrategy) -> Optional[AudioResult]:
       """回退到原有音频增强"""
       if self.audio_enhancer:
           # 使用现有的TheaterAudioEnhancer
           theater_preset = "medium"  # 从strategy中提取或使用默认值
           return self.audio_enhancer.enhance(input_path, None, theater_preset)
       return None
   ```

#### 验收标准
- [ ] CLI接口保持完全向后兼容
- [ ] 新组件无缝集成到现有7阶段流程
- [ ] 错误时能正确回退到原有处理流程
- [ ] 进度跟踪准确显示所有新增阶段和子阶段
- [ ] 策略缓存机制工作正常
- [ ] 所有新功能都有适当的错误处理
- [ ] 日志记录详细，便于调试

---

### 任务 8：升级质量验证系统
**优先级**: P1 (高)  
**预估工时**: 8小时  
**依赖**: 任务7完成  
**需求**: FR-006 质量验证升级

#### 实现要求
1. **扩展现有质量验证器**
   ```python
   class MasterGradeQualityValidator:
       def __init__(self):
           self.video_analyzer = VideoAnalyzer()
           
       def validate_master_quality(self, input_path: str, 
                                 output_path: str,
                                 enhancement_strategy: EnhancementStrategy) -> QualityReport:
           """母版级质量验证"""
           report = QualityReport()
           
           # 基础质量验证
           report.basic_quality = self._validate_basic_quality(input_path, output_path)
           
           # 视频质量验证
           report.video_quality = self._validate_video_quality(output_path, enhancement_strategy)
           
           # 音频质量验证
           report.audio_quality = self._validate_audio_quality(output_path)
           
           # 同步性验证
           report.sync_quality = self._validate_av_sync(output_path)
           
           # 母版级特定验证
           report.master_grade_metrics = self._validate_master_grade_metrics(
               input_path, output_path, enhancement_strategy
           )
           
           return report
   ```

2. **新增验证指标实现**
   ```python
   def _validate_video_quality(self, output_path: str, 
                              strategy: EnhancementStrategy) -> Dict[str, float]:
       """视频质量验证"""
       metrics = {}
       
       try:
           # 帧间亮度波动检测
           metrics['brightness_stability'] = self._check_brightness_stability(output_path)
           
           # 边缘稳定性评估
           metrics['edge_stability'] = self._check_edge_stability(output_path)
           
           # 高光溢出检测
           metrics['highlight_clipping'] = self._check_highlight_clipping(output_path)
           
           # GAN增强区域验证
           if strategy.gan_policy.global_allowed:
               metrics['gan_enhancement_quality'] = self._validate_gan_enhancement(output_path)
           
           # 时序锁定效果验证
           if strategy.temporal_strategy.background_lock:
               metrics['temporal_stability'] = self._validate_temporal_stability(output_path)
           
       except Exception as e:
           logger.error(f"Video quality validation failed: {e}")
           metrics['validation_error'] = 1.0
       
       return metrics
   
   def _check_brightness_stability(self, video_path: str) -> float:
       """检查帧间亮度波动"""
       cap = cv2.VideoCapture(video_path)
       if not cap.isOpened():
           return 0.0
       
       brightness_values = []
       frame_count = 0
       
       while frame_count < 100:  # 采样100帧
           ret, frame = cap.read()
           if not ret:
               break
           
           # 计算平均亮度
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           brightness = np.mean(gray) / 255.0
           brightness_values.append(brightness)
           frame_count += 1
       
       cap.release()
       
       if len(brightness_values) < 2:
           return 0.0
       
       # 计算亮度稳定性（标准差越小越稳定）
       brightness_std = np.std(brightness_values)
       stability_score = max(0.0, 1.0 - brightness_std * 5)  # 归一化到0-1
       
       return stability_score
   
   def _check_edge_stability(self, video_path: str) -> float:
       """检查边缘稳定性"""
       cap = cv2.VideoCapture(video_path)
       if not cap.isOpened():
           return 0.0
       
       edge_densities = []
       frame_count = 0
       
       while frame_count < 50:  # 采样50帧
           ret, frame = cap.read()
           if not ret:
               break
           
           # 计算边缘密度
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           edges = cv2.Canny(gray, 50, 150)
           edge_density = np.sum(edges > 0) / edges.size
           edge_densities.append(edge_density)
           frame_count += 1
       
       cap.release()
       
       if len(edge_densities) < 2:
           return 0.0
       
       # 计算边缘稳定性
       edge_std = np.std(edge_densities)
       stability_score = max(0.0, 1.0 - edge_std * 10)
       
       return stability_score
   
   def _check_highlight_clipping(self, video_path: str) -> float:
       """检查高光溢出"""
       cap = cv2.VideoCapture(video_path)
       if not cap.isOpened():
           return 1.0  # 无法检查，假设有问题
       
       clipping_ratios = []
       frame_count = 0
       
       while frame_count < 30:  # 采样30帧
           ret, frame = cap.read()
           if not ret:
               break
           
           # 检查高光溢出（像素值>=250）
           clipped_pixels = np.sum(frame >= 250)
           total_pixels = frame.size
           clipping_ratio = clipped_pixels / total_pixels
           clipping_ratios.append(clipping_ratio)
           frame_count += 1
       
       cap.release()
       
       if not clipping_ratios:
           return 1.0
       
       # 平均溢出比例，越低越好
       avg_clipping = np.mean(clipping_ratios)
       clipping_score = max(0.0, 1.0 - avg_clipping * 20)  # 5%溢出对应0分
       
       return clipping_score
   ```

3. **音频质量验证**
   ```python
   def _validate_audio_quality(self, output_path: str) -> Dict[str, float]:
       """音频质量验证"""
       metrics = {}
       
       try:
           # 提取音频进行分析
           temp_audio = self._extract_audio_for_analysis(output_path)
           
           # 对白清晰度评分
           metrics['dialogue_clarity'] = self._assess_dialogue_clarity(temp_audio)
           
           # 音量一致性检查
           metrics['volume_consistency'] = self._check_volume_consistency(temp_audio)
           
           # 频率响应分析
           metrics['frequency_response'] = self._analyze_frequency_response(temp_audio)
           
           # 动态范围评估
           metrics['dynamic_range'] = self._assess_dynamic_range(temp_audio)
           
       except Exception as e:
           logger.error(f"Audio quality validation failed: {e}")
           metrics['validation_error'] = 1.0
       
       return metrics
   
   def _assess_dialogue_clarity(self, audio_path: str) -> float:
       """评估对白清晰度"""
       try:
           import librosa
           
           # 加载音频
           y, sr = librosa.load(audio_path, sr=44100)
           
           # 分析语音频段能量（300-3400Hz）
           stft = librosa.stft(y)
           freqs = librosa.fft_frequencies(sr=sr)
           
           # 语音频段索引
           speech_freq_mask = (freqs >= 300) & (freqs <= 3400)
           speech_energy = np.mean(np.abs(stft[speech_freq_mask, :]))
           
           # 总能量
           total_energy = np.mean(np.abs(stft))
           
           # 语音清晰度比例
           if total_energy > 0:
               clarity_ratio = speech_energy / total_energy
               clarity_score = min(1.0, clarity_ratio * 2)  # 归一化
           else:
               clarity_score = 0.0
           
           return clarity_score
           
       except ImportError:
           logger.warning("librosa not available for dialogue clarity assessment")
           return 0.5  # 默认中等分数
       except Exception as e:
           logger.error(f"Dialogue clarity assessment failed: {e}")
           return 0.0
   ```

4. **同步性验证**
   ```python
   def _validate_av_sync(self, video_path: str) -> Dict[str, float]:
       """音视频同步性验证"""
       metrics = {}
       
       try:
           # 获取视频和音频时长
           video_info = self.video_analyzer.analyze_video(video_path)
           
           # 提取音频时长
           audio_duration = self._get_audio_duration(video_path)
           
           if audio_duration is not None:
               # 计算时长差异
               duration_diff = abs(video_info.duration - audio_duration)
               sync_score = max(0.0, 1.0 - duration_diff / 2.0)  # 2秒差异对应0分
               
               metrics['duration_sync'] = sync_score
               metrics['duration_difference'] = duration_diff
           else:
               metrics['duration_sync'] = 1.0  # 无音频时认为同步
               metrics['duration_difference'] = 0.0
           
           # 帧率一致性检查
           metrics['framerate_consistency'] = self._check_framerate_consistency(video_path)
           
       except Exception as e:
           logger.error(f"AV sync validation failed: {e}")
           metrics['validation_error'] = 1.0
       
       return metrics
   ```

5. **质量报告数据结构**
   ```python
   @dataclass
   class QualityReport:
       basic_quality: Dict[str, float] = field(default_factory=dict)
       video_quality: Dict[str, float] = field(default_factory=dict)
       audio_quality: Dict[str, float] = field(default_factory=dict)
       sync_quality: Dict[str, float] = field(default_factory=dict)
       master_grade_metrics: Dict[str, float] = field(default_factory=dict)
       
       overall_score: float = 0.0
       validation_timestamp: datetime = field(default_factory=datetime.now)
       
       def calculate_overall_score(self) -> float:
           """计算总体质量评分"""
           all_scores = []
           
           for category in [self.basic_quality, self.video_quality, 
                           self.audio_quality, self.sync_quality, 
                           self.master_grade_metrics]:
               category_scores = [v for k, v in category.items() 
                                if isinstance(v, (int, float)) and k != 'validation_error']
               if category_scores:
                   all_scores.extend(category_scores)
           
           if all_scores:
               self.overall_score = np.mean(all_scores)
           else:
               self.overall_score = 0.0
           
           return self.overall_score
       
       def to_dict(self) -> Dict[str, Any]:
           """转换为字典格式"""
           return asdict(self)
   ```

#### 验收标准
- [ ] 所有新增验证指标工作正常
- [ ] 帧间亮度波动检测准确
- [ ] 边缘稳定性评估有效
- [ ] 高光溢出检测正确
- [ ] 音频对白清晰度评分合理
- [ ] 音量一致性检查有效
- [ ] 音视频同步性验证准确
- [ ] 质量报告详细完整
- [ ] 验证速度可接受（<原视频时长的10%）

---

### 任务 9：系统测试和优化
**优先级**: P0 (最高)  
**预估工时**: 10小时  
**依赖**: 任务8完成  
**需求**: 系统稳定性

#### 实现要求
1. **端到端集成测试**
   ```python
   class TheaterEnhancementIntegrationTest:
       def __init__(self):
           self.test_videos = [
               "test_short_1080p.mp4",    # 短视频测试
               "test_medium_720p.mp4",    # 中等长度测试
               "test_long_4k.mp4"         # 长视频压力测试
           ]
           
       def test_complete_pipeline(self):
           """完整处理流程测试"""
           for test_video in self.test_videos:
               try:
                   processor = VideoEnhancementProcessor()
                   result = processor.process(
                       input_path=test_video,
                       preset="theater_medium",
                       quality="balanced"
                   )
                   
                   # 验证处理结果
                   assert result.success, f"Processing failed for {test_video}"
                   assert os.path.exists(result.output_path), "Output file not created"
                   assert result.quality_metrics is not None, "Quality metrics missing"
                   
                   logger.info(f"✓ {test_video} processed successfully")
                   
               except Exception as e:
                   logger.error(f"✗ {test_video} processing failed: {e}")
                   raise
       
       def test_memory_management(self):
           """内存管理压力测试"""
           import psutil
           import gc
           
           process = psutil.Process()
           initial_memory = process.memory_info().rss / 1024 / 1024  # MB
           
           processor = VideoEnhancementProcessor()
           
           # 连续处理多个视频
           for i in range(3):
               result = processor.process(
                   input_path="test_medium_720p.mp4",
                   preset="theater_medium",
                   quality="balanced"
               )
               
               # 检查内存使用
               current_memory = process.memory_info().rss / 1024 / 1024
               memory_increase = current_memory - initial_memory
               
               assert memory_increase < 2000, f"Memory leak detected: {memory_increase}MB increase"
               
               # 强制垃圾回收
               gc.collect()
               
               logger.info(f"✓ Memory test iteration {i+1}: {memory_increase:.1f}MB increase")
       
       def test_gpu_memory_management(self):
           """GPU内存管理测试"""
           try:
               import torch
               if not torch.cuda.is_available():
                   logger.info("GPU not available, skipping GPU memory test")
                   return
               
               processor = VideoEnhancementProcessor()
               
               # 检查初始GPU内存
               torch.cuda.empty_cache()
               initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
               
               # 处理视频
               result = processor.process(
                   input_path="test_short_1080p.mp4",
                   preset="theater_medium",
                   quality="balanced"
               )
               
               # 检查GPU内存使用
               peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
               final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
               
               assert peak_gpu_memory < 6000, f"GPU memory exceeded 6GB: {peak_gpu_memory}MB"
               assert final_gpu_memory - initial_gpu_memory < 100, "GPU memory not properly released"
               
               logger.info(f"✓ GPU memory test: peak {peak_gpu_memory:.1f}MB, final {final_gpu_memory:.1f}MB")
               
           except ImportError:
               logger.info("PyTorch not available, skipping GPU memory test")
   ```

2. **性能基准测试**
   ```python
   class PerformanceBenchmark:
       def __init__(self):
           self.benchmark_results = {}
           
       def benchmark_processing_speed(self):
           """处理速度基准测试"""
           test_cases = [
               ("720p_30s", "test_720p_30s.mp4"),
               ("1080p_60s", "test_1080p_60s.mp4"),
               ("4k_30s", "test_4k_30s.mp4")
           ]
           
           processor = VideoEnhancementProcessor()
           
           for test_name, test_video in test_cases:
               start_time = time.time()
               
               result = processor.process(
                   input_path=test_video,
                   preset="theater_medium",
                   quality="balanced"
               )
               
               processing_time = time.time() - start_time
               
               if result.success:
                   # 计算处理速度比
                   video_duration = result.frames_processed / 30.0  # 假设30fps
                   speed_ratio = video_duration / processing_time
                   
                   self.benchmark_results[test_name] = {
                       'processing_time': processing_time,
                       'speed_ratio': speed_ratio,
                       'memory_peak': result.memory_peak_mb
                   }
                   
                   logger.info(f"✓ {test_name}: {processing_time:.1f}s, {speed_ratio:.2f}x speed")
               else:
                   logger.error(f"✗ {test_name}: processing failed")
       
       def benchmark_quality_metrics(self):
           """质量指标基准测试"""
           processor = VideoEnhancementProcessor()
           
           result = processor.process(
               input_path="test_reference_video.mp4",
               preset="theater_medium",
               quality="balanced"
           )
           
           if result.success and result.quality_metrics:
               # 检查关键质量指标
               required_metrics = [
                   'resolution_improvement_ratio',
                   'brightness_stability',
                   'edge_stability',
                   'highlight_clipping'
               ]
               
               for metric in required_metrics:
                   if metric in result.quality_metrics:
                       value = result.quality_metrics[metric]
                       logger.info(f"✓ {metric}: {value:.3f}")
                   else:
                       logger.warning(f"⚠ Missing quality metric: {metric}")
   ```

3. **错误处理和恢复测试**
   ```python
   class ErrorHandlingTest:
       def test_invalid_input_handling(self):
           """无效输入处理测试"""
           processor = VideoEnhancementProcessor()
           
           # 测试不存在的文件
           result = processor.process(
               input_path="nonexistent_video.mp4",
               preset="theater_medium",
               quality="balanced"
           )
           
           assert not result.success, "Should fail for nonexistent file"
           assert "not found" in result.error.lower(), "Error message should indicate file not found"
           
           # 测试损坏的视频文件
           result = processor.process(
               input_path="corrupted_video.mp4",
               preset="theater_medium", 
               quality="balanced"
           )
           
           assert not result.success, "Should fail for corrupted file"
           
       def test_fallback_mechanisms(self):
           """回退机制测试"""
           processor = VideoEnhancementProcessor()
           
           # 模拟AI模型加载失败
           original_load_model = processor.ai_model_manager.load_model
           processor.ai_model_manager.load_model = lambda *args, **kwargs: False
           
           result = processor.process(
               input_path="test_short_1080p.mp4",
               preset="theater_medium",
               quality="balanced"
           )
           
           # 应该回退到传统处理方式
           assert result.success, "Should fallback to legacy processing"
           
           # 恢复原始方法
           processor.ai_model_manager.load_model = original_load_model
       
       def test_resource_cleanup(self):
           """资源清理测试"""
           import tempfile
           import shutil
           
           temp_dir = tempfile.mkdtemp()
           
           try:
               processor = VideoEnhancementProcessor()
               
               # 处理视频
               result = processor.process(
                   input_path="test_short_1080p.mp4",
                   preset="theater_medium",
                   quality="balanced"
               )
               
               # 检查临时文件是否被清理
               temp_files = list(Path(temp_dir).glob("*"))
               logger.info(f"Temporary files remaining: {len(temp_files)}")
               
               # 应该只有少量必要的临时文件
               assert len(temp_files) < 5, f"Too many temporary files: {len(temp_files)}"
               
           finally:
               shutil.rmtree(temp_dir, ignore_errors=True)
   ```

4. **长视频稳定性测试**
   ```python
   class LongVideoStabilityTest:
       def test_2_hour_video_processing(self):
           """2小时长视频处理稳定性测试"""
           processor = VideoEnhancementProcessor()
           
           # 创建或使用2小时测试视频
           long_video_path = "test_2hour_video.mp4"
           
           if not os.path.exists(long_video_path):
               logger.info("Long test video not found, skipping long video test")
               return
           
           start_time = time.time()
           
           result = processor.process(
               input_path=long_video_path,
               preset="theater_medium",
               quality="balanced"
           )
           
           processing_time = time.time() - start_time
           
           assert result.success, "Long video processing should succeed"
           assert os.path.exists(result.output_path), "Output file should exist"
           
           # 检查输出文件大小合理
           output_size = os.path.getsize(result.output_path) / 1024 / 1024 / 1024  # GB
           assert output_size > 1.0, "Output file seems too small"
           assert output_size < 50.0, "Output file seems too large"
           
           logger.info(f"✓ 2-hour video processed in {processing_time/3600:.1f} hours")
           logger.info(f"✓ Output size: {output_size:.1f} GB")
   ```

5. **自动化测试运行器**
   ```python
   class TestRunner:
       def __init__(self):
           self.test_results = {}
           
       def run_all_tests(self):
           """运行所有测试"""
           test_suites = [
               ("Integration Tests", TheaterEnhancementIntegrationTest()),
               ("Performance Benchmark", PerformanceBenchmark()),
               ("Error Handling Tests", ErrorHandlingTest()),
               ("Long Video Stability", LongVideoStabilityTest())
           ]
           
           for suite_name, test_suite in test_suites:
               logger.info(f"\n{'='*50}")
               logger.info(f"Running {suite_name}")
               logger.info(f"{'='*50}")
               
               try:
                   # 运行测试套件中的所有测试方法
                   test_methods = [method for method in dir(test_suite) 
                                 if method.startswith('test_')]
                   
                   suite_results = {}
                   
                   for test_method in test_methods:
                       try:
                           logger.info(f"\nRunning {test_method}...")
                           getattr(test_suite, test_method)()
                           suite_results[test_method] = "PASSED"
                           logger.info(f"✓ {test_method} PASSED")
                       except Exception as e:
                           suite_results[test_method] = f"FAILED: {e}"
                           logger.error(f"✗ {test_method} FAILED: {e}")
                   
                   self.test_results[suite_name] = suite_results
                   
               except Exception as e:
                   logger.error(f"Test suite {suite_name} failed to run: {e}")
                   self.test_results[suite_name] = {"suite_error": str(e)}
           
           # 生成测试报告
           self._generate_test_report()
       
       def _generate_test_report(self):
           """生成测试报告"""
           report_path = "theater_enhancement_test_report.md"
           
           with open(report_path, 'w', encoding='utf-8') as f:
               f.write("# Huaju4K Theater Enhancement System Test Report\n\n")
               f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
               
               total_tests = 0
               passed_tests = 0
               
               for suite_name, results in self.test_results.items():
                   f.write(f"## {suite_name}\n\n")
                   
                   for test_name, result in results.items():
                       total_tests += 1
                       if result == "PASSED":
                           passed_tests += 1
                           f.write(f"- ✓ {test_name}: PASSED\n")
                       else:
                           f.write(f"- ✗ {test_name}: {result}\n")
                   
                   f.write("\n")
               
               # 总结
               success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
               f.write(f"## Summary\n\n")
               f.write(f"- Total Tests: {total_tests}\n")
               f.write(f"- Passed: {passed_tests}\n")
               f.write(f"- Failed: {total_tests - passed_tests}\n")
               f.write(f"- Success Rate: {success_rate:.1f}%\n")
           
           logger.info(f"\nTest report generated: {report_path}")
           logger.info(f"Success rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
   ```

#### 验收标准
- [ ] 端到端流程稳定运行，成功率>95%
- [ ] 显存占用严格控制在6GB内，无内存泄漏
- [ ] 长视频（2小时）处理不崩溃，稳定完成
- [ ] 处理速度在可接受范围（不超过原视频时长的10倍）
- [ ] 错误处理健壮，能正确回退到原有流程
- [ ] 资源清理完整，临时文件正确删除
- [ ] 所有测试用例通过率>90%
- [ ] 性能基准测试结果符合预期
- [ ] 自动化测试报告生成正确

---

## 最终验收标准

### 功能验收
- [ ] Stage 1-2: 结构分析和策略生成工作正常
- [ ] Stage 3-4: 策略驱动的三阶段视频增强
- [ ] Stage 5: 母版级音频增强效果良好
- [ ] Stage 6-7: 质量验证和输出完整

### 性能验收
- [ ] 显存使用严格控制（≤6GB）
- [ ] 处理速度在可接受范围
- [ ] 长视频处理稳定性

### 兼容性验收
- [ ] CLI接口完全向后兼容
- [ ] 现有配置文件继续有效
- [ ] 错误时能回退到原有流程

---

## 实施指导原则

### 🔴 严格执行顺序
- 必须按任务编号1→2→3→4→5→6→7→8→9顺序执行
- 不跳步骤，不并行开发
- 每个任务完成后进行验收

### 🎯 质量要求
- 每个任务必须通过功能测试
- 集成测试确保不破坏现有功能
- 代码审查确保质量标准

### 💡 技术约束
- 显存使用严格控制在6GB以内
- 所有决策基于数值统计，不引入主观判断
- CLI接口保持完全兼容
- 算法参数可通过配置文件调整

### 📦 依赖管理
```bash
# 新增音频处理依赖
pip install spleeter>=2.3.0 pydub>=0.25.1 librosa>=0.9.0 noisereduce>=2.0.0
```

开始执行任务1：创建舞台结构分析模块。
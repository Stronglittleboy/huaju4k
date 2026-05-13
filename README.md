# huaju4k - 话剧视频 4K 增强工具

专为话剧录像定制的 4K 增强工具。针对话剧拍摄的固定机位、舞台灯光、对白为核心三大特征，在 6GB 显存消费级显卡上实现高质量 4K 超分。

---

## 技术方案（5W2H）

### What — 做什么

将话剧录像（720p/1080p）升级到 4K，同时增强对白清晰度。

**不是通用视频放大器**。话剧录像有三个区别于普通视频的核心特征，本工具的全部设计围绕这三点展开：

| 话剧特征 | 与普通视频的区别 | 对处理策略的影响 |
|---|---|---|
| **固定机位** | 摄像机三脚架固定，背景（舞台布景）在同一灯光场景内静止 | 按场景分段，每段背景超分一次复用，节省 60-80% GPU 算力 |
| **舞台灯光** | 聚光灯+大面积暗区，有幕间/场景灯光切换，也有渐变灯光 | 不做全局曝光校正（保留灯光意图），灯光突变和渐变均作为场景分段依据 |
| **对白为核心** | 台词是内容主体，观众席录制有环境噪声 | 音频处理以 VAD 检测的人声频段增强为优先，抑制观众噪声 |

### Why — 为什么重新设计

旧方案有两个根本错误：

**错误 1：放大顺序反了**

```
旧方案（错误）:
  Stage 4.1: FFmpeg lanczos 1080p → 4K     ← 先用传统算法放大
  Stage 4.2: Real-ESRGAN 对 4K 再处理       ← ESRGAN 是放大模型，输入 4K 会输出 16K

新方案（正确）:
  Stage 1: 在 1080p 上预处理                 ← 低分辨率处理更快
  Stage 2: Real-ESRGAN 1080p → 4K           ← 模型在原始分辨率做放大，这是它被训练的用途
  Stage 3: 在 4K 上后处理                    ← 修复 AI 逐帧处理的闪烁
```

**错误 2：没有利用话剧固定机位特征**

旧方案对每一帧都做完整的 AI 超分。但话剧固定机位——在同一灯光场景内，背景不变。对背景重复做 AI 超分是巨大的算力浪费。新方案引入**场景分段 + 背景复用**：先按灯光变化切割场景，每个场景内提取背景超分一次缓存，后续帧只对演员运动区域做逐帧 AI 处理。

### Who — 角色分工

**使用者**：有话剧录像的用户，通过 CLI 处理。

```bash
# 完整处理
python -m huaju4k enhance input.mp4 -o output_4k.mp4

# 预览模式（默认取第一个 static 段开头 30 秒；可指定时间点）
python -m huaju4k enhance input.mp4 --preview
python -m huaju4k enhance input.mp4 --preview-at 00:30:00

# 片段处理 — 只处理指定时间段
python -m huaju4k enhance input.mp4 --segment 00:15:00 00:25:00

# 质量档位
python -m huaju4k enhance input.mp4 --quality fast      # lanczos，分钟级
python -m huaju4k enhance input.mp4 --quality standard   # AI 超分 + 背景复用（默认）
python -m huaju4k enhance input.mp4 --quality master     # 全帧 AI 超分，最高质量

# 输出格式
python -m huaju4k enhance input.mp4 --codec h265 --crf 20   # H.265 更小体积
python -m huaju4k enhance input.mp4 --codec prores           # ProRes 无损，供后期剪辑

# 中断后恢复
python -m huaju4k enhance input.mp4 --resume
```

**处理启动前确认（长时任务知情权）**：

```
输入: theater_show.mp4 (1080p, 1:42:15, 5.2GB)
硬件: NVIDIA RTX 3060 6GB, 可用显存 5.1GB
策略: x4plus (tile=384) + 场景级背景复用
场景: 12 段 (11 固定机位 / 1 非固定), 其中 2 段含渐变灯光
预估: 处理时间 ~26 小时, 输出 ~18GB, 需磁盘空间 ~25GB (可用 120GB ✓)
断点: 每 500 帧自动保存，中断后可恢复

是否开始处理? [Y/n]
```

**处理完成报告**：

```
处理完成 ✓
  耗时: 25h 42m | 平均速度: 1.05 fps
  输出: output_4k.mp4 (3840x2160, 17.8GB)
  场景: 12 段 | static 9 段 (背景复用) | gradual 2 段 | dynamic 1 段
  GPU: x4plus tile=384，未发生降级
  音频: 对白增强 +4.2dB (人声段 68%), 噪底降低 -8.1dB
  报告: output_4k.report.json
```

**模块分工**：

```
┌──────────────────────────────────────────────────────┐
│                 ThreeStageEnhancer                   │
│           （管线编排 + 信号处理 + 断点续传）             │
└───┬──────────────┬───────────────┬───────────────────┘
    │              │               │
    ▼              ▼               ▼
┌────────┐  ┌─────────────┐  ┌──────────┐
│Stage1  │  │ Stage2      │  │ Stage3   │
│预处理   │  │ AI 超分     │  │ 后处理   │
│        │  │             │  │          │
│场景分段 │  │GPU: ESRGAN  │  │区域时序   │
│降噪    │  │CPU: lanczos │  │音频增强   │
│音频提取 │  │场景级背景复用│  │音视频合并  │
└────────┘  └─────────────┘  └──────────┘
```

| 模块 | 文件 | 职责 |
|---|---|---|
| ThreeStageEnhancer | `core/three_stage_enhancer.py` | 管线编排、信号处理、断点续传 |
| SceneManager | `analysis/stage_structure_analyzer.py` | 场景分段（突变+渐变+镜头切换） |
| FFmpegMediaController | `media/ffmpeg_media_controller.py` | 视频 I/O、音频提取/合并 |
| GPUSuperResolver | `gpu_stage/gpu_super_resolver.py` | Real-ESRGAN 帧级超分（单帧/裁切区域） |
| TheaterAudioEnhancer | `core/theater_audio_enhancer.py` | 对白增强、观众噪声抑制 |
| EnhancementStrategyPlanner | `strategy/enhancement_planner.py` | 根据分析结果生成处理策略 |

### When — 处理时序

```
时间轴 ─────────────────────────────────────────────────────→

[分析] → [Stage 1 预处理] → [Stage 2 AI 超分] → [Stage 3 后处理]
 2%          5%                  78%                 15%

详细:

1. 分析阶段（~2%时间）
   ├─ 输入验证:
   │   ├─ 已是 4K → 提示"输入已为 4K，是否仍要处理？"
   │   ├─ 手机竖屏 → 提示输出为竖屏 4K
   │   ├─ < 720p → 警告"分辨率过低，AI 超分效果有限"
   │   └─ 可变帧率 → 自动转固定帧率
   ├─ 采样 30 帧，计算: 帧间差异、噪点水平、高光/暗部比例
   ├─ 场景分段（两级检测）:
   │   ├─ 突变检测: 帧差 > 均值+3σ → 硬切割点（幕间/灯光骤变）
   │   ├─ 渐变检测: 滑动窗口内背景均值亮度单调变化 > 阈值 → 渐变段标记
   │   └─ 输出: 场景列表 [{start, end, type: static|gradual|dynamic}, ...]
   ├─ 每段独立判定策略:
   │   ├─ static（静态灯光+固定机位）→ 启用背景复用
   │   ├─ gradual（渐变灯光+固定机位）→ 不复用，全帧处理
   │   └─ dynamic（非固定机位/大量运动）→ 不复用，全帧处理
   ├─ 资源预估: 磁盘空间、处理时间、输出大小 → 交互确认
   └─ 输出: StructureFeatures → EnhancementStrategy

2. Stage 1 预处理（~5%时间，在原始分辨率上操作）
   ├─ FFmpeg 提取音频 → audio.wav
   ├─ FFmpeg 降噪: hqdn3d（强度由噪点分析决定）
   ├─ 不做全局曝光/色彩校正（保留舞台灯光意图）
   └─ 输出: 干净的视频 + 独立音频

3. Stage 2 AI 超分（~78%时间，核心，支持断点续传）
   ├─ 按场景分段逐段处理:
   │   ├─ static 段: 背景提取→超分→缓存，前景逐帧裁切→超分→合成
   │   ├─ gradual/dynamic 段: 全帧逐帧超分
   │   └─ 段切换时: 释放旧背景缓存
   ├─ 断点续传: 每 500 帧保存 checkpoint
   │   ├─ 记录: 已处理帧数、当前场景段、输出文件偏移、输入文件 hash
   │   └─ 中断恢复: 从最近 checkpoint 继续
   ├─ 运行时显存保护:
   │   ├─ 捕获 CUDA OOM → 自动缩小 tile_size 重试
   │   └─ 重试仍失败 → 降级到 CPU lanczos
   ├─ 实时进度:
   │   └─ [████████░░░░] 45.2% | 帧 48521/107280 | 0.8fps | ETA 20h13m | Stage2/Scene7
   └─ 输出: 4K 视频（无音频）

4. Stage 3 后处理（~15%时间，Python 帧级处理 + FFmpeg 编码）
   ├─ 分区域时序修复:
   │   ├─ 实现方式: FFmpeg pipe 解码 → Python 逐帧处理 → FFmpeg pipe 编码
   │   ├─ 运动掩码: 直接对 4K 输出帧差计算（static 段背景相同，帧差 ≡ 前景）
   │   ├─ static 段背景区域: 无需修复（缓存已保证一致）
   │   ├─ 低运动区域: 帧间指数移动平均 (alpha=0.15) 平滑纹理抖动
   │   ├─ 高运动区域: 不做平滑（运动掩盖闪烁，平滑会拖影）
   │   └─ gradual/dynamic 段: deflicker + 低运动区域平滑
   ├─ 轻微锐化: unsharp=3:3:0.3
   ├─ 音频增强:
   │   ├─ VAD 人声检测 → 动态频段增强
   │   └─ 频谱门控降噪（观众噪声抑制）
   ├─ 最终编码:
   │   ├─ 默认: H.264 CRF18 + AAC 192k
   │   ├─ 可选: H.265 CRF20 / ProRes 422 / --crf 自定义
   │   └─ 音视频合并
   └─ 输出: 4K 话剧视频 + 处理报告
```

### Where — 代码结构与重构

```
huaju4k/
├── __init__.py                      # 包入口（更新 import）
├── __main__.py                      # python -m huaju4k 入口
├── cli/
│   └── main.py                      # 唯一 CLI 入口（Click）
├── core/
│   ├── three_stage_enhancer.py      # 管线编排（重写）
│   ├── theater_audio_enhancer.py    # 话剧音频增强
│   ├── dialogue_enhancer.py         # 对白频段增强
│   ├── spatial_audio_optimizer.py   # 空间音频优化
│   ├── ai_model_manager.py          # AI 模型管理
│   ├── interfaces.py                # 抽象接口（更新）
│   ├── memory_manager.py            # 系统内存管理
│   ├── checkpoint_system.py         # 断点续传
│   ├── progress_tracker.py          # 进度追踪与 ETA
│   └── ...（保留现有辅助模块）
├── gpu_stage/
│   ├── gpu_super_resolver.py        # 帧级超分工具（重构为帧级接口）
│   └── model_manager.py             # GPU 模型生命周期
├── media/
│   └── ffmpeg_media_controller.py   # FFmpeg 管线控制
├── analysis/
│   └── stage_structure_analyzer.py  # 场景分段 + 话剧特征分析（扩展）
├── strategy/
│   └── enhancement_planner.py       # 策略生成（修改）
├── models/
│   └── data_models.py               # 数据模型（新建）
├── audio/
│   ├── audio_source_separator.py    # 音源分离
│   └── master_grade_enhancer.py     # 母版级增强
├── configs/
│   └── config_manager.py            # 统一配置管理（整合）
├── utils/
│   ├── system_utils.py
│   ├── validation_utils.py
│   └── file_utils.py
├── setup.py
├── requirements.txt
├── requirements-gpu.txt
├── requirements-dev.txt
└── pytest.ini

重构清单:
  删除: huaju4k/main.py（旧 CLI，与 cli/main.py 冲突）
  删除: configs/simple_config_manager.py, yaml_config_loader.py,
        enhanced_preset_manager.py, preset_cli.py, preset_templates.py,
        default_config.py, cli/config_manager.py
        → 整合为单一 configs/config_manager.py
  新建: models/data_models.py
  重构: gpu_super_resolver.py（视频级循环上移到 ThreeStageEnhancer）
  更新: __init__.py（移除对已删模块的 import）
  更新: setup.py entry_points → cli.main:cli
```

### How — 技术实现细节

#### 数据模型定义（models/data_models.py）

```python
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float          # 秒
    has_audio: bool
    codec: str
    file_size: int           # 字节
    is_vfr: bool             # 可变帧率

@dataclass
class SceneSegment:
    start_frame: int
    end_frame: int
    type: str                # "static" | "gradual" | "dynamic"
    avg_brightness: float
    noise_score: float

@dataclass
class StructureFeatures:
    resolution: Tuple[int, int]
    fps: float
    duration: float
    total_frames: int
    is_static_camera: bool
    noise_score: float       # 0-1
    highlight_ratio: float   # 0-1
    dark_ratio: float        # 0-1
    scenes: List[SceneSegment]

@dataclass
class EnhancementStrategy:
    model_name: str          # "x4plus" | "x2plus" | "lanczos"
    tile_size: int
    scenes: List[SceneSegment]
    denoise_strength: str    # "low" | "medium" | "high"
    audio_enabled: bool
    codec: str               # "h264" | "h265" | "prores"
    crf: int

@dataclass
class ProcessResult:
    success: bool
    output_path: str
    processing_time: float
    frames_processed: int
    error: Optional[str] = None
    report: Optional[Dict] = None

@dataclass
class CheckpointData:
    input_hash: str          # 输入文件前 1MB 的 MD5
    frame_index: int
    scene_index: int
    output_offset: int       # 输出文件字节偏移
    strategy: EnhancementStrategy
    timestamp: float

@dataclass
class AudioResult:
    success: bool
    output_path: str
    snr_improvement_db: float
    voice_ratio: float       # 人声段占比
```

#### Stage 1: 预处理（原始分辨率，话剧定制）

```
输入: 原始话剧录像 (1080p, 含音频)

步骤:
  1. FFmpeg 提取音频轨 → audio.wav

  2. 场景分段检测（两级）:
     a. 全片扫描帧间差异（缩略图 320p 加速）
     b. 突变检测:
        - 帧差 > 均值+3σ → 硬切割点
        - 适用: 幕间暗场、灯光骤变、镜头切换
     c. 渐变检测:
        - 滑动窗口 (5秒) 内，背景区域均值亮度单调变化 > 15%
        - 标记为 gradual 段 → 不启用背景复用
        - 适用: 灯光缓慢淡入淡出、聚光灯移动
     d. 每段分类:
        - static:  固定机位 + 稳定灯光 → 启用背景复用
        - gradual: 固定机位 + 渐变灯光 → 全帧处理
        - dynamic: 非固定 / 大量运动 → 全帧处理

  3. 话剧定制降噪:
     - 高噪点（观众席远景拍摄）: hqdn3d=3:3:10:10
     - 中噪点: hqdn3d=2:2:6:6
     - 低噪点: hqdn3d=1:1:4:4

  4. 不做的事（保留灯光意图）:
     - 不做全局曝光/亮度校正
     - 不做自动色彩增强

输出: stage1_clean.mp4 + audio.wav + scenes.json
```

#### Stage 2: AI 超分辨率（核心放大，6GB 显存优化）

**6GB 显存预算**：

```
可用显存: ~5.5GB（系统/桌面保留 ~0.5GB）

实际显存组成:
  CUDA context:                    ~300MB
  模型权重 (FP16):                 ~500MB (x4plus) / ~400MB (x2plus)
  tile 处理缓冲:                   ~1.2-2.0GB（取决于 tile_size）
  PyTorch 计算图 + 碎片:           ~500-700MB
  ──────────────────────────────
  合计峰值:                        ~2.5-3.5GB

策略选择（运行时按实际可用显存自动决定）:

  默认: x4plus + tile=384
    峰值: ~3.0GB，一步到位 1080p → 4K
    适用: 可用显存 ≥ 3.5GB（6GB 卡的常规情况）

  降级: x2plus + tile=384 两步放大
    峰值: ~2.5GB，两步间插入轻微降噪抑制伪影
    适用: 可用显存 < 3.5GB（降级策略，非质量优选）

  回退: FFmpeg lanczos + unsharp（CPU）
    适用: 无 GPU 或 GPU 不可用

运行时保护:
  捕获 CUDA OOM → tile_size 减半重试（384→192→96）
  仍 OOM → 降级 x2plus → 仍 OOM → CPU lanczos
```

**话剧场景级背景复用（仅对 static 段生效）**：

```
for each scene in scene_list:

  if scene.type == "static":

    1. 背景提取:
       - 段首 2 帧 + 段尾 2 帧 + 中间均匀 6 帧 = 10 帧
         （首尾混合采样，降低静止演员被融入背景的概率）
       - 计算像素中位数 → 本段静态背景 (1080p)
       - AI 超分背景帧一次 → 4K 背景 (缓存到内存, ~24MB)

    2. 逐帧处理:
       - 当前帧与本段背景帧做差 → 运动区域
       - 掩码处理:
         · 高斯模糊 (σ=3) 消除噪点假运动
         · Otsu 自适应阈值二值化
         · 形态学膨胀（确保演员边缘完整）
       - 裁切运动区域 bbox:
         · 坐标对齐 4px（x4 模型要求）
         · 四周加 pad = 画面宽度 × 0.8%（~30px@4K）
       - 仅对裁切区域 AI 超分 → 放大后粘回 4K 背景
       - pad 区域内 alpha 线性渐变混合

    3. 段结束: 释放 4K 背景缓存

  elif scene.type in ("gradual", "dynamic"):
    - 全帧逐帧 AI 超分

效果:
  - static 段: GPU 计算量减少 60-80%, 背景零闪烁
  - gradual 段: 无优化但保证灯光渐变正确
```

#### Stage 3: 后处理（4K，Python 帧级处理 + FFmpeg 编码）

```
输入: stage2_upscaled.mp4 (4K, 无音频) + audio.wav + scenes.json

实现方式: FFmpeg pipe 解码 → Python NumPy 逐帧处理 → FFmpeg pipe 编码
（分区域时序无法用纯 FFmpeg 滤镜实现，必须 Python 帧级处理）

分区域时序修复:

  运动掩码获取方式:
    - 直接对 Stage 2 输出的 4K 帧做帧间差异
    - static 段: 背景每帧相同 → 帧差 ≡ 前景运动区域（无需额外数据传递）
    - gradual/dynamic 段: 帧差区分高/低运动

  static 段:
    - 背景区域 (帧差=0): 跳过（天然一致）
    - 低运动前景 (0 < 帧差 < 高阈值): 帧间指数移动平均 alpha=0.15
    - 高运动前景 (帧差 ≥ 高阈值): 不平滑

  gradual/dynamic 段:
    - FFmpeg deflicker 预处理全局亮度
    - 低运动区域: 帧间指数平滑 alpha=0.15
    - 高运动区域: 不平滑

  全局锐化: unsharp=3:3:0.3

话剧音频增强:
  1. VAD 人声检测 → 动态频段增强（3-6dB）
  2. 频谱门控降噪（观众噪声抑制）
  3. 剧场空间感优化（预设: 小/中/大剧场）

最终编码:
  - 默认: H.264 CRF18 + AAC 192k
  - 可选: H.265 CRF20 / ProRes 422 / --crf 自定义

输出: 4K 话剧视频 + 处理报告 (report.json)
```

#### 进程生命周期管理

```
长任务必备的三项保护:

1. 信号处理:
   - SIGINT (Ctrl+C): 保存 checkpoint → 清理临时资源 → 退出码 130
   - SIGTERM (kill):  保存 checkpoint → 清理临时资源 → 退出码 143
   - 注册 signal.signal() 在 ThreeStageEnhancer 初始化时

2. 资源清理:
   - atexit.register() 注册临时文件清理
   - 确保异常退出也能清理 stage1_clean.mp4 等中间文件
   - checkpoint 文件和 scenes.json 保留（供 --resume 使用）

3. 日志:
   - RotatingFileHandler: 单文件 50MB 上限，保留 3 个备份
   - 日志路径: 输出目录下 huaju4k.log
   - 关键事件（段切换/降级/OOM）同时输出到终端和日志
```

### How Much — 资源与约束

#### 硬件约束（基于 6GB 消费级显卡）

| 资源 | 硬约束 | 实际使用 | 控制方式 |
|---|---|---|---|
| GPU 显存 | 6GB（可用 ~5.5GB） | x4plus 峰值 ~3.0GB | 运行时检测 + OOM 自动降级 |
| 系统内存 | 取决于用户 | ≤ 2GB（单段背景缓存 24MB + 工作区） | 逐段处理，段间释放 |
| 磁盘空间 | 输入 × 4 | 阶段中间文件 + 输出 | 处理前预估确认，Stage 间保留直到全部完成 |

#### 性能预估（1080p → 4K，6GB GPU）

| 场景 | 方案 | 预估速度 | 1h 视频耗时 |
|---|---|---|---|
| 固定机位话剧 | x4plus + 背景复用 | ~1.0 fps | ~24h |
| 含渐变灯光段 | x4plus 混合 | ~0.6 fps | ~40h |
| 非固定机位 | x4plus 全帧 | ~0.4 fps | ~60h |
| 无 GPU | FFmpeg lanczos | ~3 fps | ~8h |

#### 质量档位

| 档位 | 模型 | 背景复用 | 适用场景 | 速度 |
|---|---|---|---|---|
| `fast` | FFmpeg lanczos | 否 | 快速预览、测试 | ~3 fps |
| `standard` | x4plus | 是（static 段） | 日常使用（默认） | ~1 fps |
| `master` | x4plus 全帧 | 否 | 归档、最高质量 | ~0.4 fps |

#### 断点续传

| 项目 | 说明 |
|---|---|
| checkpoint 间隔 | 每 500 帧（约 20 秒视频 @25fps） |
| checkpoint 内容 | 帧数、场景段索引、输出偏移、策略状态、输入文件 hash |
| 恢复方式 | `python -m huaju4k enhance input.mp4 --resume` |
| 文件匹配 | 按输入文件前 1MB 的 MD5 匹配，非路径依赖 |
| 中间文件策略 | Stage 1 输出保留到 Stage 3 完成后才清理 |

#### 质量约束

| 指标 | 目标 | 说明 |
|---|---|---|
| 分辨率 | 3840×2160 | 标准 4K UHD |
| static 段背景闪烁 | 0 | 缓存复用保证 |
| gradual 段正确性 | 无亮度跳变 | 全帧处理，不使用可能不匹配的背景缓存 |
| 运动区域拖影 | 0 | 高运动区域不做时序平滑 |
| 对白清晰度 | SNR 提升 ≥ 3dB | VAD 动态频段增强 |
| 灯光保真 | 不改变曝光/亮度 | 不做全局曝光校正 |

#### 已知限制

| 限制 | 影响 | 建议 |
|---|---|---|
| 硬字幕/水印 | AI 超分可能使文字变形 | 处理前去除，或用 `--subtitle-mask` 标注跳过 |
| 可变帧率 | 音画不同步 | 工具自动转固定帧率 |
| 极低分辨率 (<720p) | AI 效果有限 | 建议输入不低于 720p |
| 多声道录音 | 仅支持立体声/单声道 | 多声道需先混缩 |
| 渐变灯光段 | 无法使用背景复用 | 接受速度降低，保证正确性 |

---

## 快速开始

```bash
# 安装核心依赖
pip install -r requirements.txt

# GPU 超分支持（需要 NVIDIA 6GB+ 显卡）
pip install -r requirements-gpu.txt

# 完整处理
python -m huaju4k enhance input.mp4 -o output_4k.mp4

# 先预览效果
python -m huaju4k enhance input.mp4 --preview

# 只处理精华片段
python -m huaju4k enhance input.mp4 --segment 00:15:00 00:25:00

# 中断后恢复
python -m huaju4k enhance input.mp4 --resume
```

## 系统要求

- Python 3.8+
- FFmpeg（必须）
- 4GB+ RAM（推荐 8GB+）
- NVIDIA GPU 6GB+（可选，用于 AI 超分加速）

## 许可证

MIT License

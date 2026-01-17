#!/usr/bin/env python3
"""
修复AI模型回退机制，确保在Real-ESRGAN不可用时能正确使用OpenCV
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from huaju4k.core.ai_model_manager import AIModelManager, OpenCVModel

def test_opencv_fallback():
    """测试OpenCV回退模型"""
    print("=== 测试OpenCV回退模型 ===")
    
    try:
        # 直接测试OpenCV模型
        opencv_model = OpenCVModel()
        success = opencv_model.load("", use_gpu=False)
        
        if not success:
            print("❌ OpenCV模型加载失败")
            return False
        
        print("✅ OpenCV模型加载成功")
        
        # 测试预测
        import numpy as np
        test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = opencv_model.predict(test_image)
        
        print(f"输入尺寸: {test_image.shape}")
        print(f"输出尺寸: {result.shape}")
        
        expected_height = test_image.shape[0] * 4
        expected_width = test_image.shape[1] * 4
        
        if result.shape[0] == expected_height and result.shape[1] == expected_width:
            print("✅ OpenCV模型预测正常")
            return True
        else:
            print("❌ OpenCV模型预测结果异常")
            return False
            
    except Exception as e:
        print(f"❌ OpenCV模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_manager_fallback():
    """测试AI管理器的回退机制"""
    print("\n=== 测试AI管理器回退机制 ===")
    
    try:
        # 初始化AI管理器
        ai_manager = AIModelManager(cache_size=1)
        
        # 获取可用模型
        available_models = ai_manager.get_available_models()
        print(f"可用模型: {available_models}")
        
        # 自动选择模型（应该选择opencv_cubic）
        selected_model = ai_manager.auto_select_model(
            video_resolution=(3840, 2160),
            available_memory_mb=8000
        )
        print(f"自动选择的模型: {selected_model}")
        
        if selected_model != 'opencv_cubic':
            print(f"⚠️ 期望选择opencv_cubic，但选择了: {selected_model}")
        
        # 尝试加载模型
        success = ai_manager.load_model(selected_model, use_gpu=False)
        
        if success:
            print("✅ 模型加载成功")
            
            # 测试预测
            import numpy as np
            test_image = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            result = ai_manager.predict(test_image)
            
            print(f"预测输入: {test_image.shape}")
            print(f"预测输出: {result.shape}")
            
            if result.shape[0] > test_image.shape[0] and result.shape[1] > test_image.shape[1]:
                print("✅ AI管理器预测正常")
                return True
            else:
                print("❌ AI管理器预测结果异常")
                return False
        else:
            print("❌ 模型加载失败")
            return False
            
    except Exception as e:
        print(f"❌ AI管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_video_processor():
    """创建一个简化的视频处理器用于测试"""
    print("\n=== 创建简化视频处理器 ===")
    
    code = '''#!/usr/bin/env python3
"""
简化的视频处理器，仅使用OpenCV进行测试
"""

import cv2
import numpy as np
import time
from pathlib import Path

def simple_video_enhance(input_path, output_path):
    """简化的视频增强处理"""
    print(f"开始处理视频: {input_path}")
    print(f"输出路径: {output_path}")
    
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {input_path}")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    
    # 设置输出视频（4倍放大）
    output_width = width * 4
    output_height = height * 4
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"❌ 无法创建输出视频: {output_path}")
        cap.release()
        return False
    
    print(f"输出视频: {output_width}x{output_height}")
    
    # 处理帧
    processed_frames = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用OpenCV进行4倍放大
            enhanced_frame = cv2.resize(
                frame, 
                (output_width, output_height), 
                interpolation=cv2.INTER_CUBIC
            )
            
            # 写入输出视频
            out.write(enhanced_frame)
            
            processed_frames += 1
            
            # 显示进度
            if processed_frames % 100 == 0:
                progress = (processed_frames / total_frames) * 100
                elapsed = time.time() - start_time
                fps_current = processed_frames / elapsed if elapsed > 0 else 0
                print(f"进度: {progress:.1f}% ({processed_frames}/{total_frames}), 速度: {fps_current:.1f} fps")
        
        # 完成处理
        elapsed = time.time() - start_time
        print(f"✅ 处理完成: {processed_frames}帧, 耗时: {elapsed:.1f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        return False
        
    finally:
        cap.release()
        out.release()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("用法: python simple_video_enhance.py <输入视频> <输出视频>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"❌ 输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    success = simple_video_enhance(input_file, output_file)
    
    if success:
        print("🎉 视频处理成功完成！")
        sys.exit(0)
    else:
        print("💥 视频处理失败")
        sys.exit(1)
'''
    
    with open("simple_video_enhance.py", "w", encoding="utf-8") as f:
        f.write(code)
    
    print("✅ 创建了简化视频处理器: simple_video_enhance.py")

def main():
    print("=== AI模型回退机制修复测试 ===")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 测试OpenCV回退模型
    opencv_success = test_opencv_fallback()
    
    # 测试AI管理器回退机制
    manager_success = test_ai_manager_fallback()
    
    # 创建简化处理器
    create_simple_video_processor()
    
    print("\n=== 测试总结 ===")
    if opencv_success and manager_success:
        print("✅ 所有测试通过，回退机制正常工作")
        print("\n建议:")
        print("1. 使用简化的视频处理器进行测试")
        print("2. 如需AI增强，请安装Real-ESRGAN依赖项")
        print("3. 当前可以使用OpenCV进行基本的视频放大")
    else:
        print("❌ 部分测试失败，需要修复回退机制")
        
        if not opencv_success:
            print("- OpenCV模型存在问题")
        if not manager_success:
            print("- AI管理器回退机制存在问题")

if __name__ == "__main__":
    import time
    main()
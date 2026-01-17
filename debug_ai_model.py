#!/usr/bin/env python3
"""
调试AI模型加载和预测问题
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from huaju4k.core.ai_model_manager import AIModelManager
from huaju4k.utils.system_utils import check_gpu_availability

def test_ai_model():
    """测试AI模型加载和预测"""
    print("=== AI模型测试 ===")
    
    # 检查GPU可用性
    gpu_info = check_gpu_availability()
    print(f"GPU可用: {gpu_info.get('gpu_available', False)}")
    print(f"CUDA设备数: {gpu_info.get('cuda_device_count', 0)}")
    
    # 初始化AI模型管理器
    try:
        ai_manager = AIModelManager(cache_size=1)
        print("AI模型管理器初始化成功")
    except Exception as e:
        print(f"AI模型管理器初始化失败: {e}")
        return False
    
    # 自动选择模型
    try:
        model_name = ai_manager.auto_select_model(
            target_resolution=(3840, 2160),  # 4K
            available_memory=8000  # 8GB
        )
        print(f"自动选择的模型: {model_name}")
    except Exception as e:
        print(f"模型选择失败: {e}")
        return False
    
    # 加载模型
    print(f"正在加载模型: {model_name}")
    start_time = time.time()
    
    try:
        success = ai_manager.load_model(model_name, use_gpu=True)
        load_time = time.time() - start_time
        
        if success:
            print(f"模型加载成功，耗时: {load_time:.2f}秒")
        else:
            print("模型加载失败")
            return False
    except Exception as e:
        print(f"模型加载异常: {e}")
        return False
    
    # 创建测试图像
    print("创建测试图像...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"测试图像尺寸: {test_image.shape}")
    
    # 测试预测
    print("开始预测测试...")
    start_time = time.time()
    
    try:
        result = ai_manager.predict(test_image)
        predict_time = time.time() - start_time
        
        print(f"预测完成，耗时: {predict_time:.2f}秒")
        print(f"输出图像尺寸: {result.shape}")
        
        # 检查输出是否合理
        if result.shape[0] > test_image.shape[0] and result.shape[1] > test_image.shape[1]:
            print("✅ 预测结果正常（图像被放大）")
            return True
        else:
            print("❌ 预测结果异常（图像未被放大）")
            return False
            
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_model():
    """测试回退模型"""
    print("\n=== 回退模型测试 ===")
    
    try:
        from huaju4k.core.ai_model_manager import OpenCVModel
        
        model = OpenCVModel()
        success = model.load("", use_gpu=False)
        
        if not success:
            print("OpenCV回退模型加载失败")
            return False
        
        # 测试预测
        test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = model.predict(test_image)
        
        print(f"输入尺寸: {test_image.shape}")
        print(f"输出尺寸: {result.shape}")
        
        if result.shape[0] == test_image.shape[0] * 4 and result.shape[1] == test_image.shape[1] * 4:
            print("✅ OpenCV回退模型工作正常")
            return True
        else:
            print("❌ OpenCV回退模型输出异常")
            return False
            
    except Exception as e:
        print(f"回退模型测试失败: {e}")
        return False

def check_dependencies():
    """检查依赖项"""
    print("\n=== 依赖项检查 ===")
    
    dependencies = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("realesrgan", "Real-ESRGAN"),
        ("basicsr", "BasicSR")
    ]
    
    missing_deps = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} 可用")
        except ImportError:
            print(f"❌ {name} 不可用")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\n缺少依赖项: {', '.join(missing_deps)}")
        print("请安装缺少的依赖项:")
        if "Real-ESRGAN" in missing_deps:
            print("  pip install realesrgan")
        if "BasicSR" in missing_deps:
            print("  pip install basicsr")
        if "PyTorch" in missing_deps:
            print("  pip install torch torchvision")
        return False
    
    return True

def main():
    print("=== huaju4k AI模型调试 ===")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查依赖项
    if not check_dependencies():
        print("\n❌ 依赖项检查失败，无法继续测试")
        return
    
    # 测试AI模型
    ai_success = test_ai_model()
    
    # 测试回退模型
    fallback_success = test_fallback_model()
    
    print("\n=== 测试总结 ===")
    if ai_success:
        print("✅ AI模型工作正常")
    else:
        print("❌ AI模型存在问题")
        
    if fallback_success:
        print("✅ 回退模型工作正常")
    else:
        print("❌ 回退模型存在问题")
    
    if not ai_success and not fallback_success:
        print("\n🚨 所有模型都存在问题，需要修复")
    elif not ai_success:
        print("\n⚠️ AI模型有问题，但回退模型可用")
    else:
        print("\n✅ 模型系统正常")

if __name__ == "__main__":
    main()
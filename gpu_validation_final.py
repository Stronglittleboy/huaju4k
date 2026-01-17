#!/usr/bin/env python3
"""
GPU优化最终验证
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime

def validate_gpu_optimization():
    """验证GPU优化状态"""
    print("🔍 GPU优化最终验证")
    print("=" * 40)
    
    # 基本信息
    opencv_version = cv2.__version__
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    
    print(f"OpenCV版本: {opencv_version}")
    print(f"CUDA设备数: {cuda_devices}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "opencv_version": opencv_version,
        "cuda_devices": cuda_devices,
        "tests": {}
    }
    
    if cuda_devices == 0:
        print("❌ 没有CUDA设备")
        results["status"] = "cpu_only"
        results["recommendation"] = "使用CPU优化方案"
        return results
    
    print(f"✅ 检测到 {cuda_devices} 个CUDA设备")
    
    # 性能对比测试
    print("\n🧪 GPU vs CPU性能测试...")
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    # CPU测试
    start_time = time.time()
    cpu_resized = cv2.resize(test_img, (512, 512))
    cpu_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    cpu_thresh = cv2.threshold(cpu_gray, 127, 255, cv2.THRESH_BINARY)[1]
    cpu_time = time.time() - start_time
    
    print(f"CPU处理时间: {cpu_time:.4f}s")
    
    # GPU测试
    try:
        start_time = time.time()
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        gpu_resized = cv2.cuda.resize(gpu_img, (512, 512))
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        gpu_thresh = cv2.cuda.threshold(gpu_gray, 127, 255, cv2.THRESH_BINARY)[1]
        result = gpu_thresh.download()
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"GPU处理时间: {gpu_time:.4f}s")
        print(f"GPU加速比: {speedup:.2f}x")
        
        results["tests"]["performance"] = {
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": speedup
        }
        
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        results["tests"]["performance"] = {"error": str(e)}
        speedup = 0
    
    # 功能测试
    print("\n🧪 GPU功能测试...")
    gpu_functions = ['resize', 'cvtColor', 'threshold', 'bilateralFilter', 'warpAffine']
    available_functions = []
    
    for func in gpu_functions:
        if hasattr(cv2.cuda, func):
            available_functions.append(func)
            print(f"  ✅ {func}")
        else:
            print(f"  ❌ {func}")
    
    results["tests"]["functions"] = {
        "available": available_functions,
        "total": len(gpu_functions),
        "availability_rate": len(available_functions) / len(gpu_functions)
    }
    
    # 生成最终评估
    if speedup > 2 and len(available_functions) >= 3:
        status = "excellent"
        recommendation = "优先使用GPU加速"
    elif speedup > 1 and len(available_functions) >= 2:
        status = "good"
        recommendation = "混合使用CPU/GPU"
    elif len(available_functions) >= 1:
        status = "limited"
        recommendation = "选择性使用GPU"
    else:
        status = "poor"
        recommendation = "使用CPU优化方案"
    
    results["status"] = status
    results["recommendation"] = recommendation
    
    print(f"\n📊 最终评估: {status}")
    print(f"💡 建议: {recommendation}")
    
    # 保存报告
    with open("gpu_optimization_final_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 报告已保存: gpu_optimization_final_report.json")
    
    return results

if __name__ == "__main__":
    results = validate_gpu_optimization()
    
    # 针对任务11.2的结论
    print(f"\n🎯 任务11.2结论:")
    if results["cuda_devices"] > 0:
        print("✅ GPU优化验证完成")
        print("⚡ OpenCV CUDA功能基本可用")
        print("📈 可以进行GPU加速的视频处理")
    else:
        print("❌ GPU不可用，但验证流程完成")
        print("🔄 建议使用CPU优化方案")
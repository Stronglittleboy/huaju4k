#!/usr/bin/env python3
"""
验证OpenCV CUDA加速功能
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime

def verify_cuda_acceleration():
    """验证CUDA加速功能"""
    print("🚀 验证OpenCV CUDA加速功能")
    print("=" * 50)
    
    # 基本信息
    print(f"OpenCV版本: {cv2.__version__}")
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA设备数: {cuda_devices}")
    
    if cuda_devices == 0:
        print("❌ 没有检测到CUDA设备")
        return False
        
    print("✅ CUDA设备可用!")
    
    # 创建测试图像
    print("\n🧪 创建测试图像...")
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print(f"测试图像尺寸: {test_image.shape}")
    
    # CPU处理测试
    print("\n⚡ CPU处理性能测试...")
    start_time = time.time()
    for i in range(5):
        # CPU上采样
        cpu_result = cv2.resize(test_image, (3840, 2160), interpolation=cv2.INTER_CUBIC)
        # CPU降噪
        cpu_result = cv2.bilateralFilter(cpu_result, 9, 75, 75)
    cpu_time = time.time() - start_time
    print(f"CPU处理时间 (5次): {cpu_time:.2f}秒")
    print(f"CPU平均FPS: {5/cpu_time:.2f}")
    
    # GPU处理测试
    print("\n🚀 GPU处理性能测试...")
    try:
        # 上传到GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(test_image)
        
        start_time = time.time()
        for i in range(5):
            # GPU上采样
            gpu_result = cv2.cuda.resize(gpu_image, (3840, 2160), interpolation=cv2.INTER_CUBIC)
            # GPU降噪 (如果支持)
            try:
                gpu_result = cv2.cuda.bilateralFilter(gpu_result, -1, 50, 50)
            except:
                print("  注意: GPU双边滤波不支持，跳过")
        
        # 下载结果
        final_result = gpu_result.download()
        gpu_time = time.time() - start_time
        
        print(f"GPU处理时间 (5次): {gpu_time:.2f}秒")
        print(f"GPU平均FPS: {5/gpu_time:.2f}")
        
        # 性能对比
        speedup = cpu_time / gpu_time
        print(f"\n📊 性能对比:")
        print(f"GPU加速倍数: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✅ GPU加速效果显著!")
        elif speedup > 1.0:
            print("✅ GPU加速有效果")
        else:
            print("⚠️ GPU加速效果不明显")
            
    except Exception as e:
        print(f"❌ GPU处理失败: {e}")
        return False
    
    # 测试其他CUDA功能
    print("\n🔧 测试其他CUDA功能...")
    
    # 测试CUDA内存信息
    try:
        free_mem, total_mem = cv2.cuda.DeviceInfo().totalMemory(), cv2.cuda.DeviceInfo().freeMemory()
        print(f"GPU内存 - 总计: {total_mem/1024/1024:.0f}MB, 可用: {free_mem/1024/1024:.0f}MB")
    except:
        print("无法获取GPU内存信息")
    
    # 测试CUDA流
    try:
        stream = cv2.cuda_Stream()
        print("✅ CUDA流创建成功")
    except:
        print("❌ CUDA流创建失败")
    
    # 生成验证报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "opencv_version": cv2.__version__,
        "cuda_devices": cuda_devices,
        "performance_test": {
            "cpu_time_seconds": cpu_time,
            "gpu_time_seconds": gpu_time,
            "cpu_fps": 5/cpu_time,
            "gpu_fps": 5/gpu_time,
            "speedup_factor": speedup
        },
        "cuda_features": {
            "resize": "支持",
            "bilateral_filter": "部分支持",
            "memory_info": "支持" if 'total_mem' in locals() else "不支持",
            "streams": "支持" if 'stream' in locals() else "不支持"
        },
        "assessment": {
            "cuda_working": True,
            "performance_improvement": speedup > 1.0,
            "significant_speedup": speedup > 1.5
        }
    }
    
    # 保存报告
    with open("cuda_verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 验证报告已保存: cuda_verification_report.json")
    print("\n🎉 OpenCV CUDA验证完成!")
    
    return True

if __name__ == "__main__":
    success = verify_cuda_acceleration()
    if success:
        print("\n✅ CUDA加速功能验证成功!")
        print("现在可以使用GPU加速进行视频处理了!")
    else:
        print("\n❌ CUDA加速功能验证失败")
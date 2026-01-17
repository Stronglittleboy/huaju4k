#!/usr/bin/env python3
"""
OpenCV GPU状态检查
"""

import cv2
import numpy as np
import json
from datetime import datetime

def check_opencv_gpu_status():
    """检查OpenCV GPU状态"""
    print("🔍 OpenCV GPU状态检查")
    print("=" * 40)
    
    # 基本信息
    opencv_version = cv2.__version__
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    
    print(f"OpenCV版本: {opencv_version}")
    print(f"CUDA设备数: {cuda_devices}")
    
    if cuda_devices == 0:
        print("❌ 没有检测到CUDA设备")
        return False
    
    print(f"✅ 检测到 {cuda_devices} 个CUDA设备")
    
    # 测试基本CUDA功能
    test_results = {}
    
    try:
        # 1. 测试GPU内存分配
        print("\n🧪 测试GPU内存分配...")
        gpu_mat = cv2.cuda_GpuMat(100, 100, cv2.CV_8UC3)
        print("✅ GPU内存分配成功")
        test_results["memory_allocation"] = True
        
        # 2. 测试数据上传/下载
        print("\n🧪 测试数据传输...")
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        downloaded_img = gpu_img.download()
        
        if np.array_equal(test_img, downloaded_img):
            print("✅ 数据传输成功")
            test_results["data_transfer"] = True
        else:
            print("❌ 数据传输失败")
            test_results["data_transfer"] = False
        
        # 3. 测试基本图像处理
        print("\n🧪 测试GPU图像处理...")
        
        # 测试resize
        try:
            gpu_resized = cv2.cuda.resize(gpu_img, (200, 200))
            print("✅ GPU resize 可用")
            test_results["resize"] = True
        except Exception as e:
            print(f"❌ GPU resize 失败: {e}")
            test_results["resize"] = False
        
        # 测试颜色转换
        try:
            gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
            print("✅ GPU cvtColor 可用")
            test_results["cvtColor"] = True
        except Exception as e:
            print(f"❌ GPU cvtColor 失败: {e}")
            test_results["cvtColor"] = False
        
        # 测试高斯模糊
        try:
            gpu_blur = cv2.cuda.GaussianBlur(gpu_img, (15, 15), 0)
            print("✅ GPU GaussianBlur 可用")
            test_results["GaussianBlur"] = True
        except Exception as e:
            print(f"❌ GPU GaussianBlur 失败: {e}")
            test_results["GaussianBlur"] = False
        
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False
    
    # 统计结果
    successful_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n📊 测试结果: {successful_tests}/{total_tests} 通过")
    
    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "opencv_version": opencv_version,
        "cuda_devices": cuda_devices,
        "test_results": test_results,
        "success_rate": successful_tests / total_tests,
        "status": "可用" if successful_tests >= 3 else "有限可用" if successful_tests >= 1 else "不可用"
    }
    
    # 保存报告
    with open("opencv_gpu_status_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💡 总体状态: {report['status']}")
    
    if successful_tests >= 3:
        print("🎉 OpenCV GPU功能基本可用，可以进行GPU加速处理")
        return True
    elif successful_tests >= 1:
        print("⚠️ OpenCV GPU功能有限，建议混合使用CPU/GPU")
        return True
    else:
        print("❌ OpenCV GPU功能不可用，建议使用CPU处理")
        return False

if __name__ == "__main__":
    success = check_opencv_gpu_status()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
完整的CUDA验证和状态报告
"""

import cv2
import numpy as np
import json
from datetime import datetime

def complete_cuda_verification():
    """完整的CUDA验证"""
    print("🔍 OpenCV CUDA完整验证")
    print("=" * 50)
    
    # 基本信息
    opencv_version = cv2.__version__
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    
    print(f"OpenCV版本: {opencv_version}")
    print(f"CUDA设备数: {cuda_devices}")
    
    verification_results = {
        "timestamp": datetime.now().isoformat(),
        "opencv_version": opencv_version,
        "cuda_devices_detected": cuda_devices,
        "tests": {}
    }
    
    if cuda_devices == 0:
        print("❌ 没有检测到CUDA设备")
        verification_results["status"] = "CUDA不可用"
        return verification_results
    
    print("✅ 检测到CUDA设备!")
    
    # 测试1: GPU内存分配
    print("\n🧪 测试1: GPU内存分配")
    try:
        gpu_mat = cv2.cuda_GpuMat(100, 100, cv2.CV_8UC3)
        print("✅ GPU内存分配成功")
        verification_results["tests"]["gpu_memory_allocation"] = "成功"
    except Exception as e:
        print(f"❌ GPU内存分配失败: {e}")
        verification_results["tests"]["gpu_memory_allocation"] = f"失败: {e}"
    
    # 测试2: 数据上传下载
    print("\n🧪 测试2: 数据上传下载")
    try:
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        downloaded_img = gpu_img.download()
        
        if np.array_equal(test_img, downloaded_img):
            print("✅ 数据上传下载成功")
            verification_results["tests"]["data_transfer"] = "成功"
        else:
            print("❌ 数据传输验证失败")
            verification_results["tests"]["data_transfer"] = "数据不匹配"
    except Exception as e:
        print(f"❌ 数据传输失败: {e}")
        verification_results["tests"]["data_transfer"] = f"失败: {e}"
    
    # 测试3: 基本CUDA操作
    print("\n🧪 测试3: 基本CUDA操作")
    cuda_operations = {
        "resize": False,
        "cvtColor": False,
        "threshold": False,
        "blur": False
    }
    
    try:
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        
        # 测试resize
        try:
            gpu_resized = cv2.cuda.resize(gpu_img, (200, 200))
            cuda_operations["resize"] = True
            print("  ✅ CUDA resize")
        except Exception as e:
            print(f"  ❌ CUDA resize失败: {e}")
        
        # 测试颜色转换
        try:
            gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
            cuda_operations["cvtColor"] = True
            print("  ✅ CUDA cvtColor")
        except Exception as e:
            print(f"  ❌ CUDA cvtColor失败: {e}")
        
        # 测试阈值
        try:
            if cuda_operations["cvtColor"]:
                gpu_thresh = cv2.cuda.threshold(gpu_gray, 127, 255, cv2.THRESH_BINARY)[1]
                cuda_operations["threshold"] = True
                print("  ✅ CUDA threshold")
        except Exception as e:
            print(f"  ❌ CUDA threshold失败: {e}")
        
        # 测试模糊
        try:
            gpu_blur = cv2.cuda.blur(gpu_img, (5, 5))
            cuda_operations["blur"] = True
            print("  ✅ CUDA blur")
        except Exception as e:
            print(f"  ❌ CUDA blur失败: {e}")
            
    except Exception as e:
        print(f"❌ CUDA操作测试失败: {e}")
    
    verification_results["tests"]["cuda_operations"] = cuda_operations
    
    # 测试4: GPU设备信息
    print("\n🧪 测试4: GPU设备信息")
    try:
        device_info = cv2.cuda.DeviceInfo()
        gpu_info = {
            "name": device_info.name(),
            "major_version": device_info.majorVersion(),
            "minor_version": device_info.minorVersion(),
            "multi_processor_count": device_info.multiProcessorCount(),
            "total_memory": device_info.totalMemory(),
            "free_memory": device_info.freeMemory()
        }
        
        print(f"  GPU名称: {gpu_info['name']}")
        print(f"  计算能力: {gpu_info['major_version']}.{gpu_info['minor_version']}")
        print(f"  多处理器数量: {gpu_info['multi_processor_count']}")
        print(f"  总内存: {gpu_info['total_memory'] / 1024 / 1024:.0f}MB")
        print(f"  可用内存: {gpu_info['free_memory'] / 1024 / 1024:.0f}MB")
        
        verification_results["tests"]["gpu_device_info"] = gpu_info
        
    except Exception as e:
        print(f"❌ 获取GPU设备信息失败: {e}")
        verification_results["tests"]["gpu_device_info"] = f"失败: {e}"
    
    # 总结
    working_operations = sum(1 for op in cuda_operations.values() if op)
    total_operations = len(cuda_operations)
    
    print(f"\n📊 CUDA功能总结:")
    print(f"  可用操作: {working_operations}/{total_operations}")
    
    if working_operations == 0:
        status = "CUDA不可用 - 可能是GPU架构兼容性问题"
        print("❌ 所有CUDA操作都失败")
        print("💡 建议: 可能需要重新编译OpenCV，指定正确的GPU架构")
    elif working_operations < total_operations:
        status = "CUDA部分可用"
        print("⚠️ 部分CUDA操作可用")
    else:
        status = "CUDA完全可用"
        print("✅ 所有CUDA操作都可用")
    
    verification_results["status"] = status
    verification_results["working_operations_count"] = working_operations
    verification_results["total_operations_count"] = total_operations
    
    # 保存报告
    report_file = "cuda_verification_complete_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 完整报告已保存: {report_file}")
    
    # 给出建议
    print(f"\n💡 建议:")
    if working_operations > 0:
        print("  ✅ CUDA基本功能可用，可以进行GPU加速处理")
        print("  ⚡ 对于失败的操作，可以使用CPU替代方案")
    else:
        print("  ❌ CUDA功能不可用，建议使用CPU优化方案")
        print("  🔧 或者重新编译OpenCV，确保GPU架构兼容性")
    
    return verification_results

if __name__ == "__main__":
    results = complete_cuda_verification()
    print(f"\n🏁 验证完成，状态: {results['status']}")
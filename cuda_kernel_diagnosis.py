#!/usr/bin/env python3
"""
CUDA内核诊断脚本
诊断CUDA内核不可用的具体原因
"""

import cv2
import numpy as np
import subprocess
import json
from datetime import datetime

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            return {
                "name": gpu_info[0],
                "compute_capability": gpu_info[1],
                "driver_version": gpu_info[2],
                "memory_total": gpu_info[3]
            }
    except:
        pass
    return None

def check_cuda_runtime():
    """检查CUDA运行时"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
    except:
        pass
    return None

def diagnose_opencv_cuda():
    """诊断OpenCV CUDA问题"""
    print("🔍 CUDA内核诊断开始")
    print("=" * 50)
    
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "opencv_version": cv2.__version__,
        "cuda_devices": cv2.cuda.getCudaEnabledDeviceCount(),
        "gpu_info": get_gpu_info(),
        "cuda_runtime": check_cuda_runtime(),
        "opencv_build_info": {},
        "kernel_availability": {},
        "recommendations": []
    }
    
    # 获取OpenCV构建信息
    build_info = cv2.getBuildInformation()
    diagnosis["opencv_build_info"] = {
        "full_info": build_info,
        "cuda_support": "CUDA:" in build_info and "YES" in build_info.split("CUDA:")[1].split("\n")[0],
        "cudnn_support": "cuDNN:" in build_info and "YES" in build_info.split("cuDNN:")[1].split("\n")[0] if "cuDNN:" in build_info else False
    }
    
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"CUDA设备数: {diagnosis['cuda_devices']}")
    
    if diagnosis["gpu_info"]:
        print(f"GPU: {diagnosis['gpu_info']['name']}")
        print(f"计算能力: {diagnosis['gpu_info']['compute_capability']}")
        print(f"驱动版本: {diagnosis['gpu_info']['driver_version']}")
        print(f"显存: {diagnosis['gpu_info']['memory_total']} MB")
    
    # 检查具体的内核可用性
    print("\n🧪 测试基本CUDA操作...")
    
    try:
        # 测试基本内存操作
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(test_img)
        result = gpu_mat.download()
        diagnosis["kernel_availability"]["memory_transfer"] = True
        print("✅ GPU内存传输正常")
    except Exception as e:
        diagnosis["kernel_availability"]["memory_transfer"] = False
        print(f"❌ GPU内存传输失败: {e}")
    
    # 测试设备属性
    try:
        if diagnosis["cuda_devices"] > 0:
            device_info = cv2.cuda.DeviceInfo(0)
            diagnosis["device_properties"] = {
                "name": device_info.name(),
                "major_version": device_info.majorVersion(),
                "minor_version": device_info.minorVersion(),
                "multi_processor_count": device_info.multiProcessorCount(),
                "shared_memory_per_block": device_info.sharedMemPerBlock(),
                "max_threads_per_block": device_info.maxThreadsPerBlock()
            }
            print(f"✅ 设备信息: {device_info.name()}")
            print(f"   计算能力: {device_info.majorVersion()}.{device_info.minorVersion()}")
            print(f"   多处理器数: {device_info.multiProcessorCount()}")
    except Exception as e:
        print(f"❌ 无法获取设备信息: {e}")
    
    # 分析问题并生成建议
    print("\n📋 问题分析:")
    
    if not diagnosis["opencv_build_info"]["cuda_support"]:
        diagnosis["recommendations"].append({
            "priority": "CRITICAL",
            "issue": "OpenCV未启用CUDA支持",
            "solution": "需要重新编译OpenCV并启用CUDA支持"
        })
        print("🔴 CRITICAL: OpenCV未启用CUDA支持")
    
    if diagnosis["gpu_info"] and float(diagnosis["gpu_info"]["compute_capability"]) < 3.5:
        diagnosis["recommendations"].append({
            "priority": "HIGH",
            "issue": f"GPU计算能力过低 ({diagnosis['gpu_info']['compute_capability']})",
            "solution": "需要计算能力3.5或更高的GPU"
        })
        print(f"🔴 HIGH: GPU计算能力过低 ({diagnosis['gpu_info']['compute_capability']})")
    
    if "no kernel image is available" in str(diagnosis):
        diagnosis["recommendations"].append({
            "priority": "HIGH",
            "issue": "CUDA内核映像不可用",
            "solution": "OpenCV编译时未包含当前GPU架构的内核，需要重新编译"
        })
        print("🔴 HIGH: CUDA内核映像不可用")
    
    # 检查是否是预编译版本问题
    if "4.13.0-pre" in cv2.__version__:
        diagnosis["recommendations"].append({
            "priority": "MEDIUM",
            "issue": "使用预发布版本的OpenCV",
            "solution": "建议使用稳定版本的OpenCV或确保预编译版本支持当前GPU架构"
        })
        print("🟡 MEDIUM: 使用预发布版本的OpenCV")
    
    # 保存诊断结果
    with open("cuda_kernel_diagnosis_report.json", "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)
    
    print("\n💡 建议的解决方案:")
    for i, rec in enumerate(diagnosis["recommendations"], 1):
        priority_icon = {"CRITICAL": "🔴", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec["priority"], "ℹ️")
        print(f"{i}. {priority_icon} {rec['issue']}")
        print(f"   解决方案: {rec['solution']}")
    
    print(f"\n📄 详细报告已保存到: cuda_kernel_diagnosis_report.json")
    
    return diagnosis

if __name__ == "__main__":
    diagnose_opencv_cuda()
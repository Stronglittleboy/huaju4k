#!/usr/bin/env python3
"""
最终CUDA状态报告
"""

import cv2
import numpy as np
import json
from datetime import datetime

def generate_final_cuda_report():
    """生成最终CUDA状态报告"""
    print("📋 OpenCV CUDA最终状态报告")
    print("=" * 50)
    
    # 基本信息
    opencv_version = cv2.__version__
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    
    print(f"OpenCV版本: {opencv_version}")
    print(f"CUDA设备数: {cuda_devices}")
    
    # 检查CUDA模块
    cuda_modules = []
    try:
        # 检查可用的CUDA函数
        cuda_attrs = [attr for attr in dir(cv2.cuda) if not attr.startswith('_')]
        print(f"可用CUDA函数数量: {len(cuda_attrs)}")
        
        # 测试基本功能
        test_results = {}
        
        # 1. 内存管理
        try:
            gpu_mat = cv2.cuda_GpuMat(10, 10, cv2.CV_8UC1)
            test_results["memory_allocation"] = "✅ 成功"
        except Exception as e:
            test_results["memory_allocation"] = f"❌ 失败: {str(e)[:30]}"
        
        # 2. 数据传输
        try:
            img = np.ones((10, 10), dtype=np.uint8) * 100
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            downloaded = gpu_img.download()
            if np.array_equal(img, downloaded):
                test_results["data_transfer"] = "✅ 成功"
            else:
                test_results["data_transfer"] = "❌ 数据不匹配"
        except Exception as e:
            test_results["data_transfer"] = f"❌ 失败: {str(e)[:30]}"
        
        # 3. 检查特定函数
        functions_to_test = [
            'resize', 'cvtColor', 'threshold', 'blur', 'GaussianBlur',
            'bilateralFilter', 'Canny', 'morphologyEx'
        ]
        
        available_functions = {}
        for func in functions_to_test:
            if hasattr(cv2.cuda, func):
                available_functions[func] = "可用"
            else:
                available_functions[func] = "不可用"
        
        print(f"\n🔧 CUDA函数可用性:")
        for func, status in available_functions.items():
            print(f"  {func}: {status}")
        
    except Exception as e:
        print(f"❌ CUDA模块检查失败: {e}")
        test_results = {"error": str(e)}
        available_functions = {}
    
    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "opencv_version": opencv_version,
        "cuda_devices": cuda_devices,
        "cuda_status": "可用" if cuda_devices > 0 else "不可用",
        "test_results": test_results,
        "available_functions": available_functions,
        "summary": {
            "basic_cuda": cuda_devices > 0,
            "memory_ops": "memory_allocation" in test_results and "成功" in test_results["memory_allocation"],
            "data_transfer": "data_transfer" in test_results and "成功" in test_results["data_transfer"],
            "function_count": len([f for f in available_functions.values() if f == "可用"])
        }
    }
    
    # 评估状态
    if cuda_devices == 0:
        overall_status = "CUDA不可用"
        recommendation = "使用CPU优化方案"
    elif report["summary"]["memory_ops"] and report["summary"]["data_transfer"]:
        if report["summary"]["function_count"] > 3:
            overall_status = "CUDA基本可用"
            recommendation = "可以使用CUDA加速，但某些高级功能可能不可用"
        else:
            overall_status = "CUDA有限可用"
            recommendation = "基本CUDA功能可用，建议结合CPU处理"
    else:
        overall_status = "CUDA有问题"
        recommendation = "建议重新编译或使用CPU方案"
    
    report["overall_status"] = overall_status
    report["recommendation"] = recommendation
    
    print(f"\n📊 总体状态: {overall_status}")
    print(f"💡 建议: {recommendation}")
    
    # 保存报告
    with open("final_cuda_status_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 报告已保存: final_cuda_status_report.json")
    
    # 针对任务8.1的建议
    print(f"\n🎯 针对任务8.1的建议:")
    if cuda_devices > 0:
        print("  ✅ CUDA已检测到，可以继续任务8.1")
        print("  ⚡ 使用混合方案: CUDA基本操作 + CPU复杂处理")
    else:
        print("  ❌ CUDA不可用，使用CPU优化方案完成任务8.1")
    
    return report

if __name__ == "__main__":
    report = generate_final_cuda_report()
    
    # 更新任务状态
    print(f"\n🏁 CUDA问题解决状态:")
    if report["cuda_devices"] > 0:
        print("✅ CUDA设备检测成功 - 问题已基本解决")
        print("⚡ 可以继续进行GPU加速的视频处理任务")
    else:
        print("❌ CUDA仍不可用 - 需要进一步调试")
        print("🔄 建议使用CPU优化方案作为替代")
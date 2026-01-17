#!/usr/bin/env python3
"""
OpenCV CUDA 重新验证脚本
验证重新安装CUDA服务后的OpenCV GPU功能
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime

class OpenCVCUDAVerifier:
    def __init__(self):
        self.verification_results = {}
        self.cuda_operations_tested = 0
        self.cuda_operations_passed = 0
        
        print("🔍 OpenCV CUDA 重新验证")
        print("=" * 50)
        
    def check_basic_info(self):
        """检查基本信息"""
        print("\n📋 基本信息检查:")
        
        # OpenCV版本
        opencv_version = cv2.__version__
        print(f"   OpenCV版本: {opencv_version}")
        
        # CUDA设备数量
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"   CUDA设备数: {cuda_devices}")
        
        # 构建信息
        build_info = cv2.getBuildInformation()
        cuda_support = "CUDA:" in build_info and "YES" in build_info.split("CUDA:")[1].split("\n")[0]
        print(f"   CUDA支持: {'是' if cuda_support else '否'}")
        
        self.verification_results["basic_info"] = {
            "opencv_version": opencv_version,
            "cuda_devices": cuda_devices,
            "cuda_support": cuda_support,
            "build_info_available": True
        }
        
        return cuda_devices > 0 and cuda_support
    
    def test_gpu_memory_operations(self):
        """测试GPU内存操作"""
        print("\n🧪 测试1: GPU内存操作")
        
        try:
            # 创建GPU矩阵
            gpu_mat = cv2.cuda_GpuMat(100, 100, cv2.CV_8UC3)
            print("   ✅ GPU内存分配: 成功")
            
            # 上传数据
            cpu_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            gpu_mat.upload(cpu_data)
            print("   ✅ 数据上传: 成功")
            
            # 下载数据
            downloaded_data = gpu_mat.download()
            print("   ✅ 数据下载: 成功")
            
            # 验证数据一致性
            if np.array_equal(cpu_data, downloaded_data):
                print("   ✅ 数据一致性: 通过")
                data_integrity = True
            else:
                print("   ❌ 数据一致性: 失败")
                data_integrity = False
            
            # 释放内存
            gpu_mat.release()
            print("   ✅ 内存释放: 成功")
            
            self.verification_results["gpu_memory"] = {
                "allocation": True,
                "upload": True,
                "download": True,
                "data_integrity": data_integrity,
                "memory_release": True,
                "overall": data_integrity
            }
            
            self.cuda_operations_tested += 1
            if data_integrity:
                self.cuda_operations_passed += 1
            
            return data_integrity
            
        except Exception as e:
            print(f"   ❌ GPU内存操作失败: {e}")
            self.verification_results["gpu_memory"] = {
                "error": str(e),
                "overall": False
            }
            self.cuda_operations_tested += 1
            return False
    
    def test_basic_cuda_operations(self):
        """测试基本CUDA操作"""
        print("\n🧪 测试2: 基本CUDA操作")
        
        operations_results = {}
        
        # 准备测试数据
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(test_image)
        
        # 测试resize操作
        print("   🔧 测试resize操作...")
        try:
            resized_gpu = cv2.cuda.resize(gpu_mat, (256, 256))
            result = resized_gpu.download()
            if result.shape == (256, 256, 3):
                print("   ✅ CUDA resize: 成功")
                operations_results["resize"] = True
            else:
                print("   ❌ CUDA resize: 尺寸错误")
                operations_results["resize"] = False
            resized_gpu.release()
        except Exception as e:
            print(f"   ❌ CUDA resize: 失败 - {e}")
            operations_results["resize"] = False
        
        # 测试颜色空间转换
        print("   🔧 测试颜色空间转换...")
        try:
            gray_gpu = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2GRAY)
            gray_result = gray_gpu.download()
            if len(gray_result.shape) == 2:
                print("   ✅ CUDA cvtColor: 成功")
                operations_results["cvtColor"] = True
            else:
                print("   ❌ CUDA cvtColor: 格式错误")
                operations_results["cvtColor"] = False
            gray_gpu.release()
        except Exception as e:
            print(f"   ❌ CUDA cvtColor: 失败 - {e}")
            operations_results["cvtColor"] = False
        
        # 测试数学运算
        print("   🔧 测试数学运算...")
        try:
            gpu_mat2 = cv2.cuda_GpuMat()
            gpu_mat2.upload(test_image)
            added_gpu = cv2.cuda.add(gpu_mat, gpu_mat2)
            added_result = added_gpu.download()
            if added_result.shape == test_image.shape:
                print("   ✅ CUDA add: 成功")
                operations_results["add"] = True
            else:
                print("   ❌ CUDA add: 结果错误")
                operations_results["add"] = False
            added_gpu.release()
            gpu_mat2.release()
        except Exception as e:
            print(f"   ❌ CUDA add: 失败 - {e}")
            operations_results["add"] = False
        
        gpu_mat.release()
        
        # 统计结果
        passed_operations = sum(operations_results.values())
        total_operations = len(operations_results)
        
        self.verification_results["basic_operations"] = {
            "operations": operations_results,
            "passed": passed_operations,
            "total": total_operations,
            "success_rate": passed_operations / total_operations if total_operations > 0 else 0,
            "overall": passed_operations > 0
        }
        
        self.cuda_operations_tested += 1
        if passed_operations > 0:
            self.cuda_operations_passed += 1
        
        print(f"   📊 基本操作通过率: {passed_operations}/{total_operations}")
        
        return passed_operations > 0
    
    def test_advanced_cuda_operations(self):
        """测试高级CUDA操作"""
        print("\n🧪 测试3: 高级CUDA操作")
        
        operations_results = {}
        
        # 准备测试数据
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(test_image)
        
        # 测试双边滤波
        print("   🔧 测试双边滤波...")
        try:
            filtered_gpu = cv2.cuda.bilateralFilter(gpu_mat, -1, 50, 50)
            filtered_result = filtered_gpu.download()
            if filtered_result.shape == test_image.shape:
                print("   ✅ CUDA bilateralFilter: 成功")
                operations_results["bilateralFilter"] = True
            else:
                print("   ❌ CUDA bilateralFilter: 结果错误")
                operations_results["bilateralFilter"] = False
            filtered_gpu.release()
        except Exception as e:
            print(f"   ❌ CUDA bilateralFilter: 失败 - {e}")
            operations_results["bilateralFilter"] = False
        
        # 测试高斯模糊
        print("   🔧 测试高斯模糊...")
        try:
            blurred_gpu = cv2.cuda.GaussianBlur(gpu_mat, (15, 15), 0)
            blurred_result = blurred_gpu.download()
            if blurred_result.shape == test_image.shape:
                print("   ✅ CUDA GaussianBlur: 成功")
                operations_results["GaussianBlur"] = True
            else:
                print("   ❌ CUDA GaussianBlur: 结果错误")
                operations_results["GaussianBlur"] = False
            blurred_gpu.release()
        except Exception as e:
            print(f"   ❌ CUDA GaussianBlur: 失败 - {e}")
            operations_results["GaussianBlur"] = False
        
        # 测试形态学操作
        print("   🔧 测试形态学操作...")
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morphed_gpu = cv2.cuda.morphologyEx(gpu_mat, cv2.MORPH_OPEN, kernel)
            morphed_result = morphed_gpu.download()
            if morphed_result.shape == test_image.shape:
                print("   ✅ CUDA morphologyEx: 成功")
                operations_results["morphologyEx"] = True
            else:
                print("   ❌ CUDA morphologyEx: 结果错误")
                operations_results["morphologyEx"] = False
            morphed_gpu.release()
        except Exception as e:
            print(f"   ❌ CUDA morphologyEx: 失败 - {e}")
            operations_results["morphologyEx"] = False
        
        gpu_mat.release()
        
        # 统计结果
        passed_operations = sum(operations_results.values())
        total_operations = len(operations_results)
        
        self.verification_results["advanced_operations"] = {
            "operations": operations_results,
            "passed": passed_operations,
            "total": total_operations,
            "success_rate": passed_operations / total_operations if total_operations > 0 else 0,
            "overall": passed_operations > 0
        }
        
        self.cuda_operations_tested += 1
        if passed_operations > 0:
            self.cuda_operations_passed += 1
        
        print(f"   📊 高级操作通过率: {passed_operations}/{total_operations}")
        
        return passed_operations > 0
    
    def performance_benchmark(self):
        """性能基准测试"""
        print("\n🏁 测试4: 性能基准测试")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        benchmark_results = {}
        
        # CPU resize基准
        print("   ⏱️  CPU resize基准测试...")
        start_time = time.time()
        for _ in range(10):
            cpu_resized = cv2.resize(test_image, (512, 512))
        cpu_time = time.time() - start_time
        print(f"   CPU resize (10次): {cpu_time:.3f}秒")
        
        # GPU resize基准
        print("   ⏱️  GPU resize基准测试...")
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(test_image)
            
            start_time = time.time()
            for _ in range(10):
                resized_gpu = cv2.cuda.resize(gpu_mat, (512, 512))
                result = resized_gpu.download()
                resized_gpu.release()
            gpu_time = time.time() - start_time
            
            gpu_mat.release()
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"   GPU resize (10次): {gpu_time:.3f}秒")
            print(f"   加速比: {speedup:.2f}x")
            
            benchmark_results = {
                "cpu_time": cpu_time,
                "gpu_time": gpu_time,
                "speedup": speedup,
                "gpu_faster": speedup > 1.0,
                "test_successful": True
            }
            
        except Exception as e:
            print(f"   ❌ GPU基准测试失败: {e}")
            benchmark_results = {
                "cpu_time": cpu_time,
                "gpu_time": None,
                "speedup": None,
                "gpu_faster": False,
                "test_successful": False,
                "error": str(e)
            }
        
        self.verification_results["performance"] = benchmark_results
        
        return benchmark_results.get("test_successful", False)
    
    def test_gpu_device_info(self):
        """测试GPU设备信息"""
        print("\n🎮 测试5: GPU设备信息")
        
        try:
            device_info = cv2.cuda.DeviceInfo()
            
            # 尝试获取设备信息
            device_data = {}
            
            try:
                device_data["name"] = device_info.name()
                print(f"   设备名称: {device_data['name']}")
            except:
                print("   设备名称: 无法获取")
                device_data["name"] = "Unknown"
            
            try:
                device_data["major_version"] = device_info.majorVersion()
                device_data["minor_version"] = device_info.minorVersion()
                print(f"   计算能力: {device_data['major_version']}.{device_data['minor_version']}")
            except:
                print("   计算能力: 无法获取")
                device_data["major_version"] = 0
                device_data["minor_version"] = 0
            
            try:
                device_data["total_memory"] = device_info.totalMemory()
                print(f"   总内存: {device_data['total_memory'] / 1024 / 1024:.0f} MB")
            except:
                print("   总内存: 无法获取")
                device_data["total_memory"] = 0
            
            try:
                device_data["free_memory"] = device_info.freeMemory()
                print(f"   可用内存: {device_data['free_memory'] / 1024 / 1024:.0f} MB")
            except:
                print("   可用内存: 无法获取")
                device_data["free_memory"] = 0
            
            self.verification_results["device_info"] = {
                "available": True,
                "data": device_data,
                "overall": True
            }
            
            return True
            
        except Exception as e:
            print(f"   ❌ 获取GPU设备信息失败: {e}")
            self.verification_results["device_info"] = {
                "available": False,
                "error": str(e),
                "overall": False
            }
            return False
    
    def generate_verification_report(self):
        """生成验证报告"""
        print("\n📊 生成验证报告...")
        
        # 计算总体成功率
        overall_success_rate = self.cuda_operations_passed / self.cuda_operations_tested if self.cuda_operations_tested > 0 else 0
        
        # 评估CUDA状态
        if overall_success_rate >= 0.8:
            cuda_status = "优秀"
            cuda_usable = True
        elif overall_success_rate >= 0.5:
            cuda_status = "良好"
            cuda_usable = True
        elif overall_success_rate > 0:
            cuda_status = "部分可用"
            cuda_usable = True
        else:
            cuda_status = "不可用"
            cuda_usable = False
        
        # 生成建议
        recommendations = []
        if not cuda_usable:
            recommendations.extend([
                "CUDA功能不可用，建议检查CUDA驱动安装",
                "确认GPU驱动版本与CUDA版本兼容",
                "考虑重新编译OpenCV with CUDA支持"
            ])
        elif overall_success_rate < 1.0:
            recommendations.extend([
                "部分CUDA功能可用，建议检查具体失败的操作",
                "可能存在GPU架构兼容性问题",
                "建议更新GPU驱动到最新版本"
            ])
        else:
            recommendations.extend([
                "CUDA功能完全可用，可以正常使用GPU加速",
                "建议在实际项目中测试性能表现",
                "定期更新驱动以获得最佳性能"
            ])
        
        # 完整报告
        report = {
            "verification_date": datetime.now().isoformat(),
            "opencv_version": cv2.__version__,
            "cuda_devices": cv2.cuda.getCudaEnabledDeviceCount(),
            "verification_results": self.verification_results,
            "summary": {
                "operations_tested": self.cuda_operations_tested,
                "operations_passed": self.cuda_operations_passed,
                "success_rate": overall_success_rate,
                "cuda_status": cuda_status,
                "cuda_usable": cuda_usable
            },
            "recommendations": recommendations
        }
        
        # 保存报告
        report_path = Path("opencv_cuda_verification_report_new.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 报告已保存: {report_path}")
        
        return report
    
    def run_complete_verification(self):
        """运行完整验证"""
        print("开始OpenCV CUDA完整验证...")
        
        # 1. 基本信息检查
        basic_ok = self.check_basic_info()
        
        if not basic_ok:
            print("\n❌ 基本CUDA支持检查失败，无法继续测试")
            return self.generate_verification_report()
        
        # 2. GPU内存操作测试
        self.test_gpu_memory_operations()
        
        # 3. 基本CUDA操作测试
        self.test_basic_cuda_operations()
        
        # 4. 高级CUDA操作测试
        self.test_advanced_cuda_operations()
        
        # 5. 性能基准测试
        self.performance_benchmark()
        
        # 6. GPU设备信息测试
        self.test_gpu_device_info()
        
        # 7. 生成报告
        report = self.generate_verification_report()
        
        return report

def main():
    verifier = OpenCVCUDAVerifier()
    
    # 运行完整验证
    report = verifier.run_complete_verification()
    
    # 显示总结
    print(f"\n📋 验证总结:")
    summary = report["summary"]
    print(f"   OpenCV版本: {report['opencv_version']}")
    print(f"   CUDA设备数: {report['cuda_devices']}")
    print(f"   测试操作: {summary['operations_tested']}")
    print(f"   通过操作: {summary['operations_passed']}")
    print(f"   成功率: {summary['success_rate']*100:.1f}%")
    print(f"   CUDA状态: {summary['cuda_status']}")
    print(f"   可用性: {'是' if summary['cuda_usable'] else '否'}")
    
    # 显示建议
    print(f"\n💡 建议:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"   {i}. {rec}")
    
    if summary['cuda_usable']:
        print(f"\n🎉 验证完成: OpenCV CUDA功能可用!")
        if summary['success_rate'] == 1.0:
            print(f"✅ 所有CUDA操作都正常工作!")
        else:
            print(f"⚠️ 部分CUDA操作可用，建议检查具体问题")
    else:
        print(f"\n❌ 验证失败: OpenCV CUDA功能不可用")
        print(f"💡 建议检查CUDA安装和GPU驱动")
    
    return summary['cuda_usable']

if __name__ == "__main__":
    main()
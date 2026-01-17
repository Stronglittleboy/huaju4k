#!/usr/bin/env python3
"""
全面的CUDA-OpenCV任务验证脚本
验证当前任务所需的所有CUDA功能是否可用
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime
import traceback
import sys
import os

class CUDATaskValidator:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "opencv_version": cv2.__version__,
            "cuda_available": False,
            "cuda_devices": 0,
            "task_compatibility": {},
            "performance_tests": {},
            "recommendations": []
        }
        
    def check_cuda_availability(self):
        """检查CUDA基本可用性"""
        try:
            self.results["cuda_available"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
            self.results["cuda_devices"] = cv2.cuda.getCudaEnabledDeviceCount()
            
            if self.results["cuda_available"]:
                print(f"✅ CUDA可用，检测到 {self.results['cuda_devices']} 个设备")
                return True
            else:
                print("❌ CUDA不可用")
                return False
        except Exception as e:
            print(f"❌ CUDA检查失败: {e}")
            self.results["cuda_available"] = False
            return False
    
    def test_video_processing_functions(self):
        """测试视频处理相关的CUDA功能"""
        print("\n=== 测试视频处理CUDA功能 ===")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        functions_to_test = {
            "resize": self._test_resize,
            "cvtColor": self._test_color_conversion,
            "bilateralFilter": self._test_bilateral_filter,
            "threshold": self._test_threshold,
            "blur": self._test_blur,
            "GaussianBlur": self._test_gaussian_blur,
            "Canny": self._test_canny,
            "morphologyEx": self._test_morphology
        }
        
        task_results = {}
        
        for func_name, test_func in functions_to_test.items():
            try:
                success, performance = test_func(test_image)
                task_results[func_name] = {
                    "available": success,
                    "performance_ms": performance if success else None
                }
                status = "✅" if success else "❌"
                perf_info = f" ({performance:.2f}ms)" if success and performance else ""
                print(f"{status} {func_name}{perf_info}")
            except Exception as e:
                task_results[func_name] = {
                    "available": False,
                    "error": str(e)
                }
                print(f"❌ {func_name}: {e}")
        
        self.results["task_compatibility"]["video_processing"] = task_results
        return task_results
    
    def _test_resize(self, image):
        """测试图像缩放功能"""
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            start_time = time.time()
            gpu_resized = cv2.cuda.resize(gpu_img, (3840, 2160))
            cv2.cuda.deviceSynchronize()
            end_time = time.time()
            
            result = gpu_resized.download()
            return True, (end_time - start_time) * 1000
        except:
            return False, None
    
    def _test_color_conversion(self, image):
        """测试颜色空间转换"""
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            start_time = time.time()
            gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
            cv2.cuda.deviceSynchronize()
            end_time = time.time()
            
            result = gpu_gray.download()
            return True, (end_time - start_time) * 1000
        except:
            return False, None
    
    def _test_bilateral_filter(self, image):
        """测试双边滤波"""
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            start_time = time.time()
            gpu_filtered = cv2.cuda.bilateralFilter(gpu_img, -1, 50, 50)
            cv2.cuda.deviceSynchronize()
            end_time = time.time()
            
            result = gpu_filtered.download()
            return True, (end_time - start_time) * 1000
        except:
            return False, None
    
    def _test_threshold(self, image):
        """测试阈值处理"""
        try:
            gpu_img = cv2.cuda_GpuMat()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gpu_img.upload(gray_image)
            
            start_time = time.time()
            gpu_thresh = cv2.cuda.threshold(gpu_img, 127, 255, cv2.THRESH_BINARY)[1]
            cv2.cuda.deviceSynchronize()
            end_time = time.time()
            
            result = gpu_thresh.download()
            return True, (end_time - start_time) * 1000
        except:
            return False, None
    
    def _test_blur(self, image):
        """测试模糊滤波"""
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            start_time = time.time()
            gpu_blurred = cv2.cuda.blur(gpu_img, (15, 15))
            cv2.cuda.deviceSynchronize()
            end_time = time.time()
            
            result = gpu_blurred.download()
            return True, (end_time - start_time) * 1000
        except:
            return False, None
    
    def _test_gaussian_blur(self, image):
        """测试高斯模糊"""
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            
            start_time = time.time()
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_img, (15, 15), 0)
            cv2.cuda.deviceSynchronize()
            end_time = time.time()
            
            result = gpu_blurred.download()
            return True, (end_time - start_time) * 1000
        except:
            return False, None
    
    def _test_canny(self, image):
        """测试Canny边缘检测"""
        try:
            gpu_img = cv2.cuda_GpuMat()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gpu_img.upload(gray_image)
            
            start_time = time.time()
            gpu_edges = cv2.cuda.Canny(gpu_img, 100, 200)
            cv2.cuda.deviceSynchronize()
            end_time = time.time()
            
            result = gpu_edges.download()
            return True, (end_time - start_time) * 1000
        except:
            return False, None
    
    def _test_morphology(self, image):
        """测试形态学操作"""
        try:
            gpu_img = cv2.cuda_GpuMat()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gpu_img.upload(gray_image)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            
            start_time = time.time()
            gpu_morph = cv2.cuda.morphologyEx(gpu_img, cv2.MORPH_OPEN, kernel)
            cv2.cuda.deviceSynchronize()
            end_time = time.time()
            
            result = gpu_morph.download()
            return True, (end_time - start_time) * 1000
        except:
            return False, None
    
    def test_memory_operations(self):
        """测试GPU内存操作"""
        print("\n=== 测试GPU内存操作 ===")
        
        memory_tests = {}
        
        try:
            # 测试大图像内存分配
            large_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
            gpu_mat = cv2.cuda_GpuMat()
            
            start_time = time.time()
            gpu_mat.upload(large_image)
            upload_time = time.time() - start_time
            
            start_time = time.time()
            result = gpu_mat.download()
            download_time = time.time() - start_time
            
            memory_tests["large_image_transfer"] = {
                "success": True,
                "upload_time_ms": upload_time * 1000,
                "download_time_ms": download_time * 1000,
                "image_size": "4K (3840x2160)"
            }
            print(f"✅ 4K图像传输: 上传 {upload_time*1000:.2f}ms, 下载 {download_time*1000:.2f}ms")
            
        except Exception as e:
            memory_tests["large_image_transfer"] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ 4K图像传输失败: {e}")
        
        self.results["task_compatibility"]["memory_operations"] = memory_tests
        return memory_tests
    
    def test_ai_upscaling_compatibility(self):
        """测试AI放大相关的CUDA功能"""
        print("\n=== 测试AI放大兼容性 ===")
        
        ai_tests = {}
        
        # 测试多尺度处理
        test_sizes = [(480, 640), (720, 1280), (1080, 1920)]
        
        for height, width in test_sizes:
            try:
                test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(test_image)
                
                # 模拟AI放大预处理
                start_time = time.time()
                gpu_resized = cv2.cuda.resize(gpu_img, (width*2, height*2))
                gpu_filtered = cv2.cuda.bilateralFilter(gpu_resized, -1, 25, 25)
                result = gpu_filtered.download()
                end_time = time.time()
                
                ai_tests[f"{width}x{height}"] = {
                    "success": True,
                    "processing_time_ms": (end_time - start_time) * 1000
                }
                print(f"✅ {width}x{height} 预处理: {(end_time - start_time)*1000:.2f}ms")
                
            except Exception as e:
                ai_tests[f"{width}x{height}"] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"❌ {width}x{height} 预处理失败: {e}")
        
        self.results["task_compatibility"]["ai_upscaling"] = ai_tests
        return ai_tests
    
    def generate_task_recommendations(self):
        """生成任务建议"""
        print("\n=== 生成任务建议 ===")
        
        recommendations = []
        
        # 检查基本CUDA支持
        if not self.results["cuda_available"]:
            recommendations.append({
                "priority": "HIGH",
                "category": "CUDA支持",
                "message": "CUDA不可用，建议使用CPU处理或修复CUDA安装"
            })
            return recommendations
        
        # 检查关键功能
        video_funcs = self.results.get("task_compatibility", {}).get("video_processing", {})
        critical_functions = ["resize", "cvtColor", "bilateralFilter"]
        
        available_critical = sum(1 for func in critical_functions if video_funcs.get(func, {}).get("available", False))
        
        if available_critical == len(critical_functions):
            recommendations.append({
                "priority": "INFO",
                "category": "核心功能",
                "message": "所有关键CUDA功能可用，可以进行GPU加速处理"
            })
        elif available_critical > 0:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "核心功能",
                "message": f"部分关键功能可用 ({available_critical}/{len(critical_functions)})，建议混合CPU/GPU处理"
            })
        else:
            recommendations.append({
                "priority": "HIGH",
                "category": "核心功能",
                "message": "关键CUDA功能不可用，建议使用CPU处理"
            })
        
        # 检查内存操作
        memory_ops = self.results.get("task_compatibility", {}).get("memory_operations", {})
        if memory_ops.get("large_image_transfer", {}).get("success", False):
            recommendations.append({
                "priority": "INFO",
                "category": "内存操作",
                "message": "4K图像GPU内存传输正常，支持大尺寸处理"
            })
        else:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "内存操作",
                "message": "4K图像GPU传输可能有问题，建议使用分块处理"
            })
        
        # 性能建议
        if video_funcs.get("resize", {}).get("performance_ms"):
            resize_time = video_funcs["resize"]["performance_ms"]
            if resize_time < 50:
                recommendations.append({
                    "priority": "INFO",
                    "category": "性能",
                    "message": f"GPU缩放性能良好 ({resize_time:.1f}ms)，适合实时处理"
                })
            else:
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "性能",
                    "message": f"GPU缩放性能一般 ({resize_time:.1f}ms)，建议优化参数"
                })
        
        self.results["recommendations"] = recommendations
        
        for rec in recommendations:
            priority_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "INFO": "🟢"}.get(rec["priority"], "ℹ️")
            print(f"{priority_icon} [{rec['category']}] {rec['message']}")
        
        return recommendations
    
    def run_comprehensive_validation(self):
        """运行全面验证"""
        print("🚀 开始全面CUDA-OpenCV任务验证")
        print(f"OpenCV版本: {cv2.__version__}")
        print("=" * 60)
        
        # 基本CUDA检查
        if not self.check_cuda_availability():
            self.generate_task_recommendations()
            return self.results
        
        # 功能测试
        self.test_video_processing_functions()
        self.test_memory_operations()
        self.test_ai_upscaling_compatibility()
        
        # 生成建议
        self.generate_task_recommendations()
        
        # 保存结果
        with open("comprehensive_cuda_validation_report.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("✅ 验证完成，报告已保存到 comprehensive_cuda_validation_report.json")
        
        return self.results

def main():
    validator = CUDATaskValidator()
    results = validator.run_comprehensive_validation()
    
    # 输出总结
    print("\n📊 验证总结:")
    if results["cuda_available"]:
        video_funcs = results.get("task_compatibility", {}).get("video_processing", {})
        available_count = sum(1 for func_data in video_funcs.values() if func_data.get("available", False))
        total_count = len(video_funcs)
        print(f"   CUDA功能可用率: {available_count}/{total_count} ({available_count/total_count*100:.1f}%)")
        
        high_priority_issues = sum(1 for rec in results.get("recommendations", []) if rec["priority"] == "HIGH")
        if high_priority_issues == 0:
            print("   ✅ 可以继续当前任务的CUDA加速处理")
        else:
            print(f"   ⚠️  发现 {high_priority_issues} 个高优先级问题，建议先解决")
    else:
        print("   ❌ CUDA不可用，建议使用CPU处理")

if __name__ == "__main__":
    main()
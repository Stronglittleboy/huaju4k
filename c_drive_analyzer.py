#!/usr/bin/env python3
"""
Windows C盘分析和清理建议工具
C Drive Analysis and Cleanup Recommendations Tool
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class CDriveAnalyzer:
    def __init__(self):
        self.c_drive = Path("C:/")
        self.analysis_results = {}
        self.cleanup_recommendations = []
        self.movable_items = []
        self.deletable_items = []
        
        print("🔍 Windows C盘分析工具")
        print("=" * 50)
    
    def get_disk_usage(self) -> Dict:
        """获取C盘使用情况"""
        try:
            total, used, free = shutil.disk_usage("C:/")
            
            usage_info = {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "usage_percent": (used / total) * 100
            }
            
            print(f"💾 C盘使用情况:")
            print(f"   总容量: {usage_info['total_gb']:.1f} GB")
            print(f"   已使用: {usage_info['used_gb']:.1f} GB")
            print(f"   可用空间: {usage_info['free_gb']:.1f} GB")
            print(f"   使用率: {usage_info['usage_percent']:.1f}%")
            
            return usage_info
            
        except Exception as e:
            print(f"❌ 获取磁盘使用情况失败: {e}")
            return {}
    
    def analyze_system_folders(self) -> Dict:
        """分析系统文件夹"""
        print(f"\n📁 分析系统文件夹...")
        
        system_folders = {
            "Windows": "C:/Windows",
            "Program Files": "C:/Program Files",
            "Program Files (x86)": "C:/Program Files (x86)",
            "Users": "C:/Users",
            "ProgramData": "C:/ProgramData",
            "Temp": "C:/Windows/Temp",
            "System32": "C:/Windows/System32"
        }
        
        folder_analysis = {}
        
        for folder_name, folder_path in system_folders.items():
            try:
                path = Path(folder_path)
                if path.exists():
                    size = self._get_folder_size(path)
                    folder_analysis[folder_name] = {
                        "path": folder_path,
                        "size_gb": size / (1024**3),
                        "exists": True
                    }
                    print(f"   {folder_name}: {size / (1024**3):.1f} GB")
                else:
                    folder_analysis[folder_name] = {
                        "path": folder_path,
                        "size_gb": 0,
                        "exists": False
                    }
            except Exception as e:
                print(f"   ❌ {folder_name}: 无法访问 - {e}")
                folder_analysis[folder_name] = {
                    "path": folder_path,
                    "error": str(e),
                    "exists": False
                }
        
        return folder_analysis
    
    def _get_folder_size(self, folder_path: Path) -> int:
        """获取文件夹大小"""
        total_size = 0
        try:
            for item in folder_path.rglob("*"):
                if item.is_file():
                    try:
                        total_size += item.stat().st_size
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError):
            pass
        return total_size
    
    def analyze_temp_files(self) -> Dict:
        """分析临时文件"""
        print(f"\n🗑️ 分析临时文件...")
        
        temp_locations = {
            "Windows Temp": "C:/Windows/Temp",
            "User Temp": os.path.expandvars("%TEMP%"),
            "Prefetch": "C:/Windows/Prefetch",
            "Recent": os.path.expandvars("%APPDATA%/Microsoft/Windows/Recent"),
            "Recycle Bin": "C:/$Recycle.Bin"
        }
        
        temp_analysis = {}
        total_temp_size = 0
        
        for location_name, location_path in temp_locations.items():
            try:
                path = Path(location_path)
                if path.exists():
                    size = self._get_folder_size(path)
                    file_count = len(list(path.rglob("*")))
                    
                    temp_analysis[location_name] = {
                        "path": location_path,
                        "size_gb": size / (1024**3),
                        "file_count": file_count,
                        "cleanable": True
                    }
                    
                    total_temp_size += size
                    print(f"   {location_name}: {size / (1024**3):.2f} GB ({file_count} 文件)")
                    
                    # 添加到可清理项目
                    if size > 100 * 1024 * 1024:  # 大于100MB
                        self.deletable_items.append({
                            "type": "临时文件",
                            "location": location_name,
                            "path": location_path,
                            "size_gb": size / (1024**3),
                            "description": f"{location_name}临时文件",
                            "safety": "安全"
                        })
                        
            except Exception as e:
                print(f"   ❌ {location_name}: 无法访问 - {e}")
        
        temp_analysis["total_temp_size_gb"] = total_temp_size / (1024**3)
        print(f"   总临时文件: {total_temp_size / (1024**3):.2f} GB")
        
        return temp_analysis
    
    def analyze_user_folders(self) -> Dict:
        """分析用户文件夹"""
        print(f"\n👤 分析用户文件夹...")
        
        user_folders = {
            "Desktop": os.path.expandvars("%USERPROFILE%/Desktop"),
            "Documents": os.path.expandvars("%USERPROFILE%/Documents"),
            "Downloads": os.path.expandvars("%USERPROFILE%/Downloads"),
            "Pictures": os.path.expandvars("%USERPROFILE%/Pictures"),
            "Videos": os.path.expandvars("%USERPROFILE%/Videos"),
            "Music": os.path.expandvars("%USERPROFILE%/Music"),
            "AppData": os.path.expandvars("%APPDATA%")
        }
        
        user_analysis = {}
        
        for folder_name, folder_path in user_folders.items():
            try:
                path = Path(folder_path)
                if path.exists():
                    size = self._get_folder_size(path)
                    user_analysis[folder_name] = {
                        "path": folder_path,
                        "size_gb": size / (1024**3),
                        "movable": folder_name in ["Documents", "Pictures", "Videos", "Music", "Downloads"]
                    }
                    print(f"   {folder_name}: {size / (1024**3):.1f} GB")
                    
                    # 添加到可移动项目
                    if folder_name in ["Documents", "Pictures", "Videos", "Music"] and size > 1024**3:  # 大于1GB
                        self.movable_items.append({
                            "type": "用户文件夹",
                            "folder": folder_name,
                            "path": folder_path,
                            "size_gb": size / (1024**3),
                            "description": f"{folder_name}文件夹可移动到其他盘",
                            "method": "修改注册表或使用符号链接"
                        })
                        
            except Exception as e:
                print(f"   ❌ {folder_name}: 无法访问 - {e}")
        
        return user_analysis
    
    def analyze_installed_programs(self) -> Dict:
        """分析已安装程序"""
        print(f"\n💿 分析已安装程序...")
        
        program_folders = [
            "C:/Program Files",
            "C:/Program Files (x86)"
        ]
        
        program_analysis = {}
        large_programs = []
        
        for program_folder in program_folders:
            try:
                path = Path(program_folder)
                if path.exists():
                    programs = []
                    for item in path.iterdir():
                        if item.is_dir():
                            try:
                                size = self._get_folder_size(item)
                                if size > 500 * 1024 * 1024:  # 大于500MB
                                    programs.append({
                                        "name": item.name,
                                        "path": str(item),
                                        "size_gb": size / (1024**3)
                                    })
                                    large_programs.append({
                                        "name": item.name,
                                        "size_gb": size / (1024**3),
                                        "path": str(item)
                                    })
                            except:
                                continue
                    
                    program_analysis[program_folder] = {
                        "programs": programs,
                        "count": len(programs)
                    }
                    
            except Exception as e:
                print(f"   ❌ {program_folder}: 无法访问 - {e}")
        
        # 显示大型程序
        large_programs.sort(key=lambda x: x["size_gb"], reverse=True)
        print(f"   发现 {len(large_programs)} 个大型程序 (>500MB):")
        for i, program in enumerate(large_programs[:10]):  # 显示前10个
            print(f"     {i+1}. {program['name']}: {program['size_gb']:.1f} GB")
        
        return program_analysis
    
    def analyze_browser_data(self) -> Dict:
        """分析浏览器数据"""
        print(f"\n🌐 分析浏览器数据...")
        
        browser_paths = {
            "Chrome": os.path.expandvars("%LOCALAPPDATA%/Google/Chrome/User Data"),
            "Edge": os.path.expandvars("%LOCALAPPDATA%/Microsoft/Edge/User Data"),
            "Firefox": os.path.expandvars("%APPDATA%/Mozilla/Firefox"),
            "Opera": os.path.expandvars("%APPDATA%/Opera Software")
        }
        
        browser_analysis = {}
        
        for browser_name, browser_path in browser_paths.items():
            try:
                path = Path(browser_path)
                if path.exists():
                    size = self._get_folder_size(path)
                    browser_analysis[browser_name] = {
                        "path": browser_path,
                        "size_gb": size / (1024**3),
                        "exists": True
                    }
                    print(f"   {browser_name}: {size / (1024**3):.2f} GB")
                    
                    # 如果浏览器数据较大，添加清理建议
                    if size > 1024**3:  # 大于1GB
                        self.cleanup_recommendations.append({
                            "type": "浏览器数据",
                            "browser": browser_name,
                            "size_gb": size / (1024**3),
                            "action": "清理缓存、历史记录、下载记录",
                            "safety": "中等风险"
                        })
                        
            except Exception as e:
                print(f"   ❌ {browser_name}: 无法访问 - {e}")
        
        return browser_analysis
    
    def generate_cleanup_recommendations(self) -> List[Dict]:
        """生成清理建议"""
        print(f"\n💡 生成清理建议...")
        
        # 系统清理建议
        system_cleanup = [
            {
                "category": "系统清理",
                "action": "运行磁盘清理工具",
                "command": "cleanmgr /c C:",
                "description": "清理系统文件、回收站、临时文件",
                "safety": "安全",
                "potential_space_gb": "1-5"
            },
            {
                "category": "系统清理",
                "action": "清理Windows更新文件",
                "command": "dism /online /cleanup-image /startcomponentcleanup",
                "description": "清理Windows组件存储",
                "safety": "安全",
                "potential_space_gb": "2-10"
            },
            {
                "category": "系统清理",
                "action": "清理系统还原点",
                "description": "保留最新的还原点，删除旧的",
                "safety": "中等风险",
                "potential_space_gb": "5-20"
            }
        ]
        
        self.cleanup_recommendations.extend(system_cleanup)
        
        return self.cleanup_recommendations
    
    def generate_move_recommendations(self) -> List[Dict]:
        """生成移动建议"""
        print(f"\n📦 生成移动建议...")
        
        # 系统移动建议
        system_moves = [
            {
                "category": "虚拟内存",
                "item": "页面文件 (pagefile.sys)",
                "current_location": "C:/",
                "recommended_location": "其他盘符 (如D:/)",
                "method": "系统属性 > 高级 > 性能设置 > 高级 > 虚拟内存",
                "potential_space_gb": "4-16",
                "difficulty": "简单"
            },
            {
                "category": "休眠文件",
                "item": "休眠文件 (hiberfil.sys)",
                "current_location": "C:/",
                "action": "禁用休眠功能",
                "command": "powercfg /hibernate off",
                "potential_space_gb": "4-32",
                "difficulty": "简单"
            },
            {
                "category": "程序安装",
                "item": "新程序默认安装位置",
                "current_location": "C:/Program Files",
                "recommended_location": "其他盘符",
                "method": "修改注册表或安装时选择路径",
                "difficulty": "中等"
            }
        ]
        
        self.movable_items.extend(system_moves)
        
        return self.movable_items
    
    def create_analysis_report(self) -> Dict:
        """创建分析报告"""
        print(f"\n📊 创建分析报告...")
        
        report = {
            "analysis_date": datetime.now().isoformat(),
            "disk_usage": self.analysis_results.get("disk_usage", {}),
            "system_folders": self.analysis_results.get("system_folders", {}),
            "temp_files": self.analysis_results.get("temp_files", {}),
            "user_folders": self.analysis_results.get("user_folders", {}),
            "installed_programs": self.analysis_results.get("installed_programs", {}),
            "browser_data": self.analysis_results.get("browser_data", {}),
            "cleanup_recommendations": self.cleanup_recommendations,
            "movable_items": self.movable_items,
            "deletable_items": self.deletable_items,
            "summary": self._generate_summary()
        }
        
        # 保存报告
        report_path = Path("c_drive_analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 报告已保存: {report_path}")
        
        return report
    
    def _generate_summary(self) -> Dict:
        """生成总结"""
        disk_usage = self.analysis_results.get("disk_usage", {})
        
        # 计算潜在可释放空间
        potential_cleanup_space = sum(
            item.get("size_gb", 0) for item in self.deletable_items
        )
        
        potential_move_space = sum(
            item.get("size_gb", 0) for item in self.movable_items 
            if isinstance(item.get("size_gb"), (int, float))
        )
        
        return {
            "current_usage_percent": disk_usage.get("usage_percent", 0),
            "free_space_gb": disk_usage.get("free_gb", 0),
            "potential_cleanup_space_gb": potential_cleanup_space,
            "potential_move_space_gb": potential_move_space,
            "total_recommendations": len(self.cleanup_recommendations),
            "total_movable_items": len(self.movable_items),
            "total_deletable_items": len(self.deletable_items),
            "urgency": self._assess_urgency(disk_usage.get("usage_percent", 0))
        }
    
    def _assess_urgency(self, usage_percent: float) -> str:
        """评估紧急程度"""
        if usage_percent > 90:
            return "紧急"
        elif usage_percent > 80:
            return "高"
        elif usage_percent > 70:
            return "中等"
        else:
            return "低"
    
    def run_complete_analysis(self) -> Dict:
        """运行完整分析"""
        print("开始C盘完整分析...")
        
        # 1. 磁盘使用情况
        self.analysis_results["disk_usage"] = self.get_disk_usage()
        
        # 2. 系统文件夹分析
        self.analysis_results["system_folders"] = self.analyze_system_folders()
        
        # 3. 临时文件分析
        self.analysis_results["temp_files"] = self.analyze_temp_files()
        
        # 4. 用户文件夹分析
        self.analysis_results["user_folders"] = self.analyze_user_folders()
        
        # 5. 已安装程序分析
        self.analysis_results["installed_programs"] = self.analyze_installed_programs()
        
        # 6. 浏览器数据分析
        self.analysis_results["browser_data"] = self.analyze_browser_data()
        
        # 7. 生成建议
        self.generate_cleanup_recommendations()
        self.generate_move_recommendations()
        
        # 8. 创建报告
        report = self.create_analysis_report()
        
        return report

def main():
    analyzer = CDriveAnalyzer()
    
    try:
        # 运行完整分析
        report = analyzer.run_complete_analysis()
        
        # 显示总结
        summary = report["summary"]
        print(f"\n📋 分析总结:")
        print(f"   当前使用率: {summary['current_usage_percent']:.1f}%")
        print(f"   可用空间: {summary['free_space_gb']:.1f} GB")
        print(f"   可清理空间: {summary['potential_cleanup_space_gb']:.1f} GB")
        print(f"   可移动空间: {summary['potential_move_space_gb']:.1f} GB")
        print(f"   紧急程度: {summary['urgency']}")
        print(f"   清理建议: {summary['total_recommendations']} 项")
        print(f"   可移动项: {summary['total_movable_items']} 项")
        print(f"   可删除项: {summary['total_deletable_items']} 项")
        
        print(f"\n🎯 主要建议:")
        
        # 显示前5个清理建议
        print(f"\n🧹 清理建议:")
        for i, rec in enumerate(analyzer.cleanup_recommendations[:5], 1):
            print(f"   {i}. {rec.get('action', rec.get('description', '未知'))}")
            if 'potential_space_gb' in rec:
                print(f"      可释放: {rec['potential_space_gb']} GB")
        
        # 显示前5个移动建议
        print(f"\n📦 移动建议:")
        for i, item in enumerate(analyzer.movable_items[:5], 1):
            print(f"   {i}. {item.get('item', item.get('folder', '未知'))}")
            if 'size_gb' in item and isinstance(item['size_gb'], (int, float)):
                print(f"      大小: {item['size_gb']:.1f} GB")
        
        print(f"\n✅ 分析完成！详细报告已保存到 c_drive_analysis_report.json")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
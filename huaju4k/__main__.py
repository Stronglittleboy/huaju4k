#!/usr/bin/env python3
"""
huaju4k主入口点

这个模块提供了huaju4k命令行工具的主入口点，
支持通过 `python -m huaju4k` 或直接调用 `huaju4k` 命令运行。

实现任务12.1的要求：
- 创建主CLI应用程序入口点
- 支持模块化调用
- 提供错误处理和用户友好的消息
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """主入口函数"""
    try:
        # 检查Python版本
        if sys.version_info < (3, 8):
            print("错误: huaju4k需要Python 3.8或更高版本")
            print(f"当前Python版本: {sys.version}")
            sys.exit(1)
        
        # 导入并运行CLI
        from huaju4k.cli.main import cli
        cli()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖已正确安装:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(130)
    except Exception as e:
        print(f"未预期的错误: {e}")
        print("请使用 --verbose 选项获取详细错误信息")
        sys.exit(1)

if __name__ == '__main__':
    main()
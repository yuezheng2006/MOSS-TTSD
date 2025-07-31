#!/usr/bin/env python3
"""
MOSS-TTSD 环境健康检查工具
检查系统环境、依赖包、模型文件等的完整性
"""
import sys
import os
import importlib
from pathlib import Path

def colored_print(text, color="reset"):
    """打印彩色文本"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m", 
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['reset'])}{text}{colors['reset']}")

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version >= (3, 8):
        return True, f"Python版本: {version_str} ✅"
    else:
        return False, f"Python版本过低: {version_str} (需要3.8+) ❌"

def check_package(package_name, required=True):
    """检查Python包是否安装"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        status = "✅" if required else "🔶"
        return True, f"{package_name}: {version} {status}"
    except ImportError:
        status = "❌" if required else "🔶"
        suffix = " (必需)" if required else " (可选)"
        return not required, f"{package_name}: 未安装{suffix} {status}"

def check_file_exists(file_path, description="文件"):
    """检查文件是否存在"""
    path = Path(file_path)
    if path.exists():
        if path.is_file():
            size = path.stat().st_size
            if size > 1024**3:  # > 1GB
                size_str = f"{size / (1024**3):.1f}GB"
            elif size > 1024**2:  # > 1MB
                size_str = f"{size / (1024**2):.1f}MB"
            else:
                size_str = f"{size / 1024:.1f}KB"
            return True, f"{description}: {file_path} ({size_str}) ✅"
        else:
            return True, f"{description}: {file_path} (目录) ✅"
    else:
        return False, f"{description}: {file_path} 不存在 ❌"

def check_cuda_availability():
    """检查CUDA可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"CUDA: 可用 ({device_count}个设备, {device_name}) ✅"
        else:
            return True, f"CUDA: 不可用，将使用CPU模式 🔶"
    except ImportError:
        return False, f"CUDA: 无法检查 (PyTorch未安装) ❌"

def check_model_compatibility():
    """检查模型兼容性"""
    try:
        # 检查XY_Tokenizer配置
        config_path = "XY_Tokenizer/config/xy_tokenizer_config.yaml"
        if os.path.exists(config_path):
            config_status = "✅"
        else:
            config_status = "⚠️"
        
        # 检查模型权重
        weight_path = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
        if os.path.exists(weight_path):
            weight_status = "✅"
        else:
            weight_status = "❌"
        
        return True, f"XY_Tokenizer配置: {config_status}, 权重: {weight_status}"
        
    except Exception as e:
        return False, f"模型兼容性检查失败: {str(e)} ❌"

def check_audio_backends():
    """检查音频后端"""
    backends = []
    
    # 检查soundfile
    try:
        import soundfile
        backends.append("soundfile ✅")
    except ImportError:
        backends.append("soundfile ❌")
    
    # 检查librosa
    try:
        import librosa
        backends.append("librosa ✅")
    except ImportError:
        backends.append("librosa ❌")
    
    # 检查torchaudio
    try:
        import torchaudio
        backends.append("torchaudio ✅")
    except ImportError:
        backends.append("torchaudio ❌")
    
    return True, f"音频后端: {', '.join(backends)}"

def run_functionality_test():
    """运行功能测试"""
    tests = []
    
    # 测试1: 基础导入
    try:
        from generation_utils import load_model
        tests.append("generation_utils导入 ✅")
    except ImportError as e:
        tests.append(f"generation_utils导入失败: {str(e)[:50]}... ❌")
    
    # 测试2: 模型配置检查
    try:
        if os.path.exists("XY_Tokenizer/config/xy_tokenizer_config.yaml"):
            tests.append("XY_Tokenizer配置文件 ✅")
        else:
            tests.append("XY_Tokenizer配置文件缺失 ❌")
    except Exception as e:
        tests.append(f"配置检查失败: {str(e)} ❌")
    
    # 测试3: 示例文件
    example_files = [
        "examples/examples.jsonl",
        "examples/zh_spk1_moon.wav", 
        "examples/zh_spk2_moon.wav"
    ]
    
    missing_examples = [f for f in example_files if not os.path.exists(f)]
    if not missing_examples:
        tests.append("示例文件完整 ✅")
    else:
        tests.append(f"缺失示例文件: {len(missing_examples)}个 ⚠️")
    
    return True, f"功能测试: {'; '.join(tests)}"

def main():
    """主检查函数"""
    colored_print("🔍 MOSS-TTSD 环境健康检查", "cyan")
    colored_print("=" * 60, "white")
    
    checks = []
    all_passed = True
    
    # 1. Python版本检查
    passed, msg = check_python_version()
    checks.append((passed, msg))
    if not passed:
        all_passed = False
    
    # 2. 核心依赖检查
    core_packages = [
        "torch", "torchaudio", "transformers", "gradio",
        "numpy", "accelerate", "soundfile", "librosa",
        "huggingface_hub", "tqdm", "requests"
    ]
    
    colored_print("\n📦 核心依赖检查:", "blue")
    for package in core_packages:
        passed, msg = check_package(package, required=True)
        checks.append((passed, msg))
        if not passed:
            all_passed = False
        print(f"  {msg}")
    
    # 3. 可选依赖检查
    optional_packages = [
        "liger_kernel", "flash_attn", "openai", "PyPDF2",
        "beautifulsoup4", "PyYAML", "einops", "pydub"
    ]
    
    colored_print("\n🔧 可选依赖检查:", "blue")
    for package in optional_packages:
        passed, msg = check_package(package, required=False)
        checks.append((passed, msg))
        print(f"  {msg}")
    
    # 4. 系统文件检查
    colored_print("\n📁 文件系统检查:", "blue")
    
    file_checks = [
        ("XY_Tokenizer/config/xy_tokenizer_config.yaml", "XY_Tokenizer配置"),
        ("XY_Tokenizer/weights/xy_tokenizer.ckpt", "模型权重"),
        ("examples/examples.jsonl", "示例JSONL"),
        ("examples/zh_spk1_moon.wav", "示例音频1"),
        ("examples/zh_spk2_moon.wav", "示例音频2"),
        ("requirements.txt", "依赖清单"),
        ("generation_utils.py", "生成工具"),
        ("modeling_asteroid.py", "模型定义")
    ]
    
    for file_path, description in file_checks:
        passed, msg = check_file_exists(file_path, description)
        checks.append((passed, msg))
        print(f"  {msg}")
    
    # 5. 硬件检查
    colored_print("\n💻 硬件检查:", "blue")
    passed, msg = check_cuda_availability()
    checks.append((passed, msg))
    print(f"  {msg}")
    
    # 6. 音频后端检查
    colored_print("\n🎵 音频后端检查:", "blue")
    passed, msg = check_audio_backends()
    checks.append((passed, msg))
    print(f"  {msg}")
    
    # 7. 功能测试
    colored_print("\n🧪 功能测试:", "blue")
    passed, msg = run_functionality_test()
    checks.append((passed, msg))
    print(f"  {msg}")
    
    # 统计结果
    passed_count = sum(1 for passed, _ in checks if passed)
    total_count = len(checks)
    
    colored_print("\n" + "=" * 60, "white")
    colored_print("📊 检查结果统计:", "cyan")
    print(f"✅ 通过: {passed_count}")
    print(f"❌ 失败: {total_count - passed_count}")
    print(f"📈 通过率: {passed_count/total_count*100:.1f}%")
    
    # 总结和建议
    colored_print("\n🎯 诊断结果:", "cyan")
    
    if passed_count >= total_count * 0.9:
        colored_print("🎉 环境配置优秀！可以正常使用所有功能。", "green")
        return_code = 0
    elif passed_count >= total_count * 0.8:
        colored_print("✅ 环境配置良好！基本功能可用，某些高级功能可能受限。", "green")
        return_code = 0
    elif passed_count >= total_count * 0.6:
        colored_print("⚠️ 环境配置有问题！部分功能可能无法使用。", "yellow")
        return_code = 1
    else:
        colored_print("❌ 环境配置严重缺陷！建议重新安装。", "red")
        return_code = 2
    
    # 提供建议
    colored_print("\n💡 建议操作:", "cyan")
    
    failed_checks = [msg for passed, msg in checks if not passed]
    if failed_checks:
        print("针对失败项目的解决建议:")
        
        for msg in failed_checks[:5]:  # 只显示前5个
            if "未安装" in msg:
                package = msg.split(":")[0]
                print(f"  • 安装缺失包: pip install {package}")
            elif "不存在" in msg and "模型权重" in msg:
                print(f"  • 下载模型权重: huggingface-cli download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt --local-dir ./XY_Tokenizer/weights/")
            elif "Python版本过低" in msg:
                print(f"  • 升级Python版本到3.8+")
    
    print("\n更多帮助:")
    print("  📖 查看安装指南: docs/安装配置指南.md")
    print("  🔧 重新安装: ./install.sh")
    print("  🚀 启动界面: ./start_ui.sh")
    
    return return_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
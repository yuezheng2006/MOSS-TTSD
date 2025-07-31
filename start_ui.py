#!/usr/bin/env python3
"""
MOSS-TTSD 播客生成器启动脚本
一键启动简洁美观的Web界面
"""
import os
import sys
import subprocess

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查关键文件
    required_files = [
        "XY_Tokenizer/weights/xy_tokenizer.ckpt",
        "XY_Tokenizer/config/xy_tokenizer_config.yaml",
        "examples/zh_spk1_moon.wav",
        "examples/zh_spk2_moon.wav"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ 以下文件缺失:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 请确保已完成环境设置和模型下载")
        return False
    
    print("✅ 环境检查通过")
    return True

def start_ui():
    """启动用户界面"""
    print("🚀 启动MOSS-TTSD播客生成器...")
    print("=" * 50)
    
    if not check_environment():
        print("❌ 环境检查失败，请先完成环境配置")
        return
    
    try:
        # 启动增强版界面
        print("🌐 正在启动Web界面...")
        print("📍 界面地址: http://localhost:7861")
        print("⏹️  按 Ctrl+C 停止服务")
        print("=" * 50)
        
        subprocess.run([sys.executable, "enhanced_podcast_ui.py"])
        
    except KeyboardInterrupt:
        print("\n👋 用户停止服务")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n💡 备用启动方式:")
        print("   python enhanced_podcast_ui.py")

if __name__ == "__main__":
    start_ui()
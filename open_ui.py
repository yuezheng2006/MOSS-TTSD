#!/usr/bin/env python3
"""
快速打开MOSS-TTSD播客生成界面
"""
import webbrowser
import time
import requests
import sys

def check_service():
    """检查服务状态"""
    try:
        response = requests.get("http://localhost:7861", timeout=3)
        return response.status_code == 200
    except:
        return False

def main():
    print("🔍 检查MOSS-TTSD播客生成器状态...")
    
    if check_service():
        print("✅ 服务正常运行")
        print("🌐 正在打开浏览器...")
        
        # 自动打开浏览器
        webbrowser.open("http://localhost:7861")
        
        print("📍 界面地址: http://localhost:7861")
        print("🎙️ 享受AI播客生成！")
    else:
        print("❌ 服务未运行或无法访问")
        print("💡 请先运行: python enhanced_podcast_ui.py")
        print("   或者使用: python start_ui.py")

if __name__ == "__main__":
    main()
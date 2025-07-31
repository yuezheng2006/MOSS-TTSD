#!/bin/bash
# MOSS-TTSD 快速启动脚本

set -e

echo "🚀 MOSS-TTSD 播客生成器启动脚本"
echo "================================"

# 检查虚拟环境
if [ ! -d "moss_ttsd_env" ]; then
    echo "❌ 虚拟环境不存在！"
    echo "💡 请先运行安装脚本: ./install.sh"
    exit 1
fi

# 激活虚拟环境
echo "🔌 激活虚拟环境..."
source moss_ttsd_env/bin/activate
echo "✅ 虚拟环境已激活"

# 快速健康检查
echo "🔍 快速健康检查..."
if python -c "import torch, transformers, gradio; print('✅ 核心依赖正常')" 2>/dev/null; then
    echo "✅ 基础环境正常"
else
    echo "❌ 基础环境有问题，运行完整检查："
    python health_check.py
    exit 1
fi

# 检查模型文件
if [ ! -f "XY_Tokenizer/weights/xy_tokenizer.ckpt" ]; then
    echo "❌ 模型权重文件不存在！"
    echo "💡 请下载模型权重："
    echo "   huggingface-cli download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt --local-dir ./XY_Tokenizer/weights/"
    exit 1
fi

# 提供启动选项
echo ""
echo "🎯 请选择启动模式:"
echo "1) 🎙️ 基础播客生成器 (简单易用)"
echo "2) 🎛️ 高级播客生成器 (参数控制)"
echo "3) ⌨️ 命令行推理演示"
echo "4) 🔍 完整环境检查"
echo "5) 📖 查看使用指南"
echo ""

# 自动选择模式 (如果有参数)
if [ $# -eq 1 ]; then
    choice=$1
else
    read -p "请选择 (1-5) [默认: 1]: " choice
    choice=${choice:-1}
fi

case $choice in
    1)
        echo "🎙️ 启动基础播客生成器..."
        echo "📍 界面地址: http://localhost:7861"
        echo "💡 在浏览器中打开上述地址即可使用"
        python enhanced_podcast_ui.py
        ;;
    2)
        echo "🎛️ 启动高级播客生成器..."
        echo "📍 界面地址: http://localhost:7862"
        echo "💡 在浏览器中打开上述地址即可使用"
        python advanced_podcast_ui.py
        ;;
    3)
        echo "⌨️ 命令行推理演示..."
        echo "🔄 使用示例文件进行推理..."
        
        # 创建输出目录
        mkdir -p demo_outputs
        
        # 运行推理
        echo "python inference.py --jsonl examples/examples.jsonl --output_dir demo_outputs --seed 42 --use_normalize"
        python inference.py --jsonl examples/examples.jsonl --output_dir demo_outputs --seed 42 --use_normalize
        
        # 显示结果
        echo ""
        echo "✅ 推理完成！生成的音频文件："
        ls -la demo_outputs/
        echo ""
        echo "💡 你可以播放生成的.wav文件来听取效果"
        ;;
    4)
        echo "🔍 运行完整环境检查..."
        python health_check.py
        ;;
    5)
        echo "📖 使用指南:"
        echo ""
        echo "🛠️ 安装配置: docs/安装配置指南.md"
        echo "🚀 使用指南: docs/使用指南.md"
        echo "🎛️ 参数详解: docs/生成参数详解.md"
        echo "🎭 场景示例: docs/场景示例.md"
        echo "🛠️ 技术文档: docs/技术文档.md"
        echo ""
        echo "💻 常用命令:"
        echo "  ./install.sh           # 重新安装环境"
        echo "  python health_check.py # 环境健康检查"
        echo "  ./start_ui.sh 1        # 直接启动基础UI"
        echo "  ./start_ui.sh 2        # 直接启动高级UI"
        ;;
    *)
        echo "❌ 无效选择: $choice"
        echo "💡 请选择 1-5 之间的数字"
        exit 1
        ;;
esac
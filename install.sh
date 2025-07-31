#!/bin/bash
# MOSS-TTSD 一键安装脚本
# 适用于macOS和Linux系统

set -e  # 遇到错误立即退出

echo "🚀 MOSS-TTSD 一键安装脚本"
echo "================================"

# 检测操作系统
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    echo "❌ 不支持的操作系统: $OSTYPE"
    exit 1
fi

echo "🔍 检测到操作系统: $OS"

# 检查Python版本
echo "🐍 检查Python环境..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python未安装，请先安装Python 3.8+"
    if [[ "$OS" == "macOS" ]]; then
        echo "💡 建议使用Homebrew安装: brew install python"
    elif [[ "$OS" == "Linux" ]]; then
        echo "💡 建议使用包管理器安装: sudo apt install python3 python3-pip"
    fi
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "✅ 找到Python版本: $PYTHON_VERSION"

# 检查Python版本是否满足要求
if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python版本满足要求"
else
    echo "❌ Python版本过低，需要3.8+，当前版本: $PYTHON_VERSION"
    exit 1
fi

# 检查pip
echo "📦 检查pip..."
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "❌ pip未安装，请安装pip"
    exit 1
fi
echo "✅ pip可用"

# 创建虚拟环境
echo "🏗️ 创建虚拟环境..."
if [ -d "moss_ttsd_env" ]; then
    echo "⚠️ 虚拟环境已存在，跳过创建"
else
    $PYTHON_CMD -m venv moss_ttsd_env
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境
echo "🔌 激活虚拟环境..."
source moss_ttsd_env/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📚 安装Python依赖..."
echo "⏳ 这可能需要几分钟时间..."

# 安装核心依赖
pip install torch torchaudio transformers gradio numpy accelerate

# 安装其他依赖
pip install PyPDF2 beautifulsoup4 soundfile librosa tqdm requests openai PyYAML einops huggingface_hub pydub

# macOS特殊处理 - 条件安装liger_kernel
if [[ "$OS" == "macOS" ]]; then
    echo "🍎 macOS系统检测到，跳过liger_kernel安装（这是正常的）"
else
    echo "🔧 尝试安装liger_kernel..."
    pip install liger_kernel || echo "⚠️ liger_kernel安装失败，将使用标准实现"
fi

# 可选的flash-attention
echo "⚡ 尝试安装flash-attention (可选)..."
pip install flash-attn || echo "⚠️ flash-attention安装失败，将使用其他注意力实现"

echo "✅ 依赖安装完成"

# 创建模型目录
echo "📁 创建模型目录..."
mkdir -p XY_Tokenizer/weights

# 检查huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "📥 安装huggingface-cli..."
    pip install --upgrade huggingface_hub
fi

# 下载模型权重
echo "⬇️ 下载模型权重..."
echo "📊 模型大小约1-2GB，请耐心等待..."

if [ -f "XY_Tokenizer/weights/xy_tokenizer.ckpt" ]; then
    echo "✅ 模型权重已存在，跳过下载"
else
    echo "🔄 开始下载XY_Tokenizer模型权重..."
    
    # 尝试使用镜像加速下载
    export HF_ENDPOINT=https://hf-mirror.com
    
    if huggingface-cli download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt --local-dir ./XY_Tokenizer/weights/ --resume-download; then
        echo "✅ 模型权重下载成功"
    else
        echo "❌ 模型权重下载失败"
        echo "💡 你可以稍后手动下载："
        echo "   huggingface-cli download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt --local-dir ./XY_Tokenizer/weights/"
    fi
fi

# 验证安装
echo "🧪 验证安装..."
cat > test_install.py << 'EOF'
import sys
def test_installation():
    print("🔍 验证MOSS-TTSD安装...")
    
    # 检查核心包
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers未安装")
        return False
    
    try:
        import gradio
        print(f"✅ Gradio: {gradio.__version__}")
    except ImportError:
        print("❌ Gradio未安装")
        return False
    
    # 检查设备
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"💻 计算设备: {device}")
    
    # 检查模型文件
    import os
    if os.path.exists("XY_Tokenizer/weights/xy_tokenizer.ckpt"):
        size = os.path.getsize("XY_Tokenizer/weights/xy_tokenizer.ckpt") / (1024**3)
        print(f"✅ 模型权重: 已下载 ({size:.1f}GB)")
    else:
        print("⚠️ 模型权重: 未找到")
    
    print("🎉 基础安装验证完成！")
    return True

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
EOF

$PYTHON_CMD test_install.py
TEST_RESULT=$?

# 清理临时文件
rm test_install.py

# 创建启动脚本
echo "📝 创建启动脚本..."
cat > start_ui.sh << 'EOF'
#!/bin/bash
# MOSS-TTSD 快速启动脚本

echo "🚀 启动MOSS-TTSD播客生成器..."

# 激活虚拟环境
if [ -d "moss_ttsd_env" ]; then
    source moss_ttsd_env/bin/activate
    echo "✅ 虚拟环境已激活"
else
    echo "❌ 虚拟环境不存在，请先运行install.sh"
    exit 1
fi

# 选择启动模式
echo "🎯 请选择启动模式:"
echo "1) 基础播客生成器 (简单易用)"
echo "2) 高级播客生成器 (参数控制)"
echo "3) 命令行推理"

read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo "🎙️ 启动基础播客生成器..."
        python enhanced_podcast_ui.py
        ;;
    2)
        echo "🎛️ 启动高级播客生成器..."
        python advanced_podcast_ui.py
        ;;
    3)
        echo "⌨️ 命令行推理模式..."
        echo "💡 示例命令: python inference.py --jsonl examples/examples.jsonl --output_dir outputs"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac
EOF

chmod +x start_ui.sh

# 安装总结
echo ""
echo "🎉 安装完成！"
echo "================================"

if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ 所有组件安装成功"
    echo ""
    echo "🚀 使用方法:"
    echo "1. 启动Web界面: ./start_ui.sh"
    echo "2. 或者直接运行:"
    echo "   source moss_ttsd_env/bin/activate"
    echo "   python enhanced_podcast_ui.py"
    echo "3. 命令行推理:"
    echo "   python inference.py --jsonl examples/examples.jsonl"
    echo ""
    echo "📚 更多信息请查看: docs/安装配置指南.md"
else
    echo "⚠️ 安装过程中出现问题，请查看上述错误信息"
    echo "💡 可以尝试手动安装缺失的组件"
fi

echo ""
echo "💡 提示: 重新打开终端时，记得激活虚拟环境:"
echo "   source moss_ttsd_env/bin/activate"
# 🛠️ MOSS-TTSD 快速安装指南

## 🚀 一键安装 (推荐)

### 克隆项目
```bash
git clone https://github.com/OpenMOSS/MOSS-TTSD.git
cd MOSS-TTSD
```

### 自动安装
```bash
# 一键安装脚本 (macOS/Linux)
./install.sh

# Windows用户请使用PowerShell运行:
# python -m venv moss_ttsd_env
# moss_ttsd_env\Scripts\activate
# pip install -r requirements.txt
```

### 启动使用
```bash
# 快速启动
./start_ui.sh

# 或者选择特定模式
./start_ui.sh 1  # 基础UI
./start_ui.sh 2  # 高级UI
./start_ui.sh 4  # 健康检查
```

---

## 📋 系统要求

- **Python**: 3.8+ (推荐 3.9-3.11)
- **内存**: 8GB+ RAM (推荐 16GB+)
- **存储**: 20GB+ 可用空间
- **网络**: 下载模型需要网络连接

---

## 🔧 故障排除

### 常见问题

**Q: 安装脚本执行失败**
```bash
# 给脚本添加执行权限
chmod +x install.sh start_ui.sh
# 然后重新运行
./install.sh
```

**Q: Python版本不对**
```bash
# macOS用户
brew install python@3.9

# Ubuntu用户
sudo apt install python3.9
```

**Q: 模型下载失败**
```bash
# 使用镜像下载
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download fnlp/XY_Tokenizer_TTSD_V0 xy_tokenizer.ckpt --local-dir ./XY_Tokenizer/weights/
```

**Q: 环境检查失败**
```bash
# 运行完整检查
python health_check.py

# 重新安装
./install.sh
```

---

## 📖 详细文档

完整安装配置指南请查看: **[docs/安装配置指南.md](docs/安装配置指南.md)**

---

## ✅ 验证安装

```bash
# 检查环境
python health_check.py

# 快速测试
./start_ui.sh 3  # 命令行推理演示

# 启动界面
./start_ui.sh 1  # 基础播客生成器
```

---

**🎉 安装完成后，通过Web界面即可开始制作个性化播客！**
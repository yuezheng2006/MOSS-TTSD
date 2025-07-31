#!/usr/bin/env python3
"""
MOSS-TTSD 增强版播客生成 Gradio 界面
包含默认音频示例和优化的用户体验
"""
import gradio as gr
import torch
import torchaudio
import json
import os
import tempfile
import time
import shutil
from datetime import datetime
import numpy as np

from generation_utils import load_model

# 全局配置
MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."

# 全局模型变量
tokenizer = None
model = None
spt = None
device = None
is_loading = False

def get_default_audio_files():
    """获取默认音频文件路径"""
    examples_dir = "examples"
    default_files = {
        "speaker1_audio": os.path.join(examples_dir, "zh_spk1_moon.wav"),
        "speaker1_text": "周一到周五，每天早晨七点半到九点半的直播片段。言下之意呢，就是废话有点多，大家也别嫌弃，因为这都是直播间最真实的状态了。",
        "speaker2_audio": os.path.join(examples_dir, "zh_spk2_moon.wav"), 
        "speaker2_text": "如果大家想听到更丰富更及时的直播内容，记得在周一到周五准时进入直播间，和大家一起畅聊新消费新科技新趋势。"
    }
    
    # 检查文件是否存在
    for key, path in default_files.items():
        if "audio" in key and not os.path.exists(path):
            print(f"⚠️ 默认音频文件不存在: {path}")
    
    return default_files

def get_scenario_examples():
    """获取场景示例"""
    scenarios = {
        "科技播客：AI发展": {
            "file": "scenarios/科技播客_AI发展.jsonl",
            "description": "探讨人工智能发展趋势，专业而富有前瞻性的科技对话"
        },
        "教育播客：学习方法": {
            "file": "scenarios/教育播客_学习方法.jsonl", 
            "description": "分享有效学习方法，教育专家的实用指导"
        },
        "健康播客：运动健身": {
            "file": "scenarios/健康播客_运动健身.jsonl",
            "description": "科学健身指导，营养师与教练的专业建议"
        },
        "商业播客：创业经验": {
            "file": "scenarios/商业播客_创业经验.jsonl",
            "description": "创业实战经验分享，商业思维与实践指导"
        },
        "生活播客：美食文化": {
            "file": "scenarios/生活播客_美食文化.jsonl",
            "description": "品味美食文化，传统与现代的完美结合"
        },
        "心理播客：情绪管理": {
            "file": "scenarios/心理播客_情绪管理.jsonl",
            "description": "心理健康指导，温暖专业的情绪管理建议"
        }
    }
    return scenarios

def load_scenario_content(scenario_name):
    """加载指定场景的内容"""
    scenarios = get_scenario_examples()
    if scenario_name not in scenarios:
        return None
    
    scenario_file = scenarios[scenario_name]["file"]
    if not os.path.exists(scenario_file):
        return None
    
    try:
        with open(scenario_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return content
    except Exception as e:
        print(f"❌ 加载场景失败: {e}")
        return None

def load_models_once():
    """一次性加载模型"""
    global tokenizer, model, spt, device, is_loading
    
    if tokenizer is None and not is_loading:
        is_loading = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 正在加载模型到设备: {device}")
        
        try:
            tokenizer, model, spt = load_model(
                MODEL_PATH, 
                SPT_CONFIG_PATH, 
                SPT_CHECKPOINT_PATH,
                torch_dtype=torch.float32,
                attn_implementation="eager"
            )
            
            model = model.to(device)
            spt = spt.to(device)
            print("✅ 模型加载完成")
            is_loading = False
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            is_loading = False
            return False
    
    return tokenizer is not None

def use_default_audio(audio_type):
    """使用默认音频文件"""
    defaults = get_default_audio_files()
    audio_key = f"{audio_type}_audio"
    text_key = f"{audio_type}_text"
    
    if audio_key in defaults and os.path.exists(defaults[audio_key]):
        # 创建临时副本供Gradio使用
        try:
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_filename = f"gradio_default_{audio_type}_{int(time.time())}_{os.path.basename(defaults[audio_key])}"
            temp_path = os.path.join(temp_dir, temp_filename)
            shutil.copy2(defaults[audio_key], temp_path)
            return temp_path, defaults[text_key]
        except Exception as e:
            print(f"❌ 复制默认音频文件失败: {e}")
            return None, defaults[text_key]
    else:
        return None, "默认音频文件不存在"

def quick_generate_demo():
    """快速生成演示"""
    defaults = get_default_audio_files()
    demo_text = "[S1]大家好，欢迎收听今天的AI播客。[S2]你好，我是今天的嘉宾。[S1]今天我们来聊聊MOSS-TTSD这个语音合成项目。[S2]确实很有意思，它能生成非常自然的对话音频。[S1]而且完全开源，对开发者很友好。[S2]是的，这为AI播客制作降低了门槛。"
    
    return (
        demo_text,
        defaults["speaker1_audio"] if os.path.exists(defaults["speaker1_audio"]) else None,
        defaults["speaker1_text"],
        defaults["speaker2_audio"] if os.path.exists(defaults["speaker2_audio"]) else None,
        defaults["speaker2_text"],
        True,
        42
    )

def generate_podcast_simple(
    dialogue_text,
    speaker1_audio,
    speaker1_text,
    speaker2_audio, 
    speaker2_text,
    use_normalize,
    seed,
    progress=gr.Progress()
):
    """简化版播客生成函数"""
    
    try:
        # 输入验证
        if not dialogue_text.strip():
            return None, "❌ 请输入对话文本"
        
        # 检查音频输入
        if speaker1_audio is None and speaker2_audio is None:
            return None, "❌ 请上传音频文件或点击'使用默认音频'按钮"
        
        progress(0.1, desc="🔄 加载模型...")
        
        # 确保模型已加载
        if not load_models_once():
            return None, "❌ 模型加载失败，请检查环境配置"
        
        # 使用简化的推理方法
        progress(0.3, desc="📝 准备数据...")
        
        # 创建临时JSONL文件
        temp_dir = tempfile.mkdtemp()
        jsonl_path = os.path.join(temp_dir, "temp_input.jsonl")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理音频文件路径
        if isinstance(speaker1_audio, str):
            spk1_path = speaker1_audio
        else:
            # 如果是上传的文件，复制到临时目录
            spk1_path = os.path.join(temp_dir, "speaker1.wav")
            if speaker1_audio is not None:
                shutil.copy(speaker1_audio, spk1_path)
            else:
                # 使用默认音频
                defaults = get_default_audio_files()
                spk1_path = defaults["speaker1_audio"]
        
        if isinstance(speaker2_audio, str):
            spk2_path = speaker2_audio
        else:
            spk2_path = os.path.join(temp_dir, "speaker2.wav")
            if speaker2_audio is not None:
                shutil.copy(speaker2_audio, spk2_path)
            else:
                defaults = get_default_audio_files()
                spk2_path = defaults["speaker2_audio"]
        
        # 创建JSONL数据
        jsonl_data = {
            "base_path": "",
            "text": dialogue_text,
            "prompt_audio_speaker1": spk1_path,
            "prompt_text_speaker1": speaker1_text if speaker1_text.strip() else "参考文本",
            "prompt_audio_speaker2": spk2_path,
            "prompt_text_speaker2": speaker2_text if speaker2_text.strip() else "参考文本"
        }
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            json.dump(jsonl_data, f, ensure_ascii=False)
        
        progress(0.5, desc="🧠 生成音频...")
        
        # 调用推理脚本
        import subprocess
        cmd = [
            "python", "inference.py",
            "--jsonl", jsonl_path,
            "--output_dir", output_dir,
            "--seed", str(int(seed)),
            "--dtype", "fp32",
            "--attn_implementation", "eager"
        ]
        
        if use_normalize:
            cmd.append("--use_normalize")
        
        # 执行推理
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        generation_time = time.time() - start_time
        
        progress(0.9, desc="📁 处理结果...")
        
        if result.returncode != 0:
            error_msg = f"❌ 生成失败:\n{result.stderr}"
            return None, error_msg
        
        # 查找生成的音频文件
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
        if not output_files:
            return None, "❌ 没有找到生成的音频文件"
        
        output_file = os.path.join(output_dir, output_files[0])
        
        # 获取音频信息
        try:
            audio_info = torchaudio.info(output_file)
            duration = audio_info.num_frames / audio_info.sample_rate
        except:
            duration = 0
        
        progress(1.0, desc="✅ 完成!")
        
        result_info = f"""✅ 播客生成成功！
        
📊 生成统计:
• 音频时长: {duration:.1f}秒
• 生成时间: {generation_time:.1f}秒
• 输出文件: {output_files[0]}
• 使用设备: {device}
• 文本标准化: {'是' if use_normalize else '否'}
• 随机种子: {seed}

🎵 音频文件已生成，可以点击播放或下载
        """
        
        return output_file, result_info
        
    except Exception as e:
        error_msg = f"❌ 生成过程中出错:\n{str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg

def create_enhanced_ui():
    """创建增强版Gradio界面"""
    
    # 自定义CSS
    custom_css = """
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .input-section {
        background: #f8fafc;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 1px solid #e2e8f0;
    }
    .output-section {
        background: #f0fff4;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #9ae6b4;
    }
    .control-buttons {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #fed7aa;
    }
    .speaker-controls {
        margin-top: 10px;
    }
    .main-action-buttons {
        margin-top: 20px;
        text-align: center;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🎙️ MOSS-TTSD 播客生成器", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🎙️ MOSS-TTSD 播客生成器</h1>
            <p>✨ 基于AI的高质量双人播客内容生成 ✨</p>
            <p><em>简洁 · 美观 · 高效</em></p>
        </div>
        """)
        
        with gr.Row():
            # 左侧输入区域
            with gr.Column(scale=3):
                # 对话内容区域
                with gr.Group():
                    gr.Markdown("### 📝 对话内容")
                    dialogue_text = gr.TextArea(
                        label="",
                        placeholder="请输入对话内容，使用[S1]和[S2]标记不同说话者...\n\n💡 示例：\n[S1]大家好，欢迎收听今天的播客\n[S2]你好，我是今天的嘉宾\n[S1]今天我们来聊聊AI技术...",
                        lines=6,
                        value="[S1]大家好，欢迎收听今天的《AI前沿》播客。[S2]你好，我是嘉宾阿明。[S1]今天我们来聊聊最新的语音合成技术，特别是MOSS-TTSD这个项目。[S2]是的，这个开源项目确实很有意思，它能生成非常自然的对话音频。[S1]而且支持零样本语音克隆，只需要很短的参考音频就能复制音色。[S2]这对内容创作者来说是个巨大的福音，可以大大提高播客制作效率。"
                    )
                
                # 快速操作按钮区域
                gr.HTML('<div class="control-buttons">')
                gr.Markdown("### 🚀 快速操作")
                
                # 场景选择
                with gr.Row():
                    scenario_dropdown = gr.Dropdown(
                        choices=list(get_scenario_examples().keys()),
                        label="🎭 选择播客场景",
                        value="科技播客：AI发展",
                        scale=2
                    )
                    load_scenario_btn = gr.Button("📥 加载场景", variant="secondary", scale=1)
                
                # 其他快速操作
                with gr.Row():
                    default_audio_btn = gr.Button("🎧 默认音频", variant="secondary", scale=1)
                    clear_btn = gr.Button("🗑️ 清空内容", variant="secondary", scale=1)
                    random_demo_btn = gr.Button("🎲 随机示例", variant="secondary", scale=1)
                gr.HTML('</div>')
                
                # 说话者设置区域
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown("### 🎵 说话者1 (女声)")
                                gr.HTML('<div class="speaker-controls">')
                                default_btn1 = gr.Button("🎧", variant="secondary", size="sm", scale=0, min_width=40)
                                gr.HTML('</div>')
                            speaker1_audio = gr.Audio(
                                label="参考音频",
                                type="filepath"
                            )
                            speaker1_text = gr.TextArea(
                                label="参考文本",
                                lines=2,
                                value="周一到周五，每天早晨七点半到九点半的直播片段。言下之意呢，就是废话有点多，大家也别嫌弃。"
                            )
                    
                    with gr.Column():
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown("### 🎵 说话者2 (男声)")
                                gr.HTML('<div class="speaker-controls">')
                                default_btn2 = gr.Button("🎧", variant="secondary", size="sm", scale=0, min_width=40)
                                gr.HTML('</div>')
                            speaker2_audio = gr.Audio(
                                label="参考音频",
                                type="filepath"
                            )
                            speaker2_text = gr.TextArea(
                                label="参考文本",
                                lines=2,
                                value="如果大家想听到更丰富更及时的直播内容，记得在周一到周五准时进入直播间，和大家一起畅聊。"
                            )
                
                # 生成设置区域
                with gr.Group():
                    gr.Markdown("### ⚙️ 生成设置")
                    with gr.Row():
                        use_normalize = gr.Checkbox(
                            label="✅ 文本标准化",
                            value=True,
                            info="推荐开启"
                        )
                        seed = gr.Number(
                            label="🎲 随机种子",
                            value=42,
                            precision=0,
                            info="42为默认值"
                        )
                
                # 主要操作按钮
                gr.HTML('<div class="main-action-buttons">')
                generate_btn = gr.Button("🎬 生成播客", variant="primary", size="lg", scale=1)
                gr.HTML('</div>')
            
            # 右侧输出区域
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 🎧 生成结果")
                    output_audio = gr.Audio(
                        label="生成的播客音频",
                        type="filepath"
                    )
                    
                    result_info = gr.TextArea(
                        label="生成信息",
                        lines=10,
                        interactive=False,
                        value="💡 请设置参数并点击'生成播客'按钮开始\n\n📋 使用提示:\n• 上传清晰的音频样本(10-30秒)\n• 参考文本必须与音频匹配\n• 生成时间约1-5分钟\n• 支持中英文对话"
                    )
                
                with gr.Group():
                    gr.Markdown("### 📚 使用指南")
                    
                    # 动态显示选中场景的描述
                    scenario_info = gr.Markdown("💡 **科技播客：AI发展** - 探讨人工智能发展趋势，专业而富有前瞻性的科技对话")
                    
                    gr.Markdown("""
                    **🎯 快速开始:**
                    1. 🎭 选择感兴趣的播客场景
                    2. 📥 点击"加载场景"自动填入内容  
                    3. 🎧 点击"默认音频"加载参考音频
                    4. 🎬 点击"生成播客"开始制作
                    
                    **🎲 更多选项:**
                    - **随机示例**: 随机选择一个场景体验
                    - **清空内容**: 快速重置开始自定义
                    - **🎧 按钮**: 单独加载某个说话者音频
                    
                    **📝 格式说明:**
                    - `[S1]` 标记说话者1 (通常为女声)
                    - `[S2]` 标记说话者2 (通常为男声)  
                    - 参考文本必须与音频内容完全匹配
                    
                    **🎵 音频要求:**
                    - 格式: WAV, MP3, FLAC
                    - 时长: 10-30秒最佳
                    - 质量: 清晰无背景噪音
                    - 语速: 自然正常语速
                    
                    **🎭 可用场景:**
                    - 🔬 科技类: AI、技术趋势讨论
                    - 📚 教育类: 学习方法、知识分享
                    - 💪 健康类: 运动健身、生活指导
                    - 💼 商业类: 创业经验、商业思维
                    - 🍜 生活类: 美食文化、传统传承
                    - 🧠 心理类: 情绪管理、心理健康
                    """)
        
        # 事件绑定
        generate_btn.click(
            fn=generate_podcast_simple,
            inputs=[
                dialogue_text,
                speaker1_audio,
                speaker1_text,
                speaker2_audio,
                speaker2_text,
                use_normalize,
                seed
            ],
            outputs=[output_audio, result_info],
            show_progress=True
        )
        
        # 加载场景按钮
        def load_selected_scenario(scenario_name):
            content = load_scenario_content(scenario_name)
            if content:
                # 处理音频文件路径
                base_path = content.get("base_path", "examples")
                
                def get_audio_path(audio_filename):
                    if not audio_filename:
                        return None
                    
                    # 构建完整路径
                    full_path = os.path.join(base_path, audio_filename)
                    if not os.path.exists(full_path):
                        print(f"⚠️ 音频文件不存在: {full_path}")
                        return None
                    
                    # 创建临时副本供Gradio使用
                    try:
                        import tempfile
                        temp_dir = tempfile.gettempdir()
                        temp_filename = f"gradio_audio_{int(time.time())}_{audio_filename}"
                        temp_path = os.path.join(temp_dir, temp_filename)
                        shutil.copy2(full_path, temp_path)
                        return temp_path
                    except Exception as e:
                        print(f"❌ 复制音频文件失败: {e}")
                        return None
                
                audio1_path = get_audio_path(content.get("prompt_audio_speaker1"))
                audio2_path = get_audio_path(content.get("prompt_audio_speaker2"))
                
                return (
                    content["text"],
                    audio1_path,
                    content.get("prompt_text_speaker1", ""),
                    audio2_path,
                    content.get("prompt_text_speaker2", "")
                )
            else:
                return (
                    f"❌ 无法加载场景: {scenario_name}",
                    None, "", None, ""
                )
        
        load_scenario_btn.click(
            fn=load_selected_scenario,
            inputs=[scenario_dropdown],
            outputs=[
                dialogue_text,
                speaker1_audio,
                speaker1_text,
                speaker2_audio,
                speaker2_text
            ]
        )
        
        # 随机示例按钮
        def load_random_scenario():
            scenarios = list(get_scenario_examples().keys())
            import random
            random_scenario = random.choice(scenarios)
            content = load_scenario_content(random_scenario)
            if content:
                return (
                    content["text"],
                    content.get("prompt_audio_speaker1", "zh_spk1_moon.wav"),
                    content.get("prompt_text_speaker1", ""),
                    content.get("prompt_audio_speaker2", "zh_spk2_moon.wav"),
                    content.get("prompt_text_speaker2", ""),
                    random_scenario
                )
            else:
                return ("❌ 加载随机示例失败", None, "", None, "", "科技播客：AI发展")
        
        random_demo_btn.click(
            fn=load_random_scenario,
            outputs=[
                dialogue_text,
                speaker1_audio,
                speaker1_text,
                speaker2_audio,
                speaker2_text,
                scenario_dropdown
            ]
        )
        
        # 场景选择变化时更新描述
        def update_scenario_info(scenario_name):
            scenarios = get_scenario_examples()
            if scenario_name in scenarios:
                description = scenarios[scenario_name]["description"]
                return f"💡 **{scenario_name}** - {description}"
            return "💡 请选择一个播客场景"
        
        scenario_dropdown.change(
            fn=update_scenario_info,
            inputs=[scenario_dropdown],
            outputs=[scenario_info]
        )
        
        # 加载默认音频按钮
        def load_all_defaults():
            defaults = get_default_audio_files()
            
            def copy_to_temp(file_path, speaker_name):
                if not os.path.exists(file_path):
                    return None
                try:
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    temp_filename = f"gradio_all_{speaker_name}_{int(time.time())}_{os.path.basename(file_path)}"
                    temp_path = os.path.join(temp_dir, temp_filename)
                    shutil.copy2(file_path, temp_path)
                    return temp_path
                except Exception as e:
                    print(f"❌ 复制音频文件失败 {file_path}: {e}")
                    return None
            
            audio1_path = copy_to_temp(defaults["speaker1_audio"], "spk1")
            audio2_path = copy_to_temp(defaults["speaker2_audio"], "spk2")
            
            return (
                audio1_path, defaults["speaker1_text"],
                audio2_path, defaults["speaker2_text"]
            )
        
        default_audio_btn.click(
            fn=load_all_defaults,
            outputs=[speaker1_audio, speaker1_text, speaker2_audio, speaker2_text]
        )
        
        # 单独的默认音频按钮
        default_btn1.click(
            fn=lambda: use_default_audio("speaker1"),
            outputs=[speaker1_audio, speaker1_text]
        )
        
        default_btn2.click(
            fn=lambda: use_default_audio("speaker2"),
            outputs=[speaker2_audio, speaker2_text]
        )
        
        # 清空内容按钮
        def clear_all_content():
            return (
                "",  # dialogue_text
                None,  # speaker1_audio
                "",  # speaker1_text
                None,  # speaker2_audio
                "",  # speaker2_text
                None,  # output_audio
                "🗑️ 内容已清空，请重新输入"  # result_info
            )
        
        clear_btn.click(
            fn=clear_all_content,
            outputs=[
                dialogue_text,
                speaker1_audio,
                speaker1_text,
                speaker2_audio,
                speaker2_text,
                output_audio,
                result_info
            ]
        )
    
    return demo

def main():
    """主函数"""
    print("🚀 启动增强版 MOSS-TTSD 播客生成器...")
    
    # 检查环境
    if not os.path.exists("examples"):
        print("⚠️ examples目录不存在，某些默认功能可能不可用")
    
    # 创建界面
    demo = create_enhanced_ui()
    
    # 启动服务
    print("🌐 启动Gradio服务...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # 使用不同端口避免冲突
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
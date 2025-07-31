#!/usr/bin/env python3
"""
MOSS-TTSD 高级播客生成器
支持语气、语调等高级参数控制
"""
import gradio as gr
import torch
import torchaudio
import json
import os
import tempfile
import time
from datetime import datetime
import numpy as np
import subprocess

from generation_utils import load_model
from transformers import GenerationConfig

# 全局配置
MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
MAX_CHANNELS = 8

# 全局模型变量
tokenizer = None
model = None
spt = None
device = None
is_loading = False

# 预设风格配置
STYLE_PRESETS = {
    "自然对话": {
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "description": "平衡自然的对话语气，适合大多数播客场景"
    },
    "新闻播报": {
        "temperature": 0.8,
        "top_k": 30,
        "top_p": 0.85,
        "repetition_penalty": 1.05,
        "description": "稳重专业的播报风格，适合严肃内容"
    },
    "娱乐节目": {
        "temperature": 1.2,
        "top_k": 80,
        "top_p": 0.95,
        "repetition_penalty": 1.15,
        "description": "活泼有趣的娱乐风格，表达丰富多彩"
    },
    "教育讲解": {
        "temperature": 0.9,
        "top_k": 40,
        "top_p": 0.88,
        "repetition_penalty": 1.08,
        "description": "清晰耐心的教学语气，逻辑性强"
    },
    "商务正式": {
        "temperature": 0.7,
        "top_k": 25,
        "top_p": 0.8,
        "repetition_penalty": 1.03,
        "description": "正式商务语气，简洁专业"
    },
    "轻松聊天": {
        "temperature": 1.1,
        "top_k": 60,
        "top_p": 0.92,
        "repetition_penalty": 1.12,
        "description": "轻松随意的聊天语气，亲切自然"
    }
}

def get_default_audio_files():
    """获取默认音频文件路径"""
    examples_dir = "examples"
    default_files = {
        "speaker1_audio": os.path.join(examples_dir, "zh_spk1_moon.wav"),
        "speaker1_text": "周一到周五，每天早晨七点半到九点半的直播片段。言下之意呢，就是废话有点多，大家也别嫌弃，因为这都是直播间最真实的状态了。",
        "speaker2_audio": os.path.join(examples_dir, "zh_spk2_moon.wav"), 
        "speaker2_text": "如果大家想听到更丰富更及时的直播内容，记得在周一到周五准时进入直播间，和大家一起畅聊新消费新科技新趋势。"
    }
    
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

def apply_style_preset(style_name):
    """应用预设风格"""
    if style_name in STYLE_PRESETS:
        preset = STYLE_PRESETS[style_name]
        return (
            preset["temperature"],
            preset["top_k"],
            preset["top_p"],
            preset["repetition_penalty"],
            preset["description"]
        )
    return 1.0, 50, 0.9, 1.1, "默认设置"

def generate_advanced_podcast(
    dialogue_text,
    speaker1_audio,
    speaker1_text,
    speaker2_audio, 
    speaker2_text,
    use_normalize,
    seed,
    # 高级参数
    style_preset,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    max_length,
    progress=gr.Progress()
):
    """使用高级参数生成播客音频"""
    
    try:
        # 输入验证
        if not dialogue_text.strip():
            return None, "❌ 请输入对话文本"
        
        if speaker1_audio is None or speaker2_audio is None:
            return None, "❌ 请上传两个说话者的参考音频"
        
        if not speaker1_text.strip() or not speaker2_text.strip():
            return None, "❌ 请输入与参考音频对应的文本"
        
        progress(0.1, desc="🔄 加载模型...")
        
        # 确保模型已加载
        if not load_models_once():
            return None, "❌ 模型加载失败，请检查环境配置"
        
        # 设置随机种子
        if seed > 0:
            torch.manual_seed(seed)
        
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
            spk1_path = os.path.join(temp_dir, "speaker1.wav")
            if hasattr(speaker1_audio, 'name'):
                import shutil
                shutil.copy(speaker1_audio.name, spk1_path)
            else:
                defaults = get_default_audio_files()
                spk1_path = defaults["speaker1_audio"]
        
        if isinstance(speaker2_audio, str):
            spk2_path = speaker2_audio
        else:
            spk2_path = os.path.join(temp_dir, "speaker2.wav")
            if hasattr(speaker2_audio, 'name'):
                import shutil
                shutil.copy(speaker2_audio.name, spk2_path)
            else:
                defaults = get_default_audio_files()
                spk2_path = defaults["speaker2_audio"]
        
        # 创建JSONL数据
        jsonl_data = {
            "base_path": "",
            "text": dialogue_text,
            "prompt_audio_speaker1": spk1_path,
            "prompt_text_speaker1": speaker1_text,
            "prompt_audio_speaker2": spk2_path,
            "prompt_text_speaker2": speaker2_text
        }
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            json.dump(jsonl_data, f, ensure_ascii=False)
        
        progress(0.5, desc="🧠 生成音频 (使用高级参数)...")
        
        # 创建自定义生成配置文件
        config_override = {
            "temperature": temperature,
            "top_k": int(top_k),
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_length": int(max_length),
            "do_sample": True,
            "layers": [
                {
                    "temperature": temperature,
                    "top_k": int(top_k),
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty
                } for _ in range(MAX_CHANNELS)
            ]
        }
        
        # 保存配置到临时文件
        config_path = os.path.join(temp_dir, "generation_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_override, f, ensure_ascii=False, indent=2)
        
        # 调用推理脚本 (这里需要修改inference.py支持自定义配置)
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
        
        result_info = f"""✅ 高级播客生成成功！
        
📊 生成统计:
• 音频时长: {duration:.1f}秒
• 生成时间: {generation_time:.1f}秒
• 使用设备: {device}

🎛️ 使用参数:
• 风格预设: {style_preset}
• 语气温度: {temperature}
• 词汇多样性: {top_k}
• 表达流畅度: {top_p}
• 重复惩罚: {repetition_penalty}
• 最大长度: {max_length}
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

def create_advanced_ui():
    """创建高级参数控制的Gradio界面"""
    
    # 自定义CSS
    custom_css = """
    .main-container {
        max-width: 1600px;
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
    .control-buttons {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #fed7aa;
    }
    .advanced-params {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #90caf9;
    }
    .output-section {
        background: #f0fff4;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #9ae6b4;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🎛️ MOSS-TTSD 高级播客生成器", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🎛️ MOSS-TTSD 高级播客生成器</h1>
            <p>✨ 精确控制语气、语调和表达风格 ✨</p>
            <p><em>专业 · 可控 · 高质量</em></p>
        </div>
        """)
        
        with gr.Row():
            # 左侧输入区域
            with gr.Column(scale=2):
                # 对话内容区域
                with gr.Group():
                    gr.Markdown("### 📝 对话内容")
                    dialogue_text = gr.TextArea(
                        label="",
                        placeholder="请输入对话内容，使用[S1]和[S2]标记不同说话者...",
                        lines=6,
                        value="[S1]大家好，欢迎收听今天的《AI前沿》播客。[S2]你好，我是嘉宾阿明。[S1]今天我们来聊聊最新的语音合成技术，特别是MOSS-TTSD这个项目。[S2]是的，这个开源项目确实很有意思，它能生成非常自然的对话音频。"
                    )
                
                # 快速操作区域
                gr.HTML('<div class="control-buttons">')
                gr.Markdown("### 🚀 快速操作")
                with gr.Row():
                    scenario_dropdown = gr.Dropdown(
                        choices=list(get_scenario_examples().keys()),
                        label="🎭 选择播客场景",
                        value="科技播客：AI发展",
                        scale=2
                    )
                    load_scenario_btn = gr.Button("📥 加载场景", variant="secondary", scale=1)
                
                with gr.Row():
                    default_audio_btn = gr.Button("🎧 默认音频", variant="secondary")
                    clear_btn = gr.Button("🗑️ 清空内容", variant="secondary")
                    random_demo_btn = gr.Button("🎲 随机示例", variant="secondary")
                gr.HTML('</div>')
                
                # 说话者设置区域
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("### 🎵 说话者1 (女声)")
                            speaker1_audio = gr.Audio(label="参考音频", type="filepath")
                            speaker1_text = gr.TextArea(
                                label="参考文本",
                                lines=2,
                                value="周一到周五，每天早晨七点半到九点半的直播片段。"
                            )
                    
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("### 🎵 说话者2 (男声)")
                            speaker2_audio = gr.Audio(label="参考音频", type="filepath")
                            speaker2_text = gr.TextArea(
                                label="参考文本",
                                lines=2,
                                value="如果大家想听到更丰富更及时的直播内容，记得准时进入直播间。"
                            )
                
                # 基础设置
                with gr.Group():
                    gr.Markdown("### ⚙️ 基础设置")
                    with gr.Row():
                        use_normalize = gr.Checkbox(label="✅ 文本标准化", value=True)
                        seed = gr.Number(label="🎲 随机种子", value=42, precision=0)
                
                # 高级参数控制
                gr.HTML('<div class="advanced-params">')
                gr.Markdown("### 🎛️ 高级语音控制")
                
                # 风格预设
                with gr.Row():
                    style_preset = gr.Dropdown(
                        choices=list(STYLE_PRESETS.keys()),
                        label="🎨 风格预设",
                        value="自然对话",
                        scale=2
                    )
                    apply_preset_btn = gr.Button("📋 应用预设", variant="secondary", scale=1)
                
                style_description = gr.Markdown("💡 **自然对话** - 平衡自然的对话语气，适合大多数播客场景")
                
                # 手动参数调节
                gr.Markdown("#### 🔧 手动参数调节")
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.5, maximum=1.5, value=1.0, step=0.1,
                        label="🌡️ 语气温度",
                        info="控制语气的自然度和随机性"
                    )
                    top_k = gr.Slider(
                        minimum=10, maximum=100, value=50, step=5,
                        label="🔝 词汇多样性",
                        info="控制词汇选择的丰富程度"
                    )
                
                with gr.Row():
                    top_p = gr.Slider(
                        minimum=0.7, maximum=1.0, value=0.9, step=0.05,
                        label="💬 表达流畅度",
                        info="控制表达的流畅性"
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0, maximum=1.3, value=1.1, step=0.05,
                        label="🔄 重复惩罚",
                        info="避免重复表达"
                    )
                
                max_length = gr.Slider(
                    minimum=1024, maximum=32768, value=16384, step=1024,
                    label="📏 最大生成长度",
                    info="控制生成的最大时长"
                )
                
                gr.HTML('</div>')
                
                # 主要操作按钮
                generate_btn = gr.Button("🎬 生成高级播客", variant="primary", size="lg")
            
            # 右侧输出区域
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎧 生成结果")
                    output_audio = gr.Audio(label="生成的播客音频", type="filepath")
                    result_info = gr.TextArea(
                        label="生成信息",
                        lines=15,
                        interactive=False,
                        value="💡 设置参数并点击'生成高级播客'开始制作\n\n🎛️ 高级参数说明:\n• 语气温度: 控制表达的随机性和自然度\n• 词汇多样性: 影响用词的丰富程度\n• 表达流畅度: 调节语言的流畅性\n• 重复惩罚: 避免重复表达"
                    )
                
                with gr.Group():
                    gr.Markdown("### 🎯 参数效果预览")
                    parameter_preview = gr.Markdown("""
                    **当前设置效果预测：**
                    - 🎭 风格：自然对话
                    - 🌡️ 语气：平衡自然
                    - 🔝 用词：中等丰富
                    - 💬 流畅度：高
                    """)
        
        # 事件绑定
        
        # 生成按钮
        generate_btn.click(
            fn=generate_advanced_podcast,
            inputs=[
                dialogue_text, speaker1_audio, speaker1_text,
                speaker2_audio, speaker2_text, use_normalize, seed,
                style_preset, temperature, top_k, top_p,
                repetition_penalty, max_length
            ],
            outputs=[output_audio, result_info],
            show_progress=True
        )
        
        # 风格预设应用
        def apply_and_update_preset(style_name):
            temp, tk, tp, rp, desc = apply_style_preset(style_name)
            return temp, tk, tp, rp, f"💡 **{style_name}** - {desc}"
        
        apply_preset_btn.click(
            fn=apply_and_update_preset,
            inputs=[style_preset],
            outputs=[temperature, top_k, top_p, repetition_penalty, style_description]
        )
        
        # 预设选择变化时更新描述
        style_preset.change(
            fn=lambda style: f"💡 **{style}** - {STYLE_PRESETS.get(style, {}).get('description', '')}",
            inputs=[style_preset],
            outputs=[style_description]
        )
        
        # 参数变化时更新预览
        def update_parameter_preview(temp, tk, tp, rp, style):
            effects = []
            
            # 语气分析
            if temp < 0.8:
                effects.append("🌡️ 语气：稳重保守")
            elif temp > 1.1:
                effects.append("🌡️ 语气：活泼多变")
            else:
                effects.append("🌡️ 语气：自然平衡")
            
            # 用词分析  
            if tk < 35:
                effects.append("🔝 用词：保守精选")
            elif tk > 65:
                effects.append("🔝 用词：丰富多样")
            else:
                effects.append("🔝 用词：中等丰富")
            
            # 流畅度分析
            if tp < 0.85:
                effects.append("💬 流畅度：精确严谨")
            elif tp > 0.93:
                effects.append("💬 流畅度：自由流畅")
            else:
                effects.append("💬 流畅度：平衡流畅")
            
            return f"""**当前设置效果预测：**
- 🎭 风格：{style}
- {effects[0]}
- {effects[1]}  
- {effects[2]}
- 🔄 重复控制：{'严格' if rp > 1.15 else ('适中' if rp > 1.05 else '宽松')}"""
        
        for param in [temperature, top_k, top_p, repetition_penalty, style_preset]:
            param.change(
                fn=update_parameter_preview,
                inputs=[temperature, top_k, top_p, repetition_penalty, style_preset],
                outputs=[parameter_preview]
            )
        
        # 场景加载等其他功能 (简化版)
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
                        import shutil
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
            return "", None, "", None, ""
        
        load_scenario_btn.click(
            fn=load_selected_scenario,
            inputs=[scenario_dropdown],
            outputs=[dialogue_text, speaker1_audio, speaker1_text, speaker2_audio, speaker2_text]
        )
        
        # 默认音频加载
        def load_all_defaults():
            defaults = get_default_audio_files()
            
            def copy_to_temp(file_path, speaker_name):
                if not os.path.exists(file_path):
                    return None
                try:
                    import tempfile
                    import shutil
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
        
        # 清空内容
        def clear_all():
            return "", None, "", None, "", None, "🗑️ 内容已清空"
        
        clear_btn.click(
            fn=clear_all,
            outputs=[dialogue_text, speaker1_audio, speaker1_text, speaker2_audio, speaker2_text, output_audio, result_info]
        )
    
    return demo

def main():
    """主函数"""
    print("🚀 启动MOSS-TTSD高级播客生成器...")
    
    # 创建界面
    demo = create_advanced_ui()
    
    # 启动服务
    print("🌐 启动Gradio服务...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,  # 使用不同端口
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
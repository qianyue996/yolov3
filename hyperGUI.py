import gradio as gr
import json
import os

CONFIG_PATH = "config/trainParameter.json"


# 读取配置文件的函数
def read_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}


# 更新配置文件的函数
def update_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


# Gradio界面函数，读取配置
def gradio_interface_read():
    config = read_config()
    return json.dumps(config, indent=4)


# Gradio界面函数，更新配置
def gradio_interface_update(*args):
    config = read_config()

    # 构建字典（假设配置文件和输入组件的顺序一致）
    param_keys = list(config.keys())  # 获取配置文件的键列表
    params = dict(zip(param_keys, args))

    config.update(params)
    update_config(config)
    return json.dumps(config, indent=4)


# 动态创建输入组件的函数
def create_input_components(config):
    input_components = []
    for key, value in config.items():
        if isinstance(value, (int, float)):  # 判断是否是数字类型
            input_components.append(
                gr.Slider(
                    minimum=0.000001, maximum=12, step=0.000001, label=key, value=value
                )
            )
        elif isinstance(value, list):  # 判断是否是列表
            input_components.append(
                gr.Textbox(label=f"{key} (List)", value=", ".join(map(str, value)))
            )
        else:
            # 如果是其他类型，可以做更细致的处理，默认为文本框
            input_components.append(gr.Textbox(label=key, value=value))
    return input_components


# 创建Gradio界面
def launch_gradio():
    config = read_config()

    with gr.Blocks() as demo:
        gr.Markdown("### Update or Read Configuration")

        # 动态生成的输入组件
        input_components = create_input_components(config)

        # 读取配置按钮
        read_btn = gr.Button("Read Config")
        # 更新配置按钮
        update_btn = gr.Button("Update Config")

        # 用于展示配置的输出框
        output_json = gr.Textbox(
            label="Config Output", interactive=False, value=json.dumps(config, indent=4)
        )

        # 设置按钮点击时执行的函数
        read_btn.click(
            gradio_interface_read, inputs=[], outputs=output_json
        )  # 点击 "Read Config" 按钮，读取配置
        update_btn.click(
            gradio_interface_update, inputs=input_components, outputs=output_json
        )  # 点击 "Update Config" 按钮，更新配置

    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    launch_gradio()

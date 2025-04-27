import gradio as gr
import json
import os

CONFIG_PATH = 'config/config.json'

# 读取配置文件的函数
def read_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

# 更新配置文件的函数
def update_config(lr, batch_size, l_loc, l_cls, l_obj, l_noobj):
    config = read_config()
    config['lr'] = lr
    config['batch_size'] = batch_size
    config['l_loc'] = l_loc
    config['l_cls'] = l_cls
    config['l_obj'] = l_obj
    config['l_noobj'] = l_noobj
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return json.dumps(config, indent=4)

# Gradio界面函数
def gradio_interface_read():
    config = read_config()
    return json.dumps(config, indent=4)

def gradio_interface_update(lr, batch_size, l_loc, l_cls, l_obj, l_noobj):
    update_config(lr, batch_size, l_loc, l_cls, l_obj, l_noobj)
    config = read_config()  # 返回更新后的配置
    return json.dumps(config, indent=4)

# 创建Gradio界面
def launch_gradio():
    config = read_config()

    with gr.Blocks() as demo:
        gr.Markdown("### Update or Read Learning Rate and Batch Size Configuration")
        
        # 学习率和批量大小输入组件
        lr = gr.Slider(minimum=0.000001, maximum=0.1, step=0.0001, label="Learning Rate", value=config.get('lr'))
        batch_size = gr.Slider(minimum=1, maximum=128, step=1, label="Batch Size", value=config.get('batch_size'))
        l_loc = gr.Slider(minimum=0.0001, maximum=10, step=0.001, label="L_Loc", value=config.get('l_loc'))
        l_cls = gr.Slider(minimum=0.0001, maximum=10, step=0.001, label="L_Cls", value=config.get('l_cls'))
        l_obj = gr.Slider(minimum=0.0001, maximum=10, step=0.001, label="L_Obj", value=config.get('l_obj'))
        l_noobj = gr.Slider(minimum=0.0001, maximum=10, step=0.001, label="L_NoObj", value=config.get('l_noobj'))

        # 读取配置按钮
        read_btn = gr.Button("Read Config")
        # 更新配置按钮
        update_btn = gr.Button("Update Config")
        
        # 用于展示配置的输出框
        output_json = gr.Textbox(label="Config Output", interactive=False, value=json.dumps(config, indent=4))
        
        # 设置按钮点击时执行的函数
        read_btn.click(gradio_interface_read, inputs=[], outputs=output_json)  # 点击 "Read Config" 按钮，读取配置
        update_btn.click(gradio_interface_update, inputs=[lr, batch_size, l_loc, l_cls, l_obj, l_noobj], outputs=output_json)  # 点击 "Update Config" 按钮，更新配置

    demo.launch()

if __name__ == '__main__':
    launch_gradio()

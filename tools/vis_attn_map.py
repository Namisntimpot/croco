import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, TapTool, CustomJS, LinearColorMapper, HoverTool
from bokeh.palettes import Viridis256
import argparse
import megfile
from PIL import Image

try:
    from tools.utils import load_image_as_rgba as load_image_for_bokeh, softmax, flip_attn_map
except:
    from utils import load_image_as_rgba as load_image_for_bokeh, softmax, flip_attn_map

'''
左边应该是target_image，但bokeh对象命名为了source或src，只是加载的是target_image
右边应该是source_image，但bokeh对象命名为了target或trg，只是加载的是source_image...
'''

def parse_patch_size(attn_map: np.ndarray, ori_h: int, ori_w: int):
    h = attn_map.shape[0]
    return ori_h // h

# --- 0. 解析命令行参数 ---
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True, help="path to the attention map file.")
parser.add_argument("-t", "--temperature", type=float, default=0.02)
args = parser.parse_args()

# --- 1. 加载数据 ---
temp = args.temperature
attn_map_file = args.file
with megfile.smart_open(attn_map_file, 'rb') as f:
    data = np.load(f, allow_pickle=True).item() # .item() 更安全

source_path = data['source_path']
target_path = data['target_path']
similarity_array: np.ndarray = data['attn_map']
similarity_array = flip_attn_map(similarity_array)
h, w = data['h'] if 'h' in data else None, data['w'] if 'w' in data else None

# 使用修正后的函数加载图像
src_image_data = load_image_for_bokeh(source_path)
trg_image_data = load_image_for_bokeh(target_path)

H, W = src_image_data.shape[:2]
patch_size = parse_patch_size(similarity_array, H, W)
h, w = H // patch_size, W // patch_size

# --- 2. 设置数据源 ---
src_source = ColumnDataSource(data=dict(image=[trg_image_data]))
trg_source = ColumnDataSource(data=dict(image=[src_image_data]))

patch_x = np.tile(np.arange(w) * patch_size + patch_size / 2, h)
patch_y = np.repeat(np.arange(h) * patch_size + patch_size / 2, w)
src_hover_source = ColumnDataSource(data={'x': patch_x, 'y': patch_y})

initial_similarity = similarity_array[0, 0, :, :].flatten()
heatmap_source = ColumnDataSource(data={'x': patch_x, 'y': patch_y, 'similarity': initial_similarity})

max_sim_index = np.argmax(initial_similarity)
max_point_source = ColumnDataSource(data={'x': [patch_x[max_sim_index]], 'y': [patch_y[max_sim_index]]})

src_selected_dot_source = ColumnDataSource(data={'x': [patch_x[0]], 'y': [patch_y[0]]})
signal_source = ColumnDataSource(data={'x': [], 'y': []})

# --- 3. 创建Bokeh图表 ---
hover_tool = HoverTool(tooltips=None, mode='mouse')
tools_src = [hover_tool, TapTool(), 'pan', 'wheel_zoom', 'reset']
p_src = figure(width=W + 50, height=H + 50, x_range=(0, W), y_range=(0, H),
               tools=tools_src, title="source image",
               match_aspect=True)
p_trg = figure(width=W + 50, height=H + 50, x_range=(0, W), y_range=(0, H),
               tools="pan,wheel_zoom,reset,help", title="target image",
               match_aspect=True)

# ============================ 修正点 2: 使用 image_rgba ============================
p_src.image_rgba(image='image', x=0, y=0, dw=W, dh=H, source=src_source)
p_trg.image_rgba(image='image', x=0, y=0, dw=W, dh=H, source=trg_source)
# ==============================================================================

p_src.grid.grid_line_color = None
p_src.rect(x='x', y='y', width=patch_size, height=patch_size, source=src_hover_source,
           fill_alpha=0, line_color=None,
           hover_fill_alpha=0.3, hover_fill_color='white')
p_src.circle(x='x', y='y', size=10, color='red', source=src_selected_dot_source) # 恢复源图红点

p_trg.grid.grid_line_color = None
color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)
p_trg.rect(x='x', y='y', width=patch_size, height=patch_size,
           fill_color={'field': 'similarity', 'transform': color_mapper},
           fill_alpha=0.5, line_color=None, source=heatmap_source)
p_trg.circle(x='x', y='y', size=10, color='red', source=max_point_source)
# p_trg.add_layout(p_trg.legend[0], 'right')
# p_trg.legend.click_policy = "hide"AP


# --- 4. 交互与回调 ---
p_src.js_on_event('tap', CustomJS(args={'source': signal_source}, code="""
    source.data = {x: [cb_obj.x], y: [cb_obj.y]};
"""))

def update_heatmap(attr, old, new):
    if not new['x']: return
    x_click, y_click = new['x'][0], new['y'][0]
    
    # 注意：因为图像已翻转，坐标系原点在左下角，i的计算是直接的
    i = max(0, min(h - 1, int(y_click / patch_size)))
    j = max(0, min(w - 1, int(x_click / patch_size)))
    
    # ============================ 修正点 3: 更新源图红点 ============================
    selected_x = j * patch_size + patch_size / 2
    selected_y = i * patch_size + patch_size / 2
    src_selected_dot_source.data = {'x': [selected_x], 'y': [selected_y]}
    # ==============================================================================

    new_similarity = similarity_array[i, j, :, :].flatten()
    if temp > 0:
        new_similarity = softmax(new_similarity / temp)
    else:
        new_similarity = (new_similarity - new_similarity.min()) / (new_similarity.max() - new_similarity.min())
    heatmap_source.data['similarity'] = new_similarity
    
    max_idx = np.argmax(new_similarity)
    max_point_source.data = {'x': [patch_x[max_idx]], 'y': [patch_y[max_idx]]}

signal_source.on_change('data', update_heatmap)

# --- 5. 布局与启动 ---
layout = row(p_src, p_trg)
curdoc().add_root(layout)
curdoc().title = "Attention Map"
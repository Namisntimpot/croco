# vis_attn_browser_final.py
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import megfile

from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column
from bokeh.models import (ColumnDataSource, CustomJS, LinearColorMapper, HoverTool,
                          Select, Div)
from bokeh.palettes import Viridis256

try:
    from tools.utils import load_image_as_rgba, softmax, flip_attn_map
except:
    from utils import load_image_as_rgba, softmax, flip_attn_map

'''
左边应该是target_image，但bokeh对象命名为了source或src，只是加载的是target_image
右边应该是source_image，但bokeh对象命名为了target或trg，只是加载的是source_image...
'''

# --- 1. 参数解析与文件扫描 ---
parser = argparse.ArgumentParser(description="Attention Map 浏览器")
parser.add_argument("-d", "--directory", type=str, required=True, help="包含.npy数据文件的目录路径")
parser.add_argument("-t", "--temperature", type=float, default=0.02)
args = parser.parse_args()

# data_dir = Path(args.directory)
data_dir = args.directory
npy_files = sorted([os.path.basename(p) for p in megfile.smart_glob(os.path.join(data_dir, '*.npy'))])

if not npy_files:
    error_div = Div(text=f"<h3>错误：在目录 '{data_dir}' 中没有找到任何 .npy 文件。</h3>")
    curdoc().add_root(error_div)
else:
    # --- global ---
    similarity_array = None
    temp = args.temperature

    # --- 2. create Bokeh components and data sources ---
    file_selector = Select(title="select Attention Map file:", value=npy_files[0], options=npy_files)
    info_div = Div(text="infomation panel")
    src_source = ColumnDataSource(data={'image': []})
    trg_source = ColumnDataSource(data={'image': []})
    heatmap_source = ColumnDataSource(data={'x': [], 'y': [], 'similarity': []})
    max_point_source = ColumnDataSource(data={'x': [], 'y': []})
    src_hover_source = ColumnDataSource(data={'x': [], 'y': []})
    signal_source = ColumnDataSource(data={'x': [], 'y': []})

    # create data source for the red point on the source
    src_selected_dot_source = ColumnDataSource(data={'x': [], 'y': []})

    # --- 3. create chart ---
    hover_tool = HoverTool(tooltips=None, mode='mouse')
    p_src = figure(x_range=(0, 1), y_range=(0, 1), tools=[hover_tool, 'tap', 'pan', 'wheel_zoom', 'reset'],
                   title="源图像", match_aspect=True)
    p_trg = figure(x_range=(0, 1), y_range=(0, 1), tools="pan,wheel_zoom,reset,help",
                   title="目标图像", match_aspect=True)

    src_image_renderer = p_src.image_rgba(image='image', x=0, y=0, dw=1, dh=1, source=src_source)
    trg_image_renderer = p_trg.image_rgba(image='image', x=0, y=0, dw=1, dh=1, source=trg_source)

    p_src.grid.grid_line_color = None
    p_src.rect(x='x', y='y', width=1, height=1, source=src_hover_source, fill_alpha=0,
               line_color=None, hover_fill_alpha=0.3, hover_fill_color='white')
    p_src.js_on_event('tap', CustomJS(args={'source': signal_source}, code="source.data = {x: [cb_obj.x], y: [cb_obj.y]};"))

    # add a red point to the source, indicating the current selected source patch
    p_src.circle(x='x', y='y', size=10, color='red', source=src_selected_dot_source)
    
    p_trg.grid.grid_line_color = None
    color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)
    heatmap_renderer = p_trg.rect(x='x', y='y', width=1, height=1,
                                  fill_color={'field': 'similarity', 'transform': color_mapper},
                                  fill_alpha=0.5, line_color=None, source=heatmap_source)
    p_trg.circle(x='x', y='y', size=10, color='red', source=max_point_source)
    # p_trg.add_layout(p_trg.legend[0], 'right')
    # p_trg.legend.click_policy = "hide"

    # --- 4. 核心回调函数 ---
    def update_visualization(filepath):
        global similarity_array
        try:
            name = os.path.basename(filepath)
            info_div.text = f"<b>Loading:</b> {os.path.basename(filepath)}"
            with megfile.smart_open(filepath, 'rb') as f:
                data = np.load(f, allow_pickle=True).tolist()
            src_img_path, trg_img_path = data["source_path"], data["target_path"]
            similarity_array = data["attn_map"]
            similarity_array = flip_attn_map(similarity_array)
            h = data['h'] if 'h' in data else None
            w = data['w'] if 'w' in data else None
            src_img = load_image_as_rgba(src_img_path, h, w)
            trg_img = load_image_as_rgba(trg_img_path, h, w)
            H, W = src_img.shape
            h, w = similarity_array.shape[0:2]
            patch_size_h, patch_size_w = H / h, W / w
            src_source.data = {'image': [trg_img]}
            trg_source.data = {'image': [src_img]}
            p_src.x_range.end, p_src.y_range.end = W, H
            p_trg.x_range.end, p_trg.y_range.end = W, H
            p_src.title.text = f"source image: {Path(src_img_path).name}"
            p_trg.title.text = f"target image: {Path(trg_img_path).name}"
            src_image_renderer.glyph.dw, src_image_renderer.glyph.dh = W, H
            trg_image_renderer.glyph.dw, trg_image_renderer.glyph.dh = W, H
            patch_x = np.tile(np.arange(w) * patch_size_w + patch_size_w / 2, h)
            patch_y = np.repeat(np.arange(h) * patch_size_h + patch_size_h / 2, w)
            src_hover_source.data = {'x': patch_x, 'y': patch_y}
            heatmap_source.data['x'], heatmap_source.data['y'] = patch_x, patch_y
            heatmap_renderer.glyph.width, heatmap_renderer.glyph.height = patch_size_w, patch_size_h
            update_heatmap(None, None, {'x': [0], 'y': [0]})
            info_div.text = f"<b>Selected file:</b> {name}<br><b>resolution:</b> {W}x{H}"
        except Exception as e:
            info_div.text = f"<b>Error when loading .npy file:</b> {name}<br><pre>{e}</pre>"

    def update_heatmap(attr, old, new):
        global temp
        if not new.get('x') or similarity_array is None: return
        H, W = p_src.y_range.end, p_src.x_range.end
        h, w = similarity_array.shape[0:2]
        patch_size_h, patch_size_w = H / h, W / w
        x_click, y_click = new['x'][0], new['y'][0]
        i = max(0, min(h - 1, int(y_click / patch_size_h)))
        j = max(0, min(w - 1, int(x_click / patch_size_w)))
        
        # patch's center coordinates
        selected_x = j * patch_size_w + patch_size_w / 2
        selected_y = i * patch_size_h + patch_size_h / 2
        # update the red point on the source
        src_selected_dot_source.data = {'x': [selected_x], 'y': [selected_y]}

        # update heatmap and the red point on the target image
        new_similarity = similarity_array[i, j, :, :].flatten()
        if temp > 0:
            new_similarity = softmax(new_similarity / temp)
        else:
            new_similarity = (new_similarity - new_similarity.min()) / (new_similarity.max() - new_similarity.min())
        heatmap_source.data['similarity'] = new_similarity
        max_idx = np.argmax(new_similarity)
        patch_x, patch_y = heatmap_source.data['x'], heatmap_source.data['y']
        max_point_source.data = {'x': [patch_x[max_idx]], 'y': [patch_y[max_idx]]}
    
    def on_file_select(attr, old, new):
        update_visualization(os.path.join(data_dir, new))

    # --- 5. bind callbacks. layouts ---
    file_selector.on_change('value', on_file_select)
    signal_source.on_change('data', update_heatmap)
    controls = column(file_selector, info_div, width=300)
    plots = row(p_src, p_trg)
    main_layout = row(controls, plots)
    curdoc().add_root(main_layout)
    curdoc().title = "Attention Map Browser"
    
    # --- 6. initialize. ---
    update_visualization(os.path.join(data_dir, npy_files[0]))
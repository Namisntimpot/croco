# vis_attn_browser_final.py
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

# --- 1. 参数解析与文件扫描 ---
parser = argparse.ArgumentParser(description="Attention Map 浏览器")
parser.add_argument("--directory", type=str, required=True, help="包含.npy数据文件的目录路径")
args = parser.parse_args()

data_dir = Path(args.directory)
npy_files = sorted([p.name for p in data_dir.glob("*.npy")])

if not npy_files:
    error_div = Div(text=f"<h3>错误：在目录 '{data_dir}' 中没有找到任何 .npy 文件。</h3>")
    curdoc().add_root(error_div)
else:
    # --- global ---
    similarity_array = None

    def load_image_as_rgba(path):
        '''load image and convert to RGBA, and then flip it vertically'''
        with megfile.smart_open(path, 'rb') as f:
            img = Image.open(f).convert("RGBA")
            img_np = np.array(img).copy().view(np.uint32).reshape(img.height, img.width)
        return np.flipud(img_np)

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
        # ... (此函数无变化)
        global similarity_array
        try:
            info_div.text = f"<b>Loading:</b> {filepath.name}"
            data = np.load(filepath, allow_pickle=True).item()
            src_img_path, trg_img_path = data["source_path"], data["target_path"]
            similarity_array = data["attn_map"]
            src_img = load_image_as_rgba(src_img_path)
            trg_img = load_image_as_rgba(trg_img_path)
            H, W = src_img.shape
            h, w = similarity_array.shape[0:2]
            patch_size_h, patch_size_w = H / h, W / w
            src_source.data = {'image': [src_img]}
            trg_source.data = {'image': [trg_img]}
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
            info_div.text = f"<b>Selected file:</b> {filepath.name}<br><b>resolution:</b> {W}x{H}"
        except Exception as e:
            info_div.text = f"<b>Error when loading .npy file:</b> {filepath.name}<br><pre>{e}</pre>"

    def update_heatmap(attr, old, new):
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
        heatmap_source.data['similarity'] = new_similarity
        max_idx = np.argmax(new_similarity)
        patch_x, patch_y = heatmap_source.data['x'], heatmap_source.data['y']
        max_point_source.data = {'x': [patch_x[max_idx]], 'y': [patch_y[max_idx]]}
    
    def on_file_select(attr, old, new):
        update_visualization(data_dir / new)

    # --- 5. bind callbacks. layouts ---
    file_selector.on_change('value', on_file_select)
    signal_source.on_change('data', update_heatmap)
    controls = column(file_selector, info_div, width=300)
    plots = row(p_src, p_trg)
    main_layout = row(controls, plots)
    curdoc().add_root(main_layout)
    curdoc().title = "Attention Map Browser"
    
    # --- 6. initialize. ---
    update_visualization(data_dir / npy_files[0])
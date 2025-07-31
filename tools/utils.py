import os
import numpy as np
import megfile
from PIL import Image

def load_image_as_rgba(path, h, w, flip_vertially=True):
    '''load image and convert to RGBA, and then flip it vertically to align to Bokeh's coordinate'''
    with megfile.smart_open(path, 'rb') as f:
        img = Image.open(f)
        if h is not None and w is not None:
            img = img.resize((w, h), Image.Resampling.BILINEAR)
        img = img.convert("RGBA")
        img_np = np.array(img).copy().view(np.uint32).reshape(img.height, img.width)
    if flip_vertially:
        return np.flipud(img_np)
    else:
        return img_np
    
def softmax(x, axis=-1):
    '''softmax using numpy.'''
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))

    # 2. 计算指数和
    # 沿着最后一个轴求和。
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)

    # 3. 计算 Softmax 值
    return e_x / sum_e_x

def flip_attn_map(attn_map:np.ndarray):
    return attn_map[::-1, :, ::-1, :]
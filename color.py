import cv2
import imageio.v3 as iio
import numpy as np
import streamlit as st
from streamlit import sidebar as sb


def p(var):
    d = globals()
    s = [f'{k} = {var}' for k in d if d[k] is var][0]
    print(type(s))
    print(s)


def _p(s):
    print(type(s), s)


def rgb2hex(rgb):
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'


def hex2rgb(hexcode):
    return tuple(map(ord, hexcode[1:].decode('hex')))


colors_rgb = np.array(
    [
        [255, 0, 0],
        [255, 100, 0],
        [255, 200, 0],
        [0, 150, 0],
        [0, 100, 255],
        [100, 0, 255],
        [0, 0, 0],
        [255, 255, 255],
    ]
)
colors = [
    'red',
    'orange',
    'yellow',
    'green',
    'blue',
    'purple',
    'black',
    'white',
]


def rgb2ycc(rgb):
    rgb = rgb / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def closest(c, colors_rgb):
    return np.argmin(
        np.sum((rgb2ycc(colors_rgb) - rgb2ycc(c[np.newaxis])) ** 2, axis=1)
    )


for color, rgb in zip(colors, colors_rgb):
    sb.color_picker(f'{color}', value=rgb2hex(rgb))


im = sb.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'])
if im:
    im = iio.imread(im)
    st.image(im, use_column_width=True)
    data = np.float32(np.reshape(im, (-1, 3)))

    rgb = cv2.kmeans(
        data,
        1,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )[2][0].astype(np.int32)

    predict = closest(rgb, colors_rgb)

    avg = st.color_picker(
        colors[predict],
        value=rgb2hex(colors_rgb[predict]),
        key='a',
    )

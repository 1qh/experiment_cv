import cv2
import imageio.v3 as iio
import numpy as np
import streamlit as st
from streamlit import sidebar as sb


def p(var):
    d = globals()
    s = [f'{k} = {var}' for k in d if d[k] is var][0]
    print(s)


def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def hex2rgb(hexcode):
    return tuple(map(ord, hexcode[1:].decode('hex')))


colors = {
    (196, 2, 51): 'red',
    (255, 125, 0): 'orange',
    (255, 205, 0): 'yellow',
    (0, 128, 0): 'green',
    (0, 120, 200): 'blue',
    (127, 0, 255): 'violet',
    (0, 0, 0): 'black',
    (255, 255, 255): 'white',
}

for color in colors:
    sb.color_picker(f'{colors[color]}', value=rgb2hex(*color))


def rgb2ycc(color):
    r, g, b = [i / 255.0 for i in color]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def dist(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(rgb2ycc(c1), rgb2ycc(c2)))


def closest(c, colors):
    return min(
        (
            dist(c, i),
            colors[i],
        )
        for i in colors
    )


im = sb.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'])
if im:
    im = iio.imread(im)
    st.image(im, use_column_width=True)
    data = np.float32(np.reshape(im, (-1, 3)))

    r, g, b = cv2.kmeans(
        data,
        1,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )[2][0].astype(np.int32)

    predict = closest((r, g, b), colors)[1]
    avg = st.color_picker(predict, value=rgb2hex(r, g, b))

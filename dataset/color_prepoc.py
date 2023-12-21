# Pre-processing color images

import os
import cv2
import numpy as np


__all__ = [
    "get_dctmtx",
    "dct8",
    "idct8",
    "idct8v2",
    "decode_from_dct"
]

def get_dctmtx():
    [Col, Row] = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2 * Col + 1) * Row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    return T.astype(np.float32)

# DCTMTX = np.array(
#     [
#         [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
#         [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
#         [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
#         [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
#         [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
#         [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
#         [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
#         [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975],
#     ],
#     dtype=np.float32,
# )

DCTMTX = get_dctmtx()


def dct8(image):
    assert image.shape[0] % 8 == 0
    assert image.shape[1] % 8 == 0
    dct_shape = (image.shape[0] // 8, image.shape[1] // 8, 64)
    dct_image = np.zeros(dct_shape, dtype=np.float32)

    one_over_255 = np.float32(1.0 / 255.0)
    image = image * one_over_255
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            # dct = cv2.dct(image[i : i + 8, j : j + 8])
            dct = DCTMTX @ image[i : i + 8, j : j + 8] @ DCTMTX.T
            dct_image[i // 8, j // 8, :] = dct.flatten()

    return dct_image

def idct8(dct):
    assert dct.shape[2] == 64
    dct_image = np.zeros((dct.shape[0] * 8, dct.shape[1] * 8), dtype=np.float32)

    for i in range(0, dct.shape[0]):
        for j in range(0, dct.shape[1]):
            img = DCTMTX.T @ dct[i, j].reshape((8, 8)) @ DCTMTX
            dct_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = img

    return dct_image

def idct8v2(dct, qm=None):
    decoded_image = np.zeros((dct.shape[0], dct.shape[1], 1), dtype=np.float32)

    def idct2(a):
        # import scipy
        from scipy import fftpack

        return fftpack.idct(fftpack.idct(a, axis=0, norm="ortho"), axis=1, norm="ortho")

    if qm is None:
        for i in range(0, dct.shape[0] // 8):
            for j in range(0, dct.shape[1] // 8):
                img = idct2(dct[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8])
                # img = DCTMTX.T @ (dct[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]) @ DCTMTX
                decoded_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8, 0] = img
    else:
        for i in range(0, dct.shape[0] // 8):
            for j in range(0, dct.shape[1] // 8):
                img = DCTMTX.T @ (qm * dct[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]) @ DCTMTX
                decoded_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8, 0] = img

    return decoded_image

def decode_from_dct(dct_file, img_comp='rgb'):

    assert img_comp in ['rgb', 'ycrcb'], "img_comp should be in ['rgb', 'ycrcb']"
    dct = np.load(dct_file)
    dct_y = dct["dct_y"]
    dct_cr = dct["dct_cr"]
    dct_cb = dct["dct_cb"]

    y = idct8v2(dct_y)
    cr = idct8v2(dct_cr)
    cb = idct8v2(dct_cb)

    y += 127.5
    y /= 255.0

    cr += 127.5
    cr /= 255.0

    cb += 127.5
    cb /= 255.0

    img = np.dstack([y, cr, cb])
    if img_comp == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    return img

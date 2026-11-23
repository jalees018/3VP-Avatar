# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch


# Projection and transformation matrix helpers.
def projection(focal, w, f, n):
    return np.array(
        [
            [2 * focal / w, 0, 0, 0],
            [0, focal / w, 0, 0],
            [0, 0, (f + n) / (f - n), (2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ]
    ).astype(np.float32)


def projection2(focalx, focaly, w, h, f, n, res=512):
    return torch.tensor(
        [
            [2 * focalx / res, 0, (res - 2 * w) / res, 0],
            [0, 2 * focaly / res, (-res + 2 * h) / res, 0],
            [0, 0, (f + n) / (f - n), (2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ]
    ).type(torch.float32)


def translate2(x, y, z):
    return torch.tensor([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]).type(torch.float32)


# Image display function using OpenGL.

_glfw_window = None


def display_image(image, zoom=None, size=None, title=None):  # HWC
    # Import OpenGL and glfw.
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image)
    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = "Debug window"
    global _glfw_window
    if _glfw_window is None:
        glfw.init()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {"uint8": gl.GL_UNSIGNED_BYTE, "float32": gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

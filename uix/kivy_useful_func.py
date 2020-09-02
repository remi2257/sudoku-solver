import cv2
from kivy.graphics.texture import Texture


# noinspection PyArgumentList
def convert_opencv_to_texture(frame):
    if frame is None:
        return Texture()
    buf1 = cv2.flip(frame, 0)
    buf = buf1.tostring()
    image_texture = Texture.create(
        size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    return image_texture


def convert_texture_to_opencv(texture):
    pixels = texture.pixels
    img_opencv = cv2.flip(pixels, 0)

    return img_opencv

import cv2
import math
import random
import os
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageFilter
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
from itertools import chain

class ComputerTextGenerator(object):
    @classmethod
    def generate(cls, text, font, text_color, height):
        # print(text, font, text_color)
        # image_font = ImageFont.truetype(font="/Library/Fonts/Arial Unicode.ttf", size=32)
        image_font = ImageFont.truetype(font=font, size=height)
        text_width, text_height = image_font.getsize(text)

        # text = u'日産コーポレート/個人ゴールドJBC123JAL'
        txt_img = Image.new('L', (text_width, text_height), 255)

        txt_draw = ImageDraw.Draw(txt_img)

        txt_draw.text((0, 0), u'{0}'.format(text), fill=random.randint(1, 80) if text_color < 0 else text_color, font=image_font)

        return txt_img

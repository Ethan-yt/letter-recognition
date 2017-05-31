# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import random

'''
是否随机生成 True 为随机生成，False 为顺序生成
'''
is_random = False
'''
字体总数
'''
font_number = 6


class RandomChar():
    @staticmethod
    def rand_letter():
        # return str(random.randint(0, 9))
        return chr(random.randint(65, 90))


class ImageChar():
    def __init__(self, font_color=(0, 0, 0),
                 size=(28, 28),
                 bg_color=(255, 255, 255),
                 font_size=20):
        self.size = size
        self.fontPath = []
        for i in range(font_number):
            self.fontPath.append('./fonts/{0}.ttf'.format(i))

        self.bgColor = bg_color
        self.fontSize = font_size
        self.fontColor = font_color
        self.font = []
        for i in range(len(self.fontPath)):
            self.font.append(ImageFont.truetype(self.fontPath[i], self.fontSize))
        self.image = None

    def draw_letter(self, letter, font_index):

        (letterWidth, letterHeight) = self.font[font_index].getsize(letter)
        self.image = Image.new('RGB', (letterWidth, letterHeight), self.bgColor)
        draw = ImageDraw.Draw(self.image)
        draw.text((0, 0), letter, font=self.font[font_index], fill=self.fontColor)
        self.image = self.image.resize(self.size)
        del draw

    def save(self, path):
        self.image.save(path)

    def test(self,font_index):
        self.image = Image.new('RGB', (28, 28), self.bgColor)
        draw = ImageDraw.Draw(self.image)
        draw.text((0, 0), "A", font=self.font[font_index], fill=self.fontColor)

if is_random:
    # 生成训练集
    for i in range(100):
        ic = ImageChar()
        fontType = random.randint(1, 7)
        char = RandomChar().rand_letter()
        ic.draw_letter(char, fontType)
        ic.save('./images/train/' + char + '_' + str(fontType) + ".jpeg")
    # 生成测试集
    for i in range(26):
        ic = ImageChar()
        fontType = 0
        char = RandomChar().rand_letter()
        ic.draw_letter(char, fontType)
        ic.save('./images/test/' + char + '_' + str(fontType) + ".jpeg")

else:
    for char in (chr(i + 65) for i in range(26)):
    # for char in (str(i) for i in range(10)):
        for fontType in range(1, font_number):
            ic = ImageChar()
            ic.draw_letter(char, fontType)
            ic.save('./images/train/' + char + '_' + str(fontType) + ".jpeg")
        for fontType in range(0, 1):
            ic = ImageChar()
            ic.draw_letter(char, fontType)
            ic.save('./images/test/' + char + '_' + str(fontType) + ".jpeg")
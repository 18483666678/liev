from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from numpy import random

def randomChar():
    """随机生成0——9的数字"""
    return chr(random.randint(48,57))

def randomBgColor():
    """随机生成验证码背景色"""
    return (random.randint(50,100),random.randint(50,100),random.randint(50,100))

def randomTexrColor():
    """随机生成验证码文字颜色"""
    return (random.randint(120,200),random.randint(120,200),random.randint(120,200))

width = 30 * 4
height = 60

#设置字体类型及大小
font = ImageFont.truetype(font=r"D:\PycharmProjects\tensorflowlianxi\arial.ttf",size=36)

for i in range(1000):
    #创建一张图片，指定图片mode，长宽
    image = Image.new("RGB",(width,height),(255,255,255))

    #创建Draw对象
    draw = ImageDraw.Draw(image)
    #遍历给图片的每个像素点着色
    for x in range(width):
        for y in range(height):
            draw.point((x,y),fill=randomBgColor())

    #建立文件名 将随机生成的chr，draw加入image
    filename = []
    for t in range(4):
        ch = randomChar()
        filename.append(ch)
        draw.text((30 * t,10),ch,font=font,fill=randomTexrColor())

    #保存图片
    image_path = r"D:\PycharmProjects\tensorflowlianxi\code"
    image.save("{0}/{1}.jpg".format(image_path,"".join(filename)))
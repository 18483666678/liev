from PIL import Image
from PIL import ImageDraw,ImageFont
import random #python自带的随机

#随机字母
def randomChar():
    return chr(random.randint(65,90)) #randint生成整数

#随机颜色1
def randomColor1():
    return random.randint(64,255),random.randint(64,255),random.randint(64,255)

#随机颜色2
def randomColor2():
    return random.randint(32,127),random.randint(21,127),random.randint(21,127)

#生成白板 240*60
width = 240
height = 60
image = Image.new("RGB",(width,height),(255,255,255)) #白板

#创建font对象
font = ImageFont.truetype("arial.ttf",size=36)

#创建Draw对象
draw = ImageDraw.Draw(image)

#填充像素
for x in range(width):
    for y in range(height):
        draw.point((x,y),fill=randomColor1()) #填充像素点

#填充文字
for i in range(4):
    draw.text((60*i+10,10),randomChar(),fill=randomColor2(),font=font)

image.show()
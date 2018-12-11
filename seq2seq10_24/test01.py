import random
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw
from PIL import ImageFont

#生成500张四位数字验证码

def randomChar():
    """随机生成chr，return返回一个随机生成的chr，chr对应的是ASCII可显示字符表"""
    return chr(random.randint(48,57))  #随机生成0-9的数字

def randomBgColor():
    """随机生成验证码的背景色,return"""
    return (random.randint(50,100),random.randint(50,100),random.randint(50,100))

def randomTextColor():
    """随机生成验证码的文字颜色，return"""
    return (random.randint(120,200),random.randint(120,200),random.randint(120,200))

w = 30 * 4
h = 60

#设置字体类型及大小
font = ImageFont.truetype(font="arial.ttf",size=36)

for _ in range(500):
    #创建一张图片，指定图片mode，长度
    image = Image.new("RGB",(w,h),(255,255,255)) #白色背景（255,255,255）

    #创建Draw对象
    draw = ImageDraw.Draw(image)
    #遍历给图片的每个像素点着色
    for x in range(w):
        for y in range(h):
            draw.point((x,y),fill=randomBgColor())

    #将随机生成的chr，draw入image
    filename = []
    for t in range(4):
        ch = randomChar()
        filename.append(ch)
        draw.text((30 * t,10),ch,fill=randomTextColor(),font=font)

    #设置图片模糊
    #image = image.filter（ImageFilter，BLUR）
    #保存图片
    image.save("D:\PycharmProjects\seq2seq10_24\code/{0}.jpg".format("".join(filename)),'jpeg')


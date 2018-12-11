from PIL import Image

# im1 = Image.open("3.jpg")
# im_resize1 = im1.resize((1920,1080))
# im2 = Image.open("5.jpg")
# # im_resize2 = im2.resize((256,256))
# im = Image.blend(im_resize1,im2,0.6)
#
# r,g,b = im1.split()
# print(g.mode)

img = Image.open("1.jpg")
# img.show()
# w,h = img.size  #获取图片长宽
# print(w,h)
# img.thumbnail((w//2,h//2)) #缩小 整除 避免取到小数
img1 = img.resize((500,500)) #重新定义图像尺寸
img1.save("test.jpg","jpeg") #保存
img = img1.rotate(90) #旋转 角度自己调
# img.show()

from PIL import ImageFilter

img3 = Image.open("2.jpg")
# img3.show()
# img3 = img3.filter(ImageFilter.DETAIL)  #滤波器
# img = img3.convert("L") #灰度化  “RGB"--》”L“
# img.show()
# bands = img3.getbands()  #返回通道值“RGB”
# print(bands)
# pixes = img3.getpixel((250,400)) #获取图片的像素
# print(pixes)

#返回图片的像素直方图
# pr = img3.histogram()  #像素直方图 是符合正态分布的
# print(pr)

img4 = Image.open("4.jpg")
img3.paste(img4,(300,300)) #加水印合成  后面是粘贴位置
img3.show()
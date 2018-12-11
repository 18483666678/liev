import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

#画sin和cos的图
# x = np.linspace(0,2*np.pi,100)
# y1,y2 = np.sin(x),np.cos(x)
#
# plt.title("sin&cos")  #标题名称
# plt.xlabel("x")  #X轴名称
# plt.ylabel("y")  #Y轴名称
#
# plt.plot(x,y1) #画点
# plt.plot(x,y2)
# plt.show()

#画条形图
# name_list = ["A","B","C","D"]
# num_list = [1.5,0.6,7.8,6]
#
# plt.bar(range(len(name_list)),num_list,color="rgb",tick_label=name_list)
# plt.show()

#堆叠柱状图
# name_list = ["A","B","C","D"]
# num_list = [1.5,0.6,7.8,6]
# num_list1 = [1,2,3,1]
# plt.bar(range(len(num_list)),num_list,label="boy",fc="y")
# plt.bar(range(len(name_list)),num_list1,label="girl",fc="r")
# plt.legend() #堆叠
# plt.show()

#并列柱状图
# name_list = ["A","B","C","D"]
# num_list = [1.5,0.6,7.8,6]
# num_list1 = [1,2,3,1]
# x = list(range(len(name_list)))
# totle_width,n = 0.8,2
# width = totle_width/n
# plt.bar(x,num_list,width=width,label="boy",fc="y")
# for i in range(len(x)):
#     x[i] = x[i]+width
# plt.bar(x,num_list1,width=width,label="girl",fc="r")
# plt.legend()
# plt.show()

#饼状图
# labels = "A","B","C","D"
# faces = [15,30.55,44.44,10]
# plt.axes(aspect=1) #单位圆
# explode = [0,0.1,0,0]
# plt.pie(x=faces,labels=labels,autopct="%3.1f %%",shadow=True,startangle=90,explode=explode)
# plt.show()

#3D图
# x = np.random.normal(0,1,100)
# y = np.random.normal(0,1,100)
# z = np.random.normal(0,1,100)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x,y,z)
# plt.show()


#实时画图
ax = []
ay = []
plt.ion() #开始之前用ion（）
for i in range(100):
    ax.append(i)
    ay.append(i**2)
    plt.clf() #清除上一次图
    plt.plot(ax,ay) #用plot自动画图 不需要show（）
    plt.pause(1)
plt.ioff()



#用PIL读取图片，plt查看图片
im = Image.open(r"D:\PycharmProjects\Image\PIL\5.jpg")
plt.imshow(im)
# plt.axis("off") #关掉坐标轴为off
plt.show()

# im = Image.open(r"D:\PycharmProjects\Image\PIL\2.jpg")
# plt.figure("Image") #图像窗口名称
# plt.imshow(im)
# plt.axis("on") #关掉坐标轴为off
# plt.title("image") #图像题目
# plt.show()

#
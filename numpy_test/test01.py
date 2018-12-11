import numpy as np
import numpy.random as npr

print(np.__version__)

print("使用列表生成一维数组")
data = [1,2,3,4,5,6]
x = np.array(data)
print(x)
print(type(x))   #输出class 'numpy.ndarray'是numpy的类型
print(x.dtype)

print("使用列表生成二维数组")
data = [[1,2],[3,4],[5,6]]
x = np.array(data)
print(x)
print(x.ndim)   #查看轴个数 看有几个方括号
print(x.shape)   #查看数组的维度
print(x.size)   #查看元素总数

print("使用zreo/ones/empty创建数组，通过shape来创建")
x = np.zeros((2,3),dtype=np.int32)   #添加类型dtype=
print(x)
print(x.dtype)

x = np.ones((2,3))
print(x)

x = np.empty((3,2))  #创建空数组
print(x)

print("使用arrange生成连续的元素")
print(np.arange(6))
print(np.arange(1,6,2))

print("使用astype复制数组，并转换类型")
x = np.array(np.arange(6),dtype=np.float32)
y = x.astype(np.int32)
print(x)
print(y)

print("将字符串元素转换为数值元素")
x = np.array(['1','2','3','4'],dtype=np.string_)  #numpy的字符串表示np.string_
print(x)
print(x.dtype)
y = x.astype(np.int32)  #用.astype转换类型
print(y)
print(y.astype(x.dtype))

print("ndarray数组与标量/数组的运算")
x = np.array([1,2,3])
print(x*2)
print(x>2)  #逻辑运算符输出的是布尔结果
y = np.array([3,4,5])
print(x*y)
print(x>y)

print("ndarray的基本索引")
x = np.array([[1,2],[3,4],[5,6]])
print(x[0])
print(x[0][0])
print(x[0,0])
x = np.array([[1,2],[3,4],[5,6],[7,8]])
print(x.shape)
print(x[0])
y = x[0].copy() #生成一个副本  原元素不变
z = x[0]   #未生成副本
print(y)

print("ndarray的切片")
x = np.array([1,2,3,4,5])
print(x[1:3])
print(x[:3])
print(x[1:])
print(x[:])
print(x[:-1])
print(x[0:4:2])
x = np.array([[1,2],[3,4],[5,6]])
print(x[:2])
print(x[:2,:1])  #加逗号是每个元素的索引
print(x[:2][:1])
# x[:2,:1] = 0
# print(x)
x[:2][:1] = 0
print(x)
x[:3,:1] = [[8],[7],[6]]
print(x)

print("ndarray的布尔型索引")
x = np.array([3,2,3,1,3,0])
y = np.array([True,True,True,False,True,False])
print(x[y])  #只输出True的值
print(x[y==False])
print(x>=3)
print(x[(x==2) | (x==1)])
x[(x==2) | (x==1)] = 0
print(x)

print("ndarry的花式索引：使用整形数组作为索引")
x = np.array([1,2,3,4,5,6])
print(x[[0,1,4]])
print(x[[-1,-2,-3]])
x = np.array([[1,2],[3,4],[5,6]])
print(x[[0,1]])
print(x[[[0,1],[0,1]]])
print(x[[0,1]][:,[0]])
x[[0,1],[0,1]] = 9,8
print(x)

print("ndarray数组的装置和轴对称")
k = np.arange(9)
m = k.reshape((3,3))
print(m)
#转置
print(m.T)
#内积（点乘）
print(np.dot(m,m.T))
#高维数组的轴对象
k = np.arange(24).reshape(2,3,4)
print(k)
print(k[0][1][0])
#轴变换
m = k.transpose((0,2,1))
#m = k.swapaxes(1,2)
print(m)
#轴交换做转置
m = np.arange(8).reshape(2,4)
print(m)
print(m.swapaxes(1,0))
print(m.T)

print("不重要 一元ufunc示例")
x = np.arange(6)
print(x)
print(np.square(x))  #square平方
x = np.array([1.5,1.6,1.7,1.8])
y,z = np.modf(x)   #分解整数和小数
print(y)
print(z)

print("二元ufunc示例")
x = np.array([[1,4],[6,7]])
y = np.array([[2,4],[5,8]])
print(np.maximum(x,y)) #比较最大值
print(np.minimum(x,y))
print(np.max(x))  #最大值

print("where函数的使用")
cond = np.array([True,False,False,False])
x = np.where(cond,-2,2)  #-2表示True，2表示False
print(x)
cond = np.array([1,2,3,4])
x = np.where(cond>2,-2,2)
print(x)

print("numpy的基本统计方法")
x = np.array([[1,2],[3,4],[5,6]])
print(x)
print(x.shape)
print(x.mean())  #求均值
print(x.mean(axis=1))  #求行的均值
print(x.mean(axis=0))   #求列的均值
print(x.sum())  #求和
print(x.sum(axis=1))
print(x.max())
print(x.max(axis=1))

print(".sort就地排序")
x = np.array([[6,1,3],[1,5,2]])
x.sort(axis=0)
print(x)

print("ndarray的存取")
x = np.array([[1,6,2],[6,1,3],[1,5,2]])
np.save("file",x)  #以二进制 .npy保存
y = np.load("file.npy")
print(y)

print("矩阵求逆")
x = np.array([[1,1],[1,2]])
y = np.linalg.inv(x)
print(y)
print(x.dot(y))

a = np.arange(6).reshape(6,1)
print(a)
b = np.arange(0,5)
print(b)
c = a+b
print(c)

print("numpy随机数")
x = npr.randint(0,2,size=100000) #抛硬币
print((x>0).sum())  #正面朝上的次数

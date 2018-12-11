import pandas as pd
from pandas import Series,DataFrame
from pandas import Index
import numpy

print("用一维数组生成Series")
x = Series([1,2,3,4])
print(x)  #左边为索引 右边为元素
print(x.values)  #查看元素
print(x.index)   #查看索引 (start=0, stop=4, step=1)0开始，4结尾 取不到4，步长为1

print("指定Series的index") #可将index理解为行索引
x = Series([1,2,3,4],index=['a','b','c','d']) #手动指定，改变索引
print(x)
print(x.index)
print(x['c'])  #索引取值
x['d'] = 6
print(x)
print(x[['c','a','d']])  #多个索引
print(x[x>2]) #bool型索引

print("使用字典生成Series")
data = {'a':1,'b':2,'d':3,'c':4}
x = Series(data)
print(x)

print("使用字典生成DataFrame，key为列名")
data = {'state':['ok','yes','hello','good'],
        'year':[True,False,True,True],
        'num':[1,2,3,4]}
x = DataFrame(data)
print(x)

print("使用字典来生成Series，并指定额外的index，不匹配的索引部分为NaN")
data = {'a':1,'b':2,'d':3,'c':4}
exindex = ['a','b','c','e']
y = Series(data,index=exindex)
print(y)

print("Series相加，相同行索引相加，不同行索引则为NaN")
data = {'a':1,'b':2,'d':3,'c':4}
x = Series(data)
exindex = ['a','b','c','e']
y = Series(data,index=exindex)
print(x+y)

print("指定Series/索引的名字")
exindex = ['a','b','c','e']
y = Series(data,index=exindex)
y.name = '对象一'
y.index.name = '字母'
print(y)

print("替换index")
exindex = ['a','b','c','e']
y = Series(data,index=exindex)
y.name = '对象一'
y.index = ['a','b','c','f']
print(y)

print("使用字典生成DataFrame，key为列名")
data = {'state':['ok','yes','hello','good'],
        'year':[2016,2018,2001,2005],
        'num':[12.1,31.2,28.3,9.4]}
x = DataFrame(data)
print(x)

print("指定列索引columns，不匹配的列为NaN")
data = {'state':['ok','yes','hello','good'],
        'year':[2016,2018,2001,2005],
        'num':[12.1,31.2,28.3,9.4]}
print(DataFrame(data,columns=['state','year','num','pop']))

print("指定行索引index")
x = DataFrame(data,columns=['state','year','num','pop'],
              index=['one','two','three','four'])
print(x)

print("DataFrame元素的索引与修改")
print(x['state']) #索引
print(x.state) #属性
print(x.ix['three']) # .ix行调用
x['state'] = 3.2222
print(x)

print("获取index对象")
x = Series(range(3),index=['a','b','c'])
index = x.index
print(index)
print(index[0:2])  #切片

print("构造/使用Index对象")
index = Index(numpy.arange(3))
obj2 = Series([1.5,-2.1,0],index=index)
print(obj2)
print(obj2.index is index)  #判断index对象是不是index

print("判断行/列索引是否存在")
data = {'pop':[2.4,2.9],
        'year':[2001,2002]}
x = DataFrame(data)
print(x)
print('pop' in x.columns)
print(8 in x.index)

print("重新指定索引，以及对NaN值填充")
x = Series([4,7,6],index=['a','b','c'])
y = x.reindex(['a','b','c','d'],fill_value=0) #填充空值d fill_value=0
print(y)

print("重新指定索引，并指定填充NaN的方法")
x = Series(['blue','yellow'],index=[0,2])
print(x)
print(x.reindex(range(4),method='ffill')) #平均填充method='ffill'
print(x.reindex(range(4)))

print("Series根据行索引删除行")
x = Series(numpy.arange(4),index=['a','b','c','d'])
print(x)
print(x.drop('c'))
print(x.drop(['a','b']))

print("DataFrame根据索引行/列删除行/列")
x = DataFrame(numpy.arange(16).reshape(4,4),
              index=['a','b','c','d'],
              columns=['A','B','C','D'])
print(x)
print(x.drop(['A','B'],axis=1)) #在列的维度上删除A,B两行
print(x.drop('a',axis=0))  #axis= 1是列，0是行

print("Series的数组/字典索引")
x = Series(numpy.arange(4),index=['a','b','c','d'])
print(x)
print(x['b'])  #像字典一样索引
print(x[1])  #像数组一样索引 0123对应abcd
print(x[[1,3]])  #取1和3
print(x[x<2])

print("Series的数组切片")
print(x['a':'c'])  #['a':'c']的切片包含’c'
print(x[0:3])  #x[0:3]只取到2
x['a','c'] = 5
print(x)

print("DataFrame的索引")
data = DataFrame(numpy.arange(16).reshape(4,4),
                 index=['a','b','c','d'],
                 columns=['A','B','C','D'])
print(data)
print(data['A'])
print(data[['A','C']])
print(data[:2])
print(data.ix[:2,['A','B']])
print(data.ix[['a','c'],[3,0,1]])  #[3,0,1]表示ABCD的索引0123变动301DAB

print("DataFrame算术，不重叠部分为NaN,重叠部分元素做运算")
x = DataFrame(numpy.arange(9).reshape(3,3),
              index=['a','b','c'],
              columns=['A','B','C'])
y = DataFrame(numpy.arange(12).reshape(4,3),
              index=['a','b','c','d'],
              columns=['A','B','C'])
print(x)
print(y)
print(x+y)

print("对x+y不重叠部分填充，不是对结果NaN填充")
print(x.add(y,fill_value=10))  #d  9  10  11加上fill_value=10
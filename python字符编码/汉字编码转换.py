#coding=utf-8

s1='中'
print("直接用ord()显示中文在python中存储时的编码：",hex(ord(s1))) #结果是二个字节，表示中文在python内部用unicode编码方式
#ds=s1.encode("ascii") #对s进行编码,编码方式为ascii，但对中文这里会失败，因为无法将中文转为ascii码
ds=s1.encode() #对s1进行编码，这里等同于encode('utf-8'),因为设过了#coding=utf-8
print("ds变量类型：",type(ds)) #ds为bytes类型
print("显示中文的utf-8编码：",ds,"长度：",len(ds))

s2=ds.decode() #对bytes类型变量ds解码，重新还原汉字(即还原为unicode)
print("s2:",s2,"长度:",len(s2),"ord()编码：",hex(ord(s2)))

a=15 #0f
b=255 #ff
print(a^b,"%#x"%(a^b))
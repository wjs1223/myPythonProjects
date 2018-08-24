#coding=utf-8

with open('密文.dat','rb') as file_object:
    miBytes = file_object.read()
    #print(miBytes)

i=0
key=['1','2','3']
mingByteList=[]
for b in miBytes:    
    mingByteList.append(b ^ ord(key[i]))
    i+=1
    if(i>=len(key)):
        i=0
mingBytes=bytes(mingByteList)
mingContents=mingBytes.decode()

with open('明文.txt','w') as file_object:
    file_object.write(mingContents)
    print("已保存明文文件！")


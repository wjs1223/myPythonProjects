#coding=utf-8

with open('原文.txt') as file_object:
    contents = file_object.read()
    #print(type(contents))
    #print(contents)

ec=contents.encode('utf-8')
#print(ec)

i=0
key=['1','2','3']
miByteList=[]
for b in ec:    
    miByteList.append(b ^ ord(key[i]))
    i+=1
    if(i>=len(key)):
        i=0
#print(miByteList)
miBytes=bytes(miByteList)

with open('密文.dat','wb') as file_object:
    file_object.write(miBytes)
    print("已保存加密文件！")




'''
将生成的数据集转换成yolox需要的形状
'''
import os
import shutil
import random

if __name__ == "__main__":
    train = './train/'
    os.makedirs('./VOCdevkit/VOC2007/Annotations')
    os.makedirs('./VOCdevkit/VOC2007/JPEGImages')
    os.makedirs('./VOCdevkit/VOC2007/ImageSets/Main')

    for file in os.listdir(train):
        print(file)
        if file.split('.')[-1] == 'xml':
            shutil.copy(train + file,
            './VOCdevkit/VOC2007/Annotations/' + file)
        else:
            shutil.copy(train + file,
            './VOCdevkit/VOC2007/JPEGImages/' + file)


    trainval_percent = 0.1
    train_percent = 0.9
    xmlfilepath = './VOCdevkit/VOC2007/Annotations'
    
    total_xml = os.listdir(xmlfilepath)

    num = len(os.listdir(xmlfilepath))
    llist = range(num)
    tv = int(num * trainval_percent)
    
    trainval = random.sample(llist, tv)
    

    ftest = open('./VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
    ftrain = open('./VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')

    for i in llist:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftest.write(name)
        else:
            ftrain.write(name)

    ftrain.close()
    ftest.close()

'''
已经是yolox的形状了
'''
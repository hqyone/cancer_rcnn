import json
import os
import cv2
import numpy as np
'''
由点到线再到面
'''
'''
json文件标注的地方是一些坐标
首先获得这些点  get_region(points)
'''
def get_region(points):
    regions=[]
    for  i  in  range(len(points)):
        if i==(len(points)-1):
            region=get_line(points[i],points[0])
        else:
            region=get_line(points[i],points[i+1])
        regions=regions+region
    return regions


def get_line(point1,point2):
    point1=[int(point1[0]),int(point1[1])]
    point2=[int(point2[0]),int(point2[1])]
    points=[]
    # point=[0,0]
    if (point2[0]-point1[0])==0:
        for i in range(min(point2[1], point1[1]),max(point2[1], point1[1])+1):
            x= point1[0]
            point = [x, i]
            points.append(point)
    else:
        k=(point2[1]-point1[1])/(point2[0]-point1[0])
        if abs(point2[1]-point1[1])>abs(point2[0]-point1[0]):
            for i in range(min(point2[1], point1[1]), max(point2[1], point1[1])+1):
                x=(i-point1[1])/k +point1[0]
                point = [x, i]
                points.append(point)
        else:
            for i in range(int(min(point2[0], point1[0])), int(max(point2[0], point1[0]))+1):
                y = k * (i - point1[0]) + point1[1]
                point = [i, y]
                points.append(point)
    return points

def  get_area_allpoint(one_region):
    allpoint=[]
    angepoint=[]
    maxx=one_region[0][1]
    minn=one_region[0][1]
    for point in one_region:
        if point[1]>maxx:
            maxx=int(point[1])
        if point[1]<minn:
            minn=int(point[1])
    for m in range(int(minn),int(maxx)):
        for point2 in one_region:
            if(m==int(point2[1])):
                angepoint.append(int(point2[0]))
        for i  in  range(min(angepoint),max(angepoint)+1):
            allpoint.append([i,m])
        angepoint.clear()
    return allpoint


def  get_json(json_path):
    paths=[]
    for root, dir, file in os.walk(json_path):
        for f in file:
            if '.json' in f:
                paths.append(os.path.join(root, f))
    return paths

# Gray = R*0.299 + G*0.587 + B*0.114
if __name__ == '__main__':
    paths=get_json(r'D:\\Desktop\\lungcancerdataset\\json&image')
    outpath=r'D:\\Desktop\\lungcancerdataset\\json&image\\stage1_train'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for j,path in enumerate(paths):
        #srcImg = cv2.imread(path.split('.')[0] + '.jpg', 0)  灰度图
        srcImg = cv2.imread(path.split('.')[0] + '.png')
        print('srcImg:',path.split('.')[0]+'.png')
        wbool=True
        with open(path, 'rb') as f:
            json_file = json.load(f)
            shapes = json_file.get('shapes')
        #with  open(path,'b') as f:  #  不能  ’gbk‘
            os.mkdir(os.path.join(outpath,str(j)))
            os.mkdir(os.path.join(outpath,str(j)+'/images'))
            os.mkdir(os.path.join(outpath,str(j)+'/masks'))
            print(path.split('.')[0]+'.png')
            outsrc=cv2.resize(srcImg,(1024,1024))
            cv2.imwrite(os.path.join(outpath,str(j)+'/images/'+str(j)+'.png'),outsrc)
            for i,mark in enumerate(shapes):
                out=np.zeros((1200,1920),dtype=np.uint8)
                one_region=get_region(mark.get('points'))
                # print(mark.get('label'))
                allpoint=get_area_allpoint(one_region)
                for p in allpoint:
                    out[min(p[1], 1199), min(p[0], 1919)] = 255
                out = cv2.resize(out, dsize=(1024, 1024))
                class_name = mark.get('label').lower()

                # 宫颈癌分类标签
                if class_name == 'yin':
                    lab = 1
                if class_name == 'yin-yang':
                    lab = 2
                elif class_name == 'yang':
                    lab = 3

                #肺癌分类标签
                # if class_name=='h-positive':
                #     lab = 1
                # elif class_name=='b-positive':
                #     lab = 2
                # elif class_name=='t-positive':
                #     lab = 3
                # 活细胞分类标签
                # if class_name == 'xf-xb':
                #     lab = 1
                # elif class_name == 'tb-xb':
                #     lab = 2
                cv2.imwrite(os.path.join(outpath, str(j) + '/masks/' + str(i) + '_' + str(lab) + '.png'), out)










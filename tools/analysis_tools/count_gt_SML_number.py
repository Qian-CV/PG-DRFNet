# 1、统计数据集中小、中、大 GT的个数
# 2、统计某个类别小、中、大 GT的个数
# 3、统计数据集中ss、sm、sl GT的个数
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from mmrotate.structures.bbox import qbox2rbox

# 设置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def getGtAreaAndRatio(label_dir):
    """
    得到不同尺度的gt框个数
    :params label_dir: label文件地址
    :return data_dict: {dict: 3}  3 x {'类别':{’area':[...]}, {'ratio':[...]}}
    """
    data_dict = {}
    class_list = []
    assert Path(label_dir).is_dir(), "label_dir is not exist"

    txts = os.listdir(label_dir)  # 得到label_dir目录下的所有txt GT文件

    for txt in txts:  # 遍历每一个txt文件
        with open(os.path.join(label_dir, txt), 'r') as f:  # 打开当前txt文件 并读取所有行的数据
            lines = f.readlines()

        if len(lines) == 0:
            continue
        if lines[0].split()[0][0:5] == 'image':
            lines = lines[2:]
        for line in lines:  # 遍历当前txt文件中每一行的数据
            temp = line.split()  # str to list{5}
            coor_list = list(map(lambda x: float(x), temp[:8]))  # [x, y, w, h]
            coor_torch = torch.tensor(coor_list)
            rbox = qbox2rbox(coor_torch)
            area = rbox[2] * rbox[3]  # 计算出当前txt文件中每一个gt的面积
            # center = (int(coor_list[0] + 0.5*coor_list[2]),
            #           int(coor_list[1] + 0.5*coor_list[3]))
            ratio = rbox[2] / rbox[3]  # 计算出当前txt文件中每一个gt的 w/h

            if temp[-2] not in data_dict:
                class_list.append(temp[-2])
                data_dict[temp[-2]] = {}
                data_dict[temp[-2]]['area'] = []
                data_dict[temp[-2]]['ratio'] = []

            data_dict[temp[-2]]['area'].append(area)
            data_dict[temp[-2]]['ratio'].append(ratio)

    return data_dict, class_list


def getSMLGtNumByClass(data_dict, class_num):
    """
    计算某个类别的小物体、中物体、大物体的个数
    params data_dict: {dict: 3}  3 x {'类别':{’area':[...]}, {'ratio':[...]}}
    params class_num: 类别  0, 1, 2
    return s: 该类别小物体的个数  0 < area <= 0.5%
           m: 该类别中物体的个数  0.5% < area <= 1%
           l: 该类别大物体的个数  area > 1%
    """
    ts, s, m, l = 0, 0, 0, 0
    # 图片的尺寸大小 注意修改!!!
    h = 1024
    w = 1024
    for item in data_dict[class_num]['area']:
        if item <= 16 ** 2:
            ts += 1
            s += 1
        elif 16 ** 2 < item <= 32 ** 2:
            s += 1
        elif 32 ** 2 < item <= 96 ** 2:
            m += 1
        else:
            l += 1
    return ts, s, m, l


def getAllSMLGtNum(data_dict, CLASS, isEachClass=False):
    """
    数据集所有类别小、中、大GT分布情况
    isEachClass 控制是否按每个类别输出结构
    """
    TS, S, M, L = 0, 0, 0, 0
    # 需要手动初始化下，有多少个类别就需要写多个
    classDict = dict()
    for i in CLASS:
        classDict[i] = {'TS': 0, 'S': 0, 'M': 0, 'L': 0}

    # print(classDict['0']['S'])
    # range(class_num)类别数 注意修改!!!
    if isEachClass == False:
        for i in CLASS:
            ts, s, m, l = getSMLGtNumByClass(data_dict, i)
            TS += ts
            S += s
            M += m
            L += l
        return [TS, S, M, L]
    else:
        for i in CLASS:
            TS = 0
            S = 0
            M = 0
            L = 0
            ts, s, m, l = getSMLGtNumByClass(data_dict, i)
            TS += ts
            S += s
            M += m
            L += l
            classDict[i]['TS'] = TS
            classDict[i]['S'] = S
            classDict[i]['M'] = M
            classDict[i]['L'] = L
        return classDict


# 画图函数
def plotAllSML(SML):
    x = ['S:[0, 32x32]', 'M:[32x32, 96x96]', 'L:[96x96, 640x640]']
    fig = plt.figure(figsize=(10, 8))  # 画布大小和像素密度
    plt.bar(x, SML, width=0.5, align="center", color=['skyblue', 'orange', 'green'])
    for a, b, i in zip(x, SML, range(len(x))):  # zip 函数
        plt.text(a, b + 0.01, "%d" % int(SML[i]), ha='center', fontsize=15, color="r")  # plt.text 函数
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('gt_size)', fontsize=16)
    plt.ylabel('count', fontsize=16)
    plt.title('The count of gt size', fontsize=16)
    plt.show()
    # 保存到本地
    # plt.savefig("")


if __name__ == '__main__':
    # labeldir = '/media/ubuntu/nvidia/dataset/VEDAI_1024/trainval/labelTxt_noOther'  # VEDAI
    # labeldir = '/media/ubuntu/CE425F4D425F3983/datasets/DOTA/train/labelTxt-v1.0/labelTxt'  # DOTA1.0
    # labeldir = '/media/ubuntu/CE425F4D425F3983/datasets/DOTA/train/labelTxt-v1.5/DOTA-v1.5_train'  # DOTA1.5
    # labeldir = '/media/ubuntu/nvidia/dataset/DOTA-2/train/labelTxt-v2.0/DOTA-v2.0_train'  # DOTA2.0

    # labeldir = '/media/ubuntu/nvidia/dataset/ssdd/train/labelTxt'  # SSDD
    labeldir = '/media/ubuntu/nvidia/dataset/hrsid/trainsplit/labelTxt'  # hrsid
    # CLASS = {'car', 'camping-car', 'tractor', 'van', 'pickup', 'truck'}
    data_dict, CLASS = getGtAreaAndRatio(labeldir)
    # 1、数据集所有类别微小、小、中、大GT分布情况，其中微小目标属于单独统计，也属于小目标
    # 控制是否按每个类别输出结构
    isEachClass = False
    SML = getAllSMLGtNum(data_dict, CLASS, isEachClass)
    instance_num = SML[1] + SML[2] + SML[3]
    print(f'{SML}\n'
          f'微小目标占比为{round(SML[0] / instance_num * 100, 2)}%\n'
          f'小目标占比为{round(SML[1] / instance_num * 100, 2)}%\n'
          f'中目标占比为{round(SML[2] / instance_num * 100, 2)}%\n'
          f'大目标占比为{round(SML[3] / instance_num * 100, 2)}%\n')
    # if not isEachClass:
    #     plotAllSML(SML)

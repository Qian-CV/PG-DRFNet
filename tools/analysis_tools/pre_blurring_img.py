import cv2
from imgaug import augmenters as iaa
import os
import glob

######## 随机光亮
# seq = iaa.Sequential([
#     iaa.AddToBrightness((-40, 40))
# ])
######## 随机模糊
# seq = iaa.OneOf([
#     iaa.GaussianBlur(sigma=(0, 3)),  # 使用0到3.0的sigma模糊图像
#     iaa.AverageBlur(),
#     iaa.MedianBlur()
# ])
######## 随机噪声
seq = iaa.OneOf([
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
    iaa.SaltAndPepper((0, 0.1)),
    iaa.ImpulseNoise((0, 0.1))
])
####### 云层遮挡
# seq = iaa.Sequential([
#     iaa.Clouds()
# ])

img_dir = '/media/ubuntu/CE425F4D425F3983/datasets/HRSC2016fg/test/AllImages/'
# img_dir = '/media/ubuntu/CE425F4D425F3983/datasets/HRSC2016fg/trainval/extracted_obj/所有类别/AllImages/'
# img_dir = '/media/ubuntu/CE425F4D425F3983/datasets/HRSC2016fg/test/extracted_obj_gt/所有类别/AllImages/'
new_img_dir = '/media/ubuntu/CE425F4D425F3983/datasets/HRSC2016fg/test/Blurring_img/Noise/'
# new_img_dir = '/media/ubuntu/CE425F4D425F3983/datasets/HRSC2016fg/trainval/Blurring_extracted_img/Noise/'
# new_img_dir = '/media/ubuntu/CE425F4D425F3983/datasets/HRSC2016fg/test/extracted_obj_gt/Blurring_test_img/Noise/'
if not os.path.exists(new_img_dir):
    os.mkdir(new_img_dir)
filelist = glob.glob(f'{img_dir}/*.bmp')
img_name_list = os.listdir(img_dir)
for img in img_name_list:
    img_path = os.path.join(img_dir, img)
    if os.path.isdir(img_path):
        img_name_list.remove(img)
        break
######### 单个图像测试
# imglist = []
# img = cv2.imread(img_dir + '100000003.bmp')
# imglist.append(img)
# images_aug = seq.augment_images(imglist)
# cv2.imwrite(new_img_dir + "imgaug.bmp", images_aug[0])
# cv2.imwrite(new_img_dir + "yuan.bmp", imglist[0])
######### 批量处理
imglist = []
for file in filelist:
    img = cv2.imread(file)
    imglist.append(img)
print('图片读取完成，开始增强')
images_aug = seq.augment_images(imglist)
for num, img in enumerate(img_name_list):
    cv2.imwrite(new_img_dir + img, images_aug[num])
print(f'增强完成！，共增强{len(img_name_list)}张图片')

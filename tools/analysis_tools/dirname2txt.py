import os


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


file_dir = '/media/ubuntu/CE425F4D425F3983/datasets/VEDAI_1024/test/labelTxt_noOther'
num_list = []

for txt in os.listdir(file_dir):
    txt_list = txt.split('.')
    num_list.append(txt_list[0])
    with open('/media/ubuntu/CE425F4D425F3983/datasets/VEDAI_1024/test' + '/test2.txt', 'a+') as f:
        f.write(f'{txt_list[0]}\n')
f.close()
print(num_list)


import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap


# 假设 feature_map 是形状为 (1, 192, 32, 32) 的张量
# 例如，使用随机数据进行模拟
# feature_map = torch.randn(1, 192, 32, 32)
def show_feature(feature_map):
    for n, layer in enumerate(feature_map):
        # 计算每个位置在所有通道的平均值
        layer = layer.squeeze(0)  # 从 (1, 192, 32, 32) 变为 (192, 32, 32)
        attention_map = layer.mean(dim=0)  # 计算在所有 192 个通道上的平均值 (得到 32x32)
        norm_attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # 创建自定义色图：绿色到黄色的渐变
        colors = ["#2E8B57", "#FFD700"]  # 绿色到黄色的渐变
        n_bins = 100  # 创建100个颜色段
        cmap_name = "green_to_yellow"
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # 显示生成的特征注意力图
        cmap_reversed = plt.cm.YlGnBu_r
        plt.imshow(norm_attention_map.cpu().detach().numpy(), cmap=custom_cmap)  # 使用 'jet' 色图显示
        plt.axis('off')  # 去除坐标轴
        # 保存图像
        dir = f'/media/ubuntu/nvidia/wlq/part1_tiny_detection/mmrotate-1.x/demo/feature_map_show/'
        os.makedirs(dir, exist_ok=True)
        save_dir = os.path.join(dir, f'feature_{n + 1}')
        plt.savefig(save_dir, bbox_inches='tight', pad_inches=0.1)  # 保存为 PNG 文件
        # plt.colorbar()  # 显示颜色条
        # plt.title('Feature Attention Map')
        # plt.show()

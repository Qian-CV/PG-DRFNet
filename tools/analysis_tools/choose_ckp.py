import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            epoch = 1
            for i, line in enumerate(log_file):
                log = json.loads(line.strip())
                val_flag = False
                # skip lines only contains one key
                if not len(log) > 1:
                    continue

                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)

                for k, v in log.items():
                    if '/' in k:
                        log_dict[epoch][k.split('/')[-1]].append(v)
                        val_flag = True
                    elif val_flag:
                        continue
                    else:
                        log_dict[epoch][k].append(v)

                if 'epoch' in log.keys():
                    epoch = log['epoch']

    return log_dicts


def plot_curve(log_dicts, args):
    """Plot curve."""
    if args.backend is not None:
        plt.switch_backend(args.backend)
    if sns is None:
        raise ImportError('Please run "pip install seaborn" '
                          'to install seaborn first.')
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[0]]:
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}')

            if 'mAP' in metric:
                xs = np.arange(1, max(epochs) + 1)
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                ax = plt.gca()
                ax.set_xticks(xs)
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-2]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('iter')
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def main():
    log_json_dir = ['/media/ubuntu/nvidia/wlq/part1_tiny_detection/mmrotate-1.x/tools/work_dirs/rotated_rtmdet_m-3x-dota/20230715_183104/vis_data/20230715_183104.json']
    metric = 'mAP'
    # metric = 'accuracy'
    log_dicts = load_json_logs(log_json_dir)
    if metric == 'mAP':
        map = []
        for i, log_dict in enumerate(log_dicts):
            epochs = list(log_dict.keys())
            for j, epoch in enumerate(epochs):
                # print(log_dict[epoch]['mAP'])
                if log_dict[epoch][metric]:
                    map.append(log_dict[epoch]['mAP'][0])
                else:
                    continue
            max_index_map = np.argmax(np.array(map))
            per_val_epoch_num = len(epochs) / len(map)
            print(f'总共有{len(map)}个val,其中mAP最准确的权重是第{((max_index_map + 1) * per_val_epoch_num):.0f}, mAP:{map[max_index_map]}')
            # 可视化绘图
            xs = np.arange(per_val_epoch_num, per_val_epoch_num * (len(map) + 1), per_val_epoch_num)
            ys = map
            # for epoch in epochs:
            #     ys += log_dict[epoch][metric]
            ax = plt.gca()
            ax.set_xticks(xs)
            plt.xlabel('epoch')
            plt.ylabel('mAP')
            plt.plot(xs, ys, marker='o')
            plt.show()
    elif metric == 'accuracy':
        top_1 = []
        top_5 =[]
        for i, log_dict in enumerate(log_dicts):
            epochs = list(log_dict.keys())
            for j, epoch in enumerate(epochs):
                if log_dict[epoch]['accuracy_top-1']:
                    top_1.append(log_dict[epoch]['accuracy_top-1'][0])
                    top_5.append(log_dict[epoch]['accuracy_top-5'][0])
                else:
                    continue
            max_index_top1 = np.argmax(np.array(top_1))
            max_index_top5 = np.argmax(np.array(top_5))
            per_val_epoch_num = len(epochs) / len(top_1)
            if per_val_epoch_num == 1:
                print(f'总共有{len(top_1)}个val,top_1最准确的权重是第{((max_index_top1 + 1)* epochs[0]):.0f}, top_1:{top_1[max_index_top1]}')
                print(f'总共有{len(top_5)}个val, top_5最准确的权重是第{((max_index_top5 + 1)* epochs[0]):.0f}, top_5:{top_5[max_index_top5]}')
            else:
                print(f'总共有{len(top_1)}个val,top_1最准确的权重是第{((max_index_top1 + 1)* per_val_epoch_num):.0f}, top_1:{top_1[max_index_top1]}')
                print(f'总共有{len(top_5)}个val, top_5最准确的权重是第{((max_index_top5 + 1)* per_val_epoch_num):.0f}, top_5:{top_5[max_index_top5]}')
            # 可视化绘图
            xs = np.arange(1, len(top_1) + 1)
            ys1 = top_1
            ys2 = top_5
            ax = plt.gca()
            ax.set_xticks(xs)
            plt.xlabel('epoch')
            plt.ylabel('top-1 and top-5')
            plt.plot(xs, ys1, marker='o')
            plt.plot(xs, ys2, marker='o')
            plt.show()


if __name__ == '__main__':
    main()
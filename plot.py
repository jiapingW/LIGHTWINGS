import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def extract_scalars_from_tfevents(log_dir, tags):
    """
    从 TensorBoard 日志目录中提取指定 tags 的标量数据。
    
    Args:
        log_dir (str): TensorBoard 日志根目录（可包含多个子 run）
        tags (list of str): 要提取的标量 tag 名称，如 ['acc', 'accuracy', 'val_acc']
    
    Returns:
        dict: {run_name: {tag: (steps, values)}}
    """
    data = defaultdict(lambda: defaultdict(list))
    
    # 遍历所有子目录（每个子目录视为一个 run）
    for root, dirs, files in os.walk(log_dir):
        # 检查是否有 events 文件
        event_files = [f for f in files if f.startswith("events.out.tfevents")]
        if not event_files:
            continue
        
        run_name = os.path.relpath(root, log_dir)
        if run_name == '.':
            run_name = 'main'
        
        print(f"Processing run: {run_name} (dir: {root})")
        
        # 使用 EventAccumulator 加载事件
        event_acc = EventAccumulator(root)
        event_acc.Reload()
        
        # 获取所有可用的标量 tags
        available_tags = event_acc.Tags()['scalars']
        print(f"  Available scalar tags: {available_tags}")
        
        # 提取每个目标 tag
        for tag in tags:
            if tag in available_tags:
                scalar_events = event_acc.Scalars(tag)
                steps = [s.step for s in scalar_events]
                values = [s.value for s in scalar_events]
                data[run_name][tag] = (steps, values)
                print(f"    Extracted {len(steps)} points for tag '{tag}'")
            else:
                print(f"    Tag '{tag}' not found in this run.")
    
    return dict(data)

def plot_accuracy(data, save_path=None):
    """
    使用 matplotlib 绘制准确率曲线。
    
    Args:
        data (dict): 由 extract_scalars_from_tfevents 返回的数据
        save_path (str, optional): 保存图像的路径，如 'acc_plot.png'
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0
    
    for run_name, tags_data in data.items():
        for tag, (steps, values) in tags_data.items():
            label = f"{run_name}/{tag}" if run_name != 'main' else tag
            plt.plot(steps, values, marker='.', linestyle='-', 
                     color=colors[color_idx % len(colors)], label=label)
            color_idx += 1
    
    plt.title('Training / Validation Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy from TensorBoard logs")
    parser.add_argument("--log_dir", type=str, help="Path to TensorBoard log directory")
    parser.add_argument("--tags", type=str, nargs='+', default=['acc', 'accuracy', 'val_acc', 'val_accuracy'],
                        help="List of scalar tags to extract (default: %(default)s)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the plot image (e.g., acc.png)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.log_dir):
        print(f"Error: {args.log_dir} is not a valid directory.")
        return
    
    print(f"Searching for TensorBoard logs in: {args.log_dir}")
    data = extract_scalars_from_tfevents(args.log_dir, args.tags)
    
    if not data:
        print("No scalar data found with the specified tags.")
        return
    
    plot_accuracy(data, save_path=args.save)

if __name__ == "__main__":
    main()
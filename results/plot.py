import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tb_scalars(logdir):
    """
    从 TensorBoard 日志目录加载所有 scalar 数据。
    返回 DataFrame: ['source', 'tag', 'step', 'value']
    """
    print(f"Loading scalars from: {logdir}")
    all_records = []
    
    # 遍历所有子目录，寻找 events 文件
    for root, dirs, files in os.walk(logdir):
        if not any(f.startswith("events.out.tfevents") for f in files):
            continue
        
        try:
            acc = EventAccumulator(root)
            acc.Reload()
            
            scalars = acc.Tags().get('scalars', [])
            source_name = os.path.basename(os.path.abspath(logdir))
            
            for tag in scalars:
                events = acc.Scalars(tag)
                for e in events:
                    all_records.append({
                        'source': source_name,
                        'tag': tag,
                        'step': e.step,
                        'value': e.value
                    })
                    
        except Exception as ex:
            print(f"  ⚠️ Skip {root}: {ex}")
    
    df = pd.DataFrame(all_records)
    print(f"✅ Loaded {len(df)} scalar points from {logdir}")
    return df

def plot_four_acc(df_all, output_file="accuracy_comparison.png"):
    target_tags = [
        'train/acc_0',
        'train/acc_6',
        'eval/acc_0',
        'eval/acc_6'
    ]
    
    # 子图标题
    subplot_titles = [
        "Accuracy on position 0",
        "Accuracy on position 6",
        "Accuracy on position 0",
        "Accuracy on position 6"
    ]
    
    # 【新增】底部说明文字 (Caption)
    bottom_caption = (
        "We trained drafters using linear attention and softmax attention on the ShareGPT dataset, tracking prediction accuracy at positions 0 and 6 on both training and evaluation sets.\nThe results demonstrate that the linear attention drafter exhibits stable training and achieves a higher acceptance rate."
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 顶部大标题
    # fig.suptitle("Training on sharegpt dataset", fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    sources = sorted(df_all['source'].unique())
    linestyles = ['-', '--', '-.', ':']
    
    for idx, tag in enumerate(target_tags):
        ax = axes[idx]
        plotted = False
        
        for i, source in enumerate(sources):
            subset = df_all[(df_all['tag'] == tag) & (df_all['source'] == source)]
            if not subset.empty:
                ax.plot(
                    subset['step'],
                    subset['value'],
                    label=source,
                    linestyle=linestyles[i % len(linestyles)],
                    marker='.',
                    markersize=3
                )
                plotted = True
        
        ax.set_title(subplot_titles[idx], fontsize=14)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.grid(True, linestyle='--', alpha=0.6)
        if plotted:
            ax.legend()
    
    # 【修改布局】
    # rect=[left, bottom, right, top]
    # 将 bottom 从 0.03 改为 0.1，为底部的 caption 留出足够的空白区域
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # 【新增】在底部添加文字
    # x=0.5, y=0.02 表示水平居中，位于底部上方 2% 的位置
    fig.text(
        0.5, 0.02, 
        bottom_caption, 
        ha='center', 
        fontsize=14, 
        wrap=True
    )
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot train/eval acc_0 and acc_6 from two TensorBoard logs")
    parser.add_argument("--logdir1", help="Path to first TensorBoard log directory")
    parser.add_argument("--logdir2", help="Path to second TensorBoard log directory")
    parser.add_argument("--output", "-o", default="accuracy_comparison.png", help="Output image file")
    args = parser.parse_args()
    
    df1 = load_tb_scalars(args.logdir1)
    df2 = load_tb_scalars(args.logdir2)
    
    if df1.empty and df2.empty:
        print("❌ No data loaded from either directory.")
        return
    
    df_all = pd.concat([df1, df2], ignore_index=True)
    plot_four_acc(df_all, args.output)

if __name__ == "__main__":
    main()
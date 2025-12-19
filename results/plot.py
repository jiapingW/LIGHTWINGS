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
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    sources = sorted(df_all['source'].unique())
    linestyles = ['-', '--', '-.', ':']  # 支持最多4个实验
    
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
        
        ax.set_title(tag, fontsize=14)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.grid(True, linestyle='--', alpha=0.6)
        if plotted:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot train/eval acc_0 and acc_6 from two TensorBoard logs")
    parser.add_argument("--logdir1", help="Path to first TensorBoard log directory (e.g., linear_attn)")
    parser.add_argument("--logdir2", help="Path to second TensorBoard log directory (e.g., softmax_attn)")
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
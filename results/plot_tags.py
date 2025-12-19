import os
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def inspect_tb_tags(logdir):
    print(f"🔍 Inspecting TensorBoard logs in: {logdir}")
    if not os.path.isdir(logdir):
        print("❌ Error: Not a valid directory.")
        return

    # 查找所有包含 events 文件的子目录（每个子目录是一个 run）
    runs = []
    for root, dirs, files in os.walk(logdir):
        if any(f.startswith("events.out.tfevents") for f in files):
            rel_path = os.path.relpath(root, logdir)
            runs.append(('', root) if rel_path == '.' else (rel_path, root))

    if not runs:
        print("⚠️  No TensorBoard event files found. Make sure the directory contains 'events.out.tfevents.*'.")
        return

    for run_name, run_path in runs:
        print(f"\n📁 Run: '{run_name or '.'}' (path: {run_path})")
        try:
            acc = EventAccumulator(run_path)
            acc.Reload()  # 加载所有摘要数据

            scalars = acc.Tags().get('scalars', [])
            if scalars:
                print("  📈 Scalars:")
                for tag in sorted(scalars):
                    print(f"    - {tag}")
            else:
                print("  📉 No scalar tags found.")

        except Exception as e:
            print(f"  ❌ Failed to load run '{run_name}': {e}")

def main():
    parser = argparse.ArgumentParser(description="List all scalar tags in a TensorBoard log directory")
    parser.add_argument("--logdir", help="Path to TensorBoard log directory")
    args = parser.parse_args()
    inspect_tb_tags(args.logdir)

if __name__ == "__main__":
    main()
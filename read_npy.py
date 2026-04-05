import numpy as np
import os

def inspect_npy(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 '{file_path}'")
        return

    try:
        # 加载数据
        # allow_pickle=True 是为了防止数据中包含 python 对象
        data = np.load(file_path, allow_pickle=True)

        print(f"--- 文件信息: {os.path.basename(file_path)} ---")
        print(f"数据类型 (dtype): {data.dtype}")
        print(f"数据维度 (shape): {data.shape}")
        print(f"元素总数: {data.size}")
        print(f"内存占用: {data.nbytes / 1024:.2f} KB")
        print("-" * 30)
        
        # 打印具体内容
        print("数据内容预览:")
        print(data)
        
    except Exception as e:
        print(f"解析文件时出错: {e}")

if __name__ == "__main__":
    # 在这里输入你的 npy 文件路径
    path = "/data/home/zyzeng/project/DiMTS/OUTPUT/fmri_seq256_pred128_fft/samples/fmri_norm_truth_256_train.npy" 
    inspect_npy(path)
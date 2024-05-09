import numpy as np
import struct

def read_fvecs(file_path):
    # 读取.fvecs格式的文件并返回向量数组
    vectors = []
    with open(file_path, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector = np.fromfile(f, dtype=np.float32, count=dim)
            if vector.size == 0:
                break
            vectors.append(vector)
    return np.array(vectors)

def vector_lengths(vectors):
    # 计算并返回向量数组中每个向量的长度（欧几里得范数）
    lengths = np.linalg.norm(vectors, axis=1)
    return lengths

def statistics(lengths):
    # 计算并返回长度数组的平均值、标准差、最大值、最小值和中位数
    mean = np.mean(lengths)
    std_dev = np.std(lengths)
    max_length = np.max(lengths)
    min_length = np.min(lengths)
    median_length = np.median(lengths)
    return mean, std_dev, max_length, min_length, median_length

# 计算fvces文件中向量长度的平均值、标准差、最大值、最小值和中位数
# 可以发现sift数据集中的所有向量长度相近
if __name__ == "__main__":
    file_paths = [
        'dataset/gist/gist_query.fvecs',
        'dataset/sift/sift_query.fvecs',
        'dataset/gist/gist_base.fvecs',
        'dataset/sift/sift_base.fvecs'
                ]
    for file_path in file_paths:
        vectors = read_fvecs(file_path)
        lengths = vector_lengths(vectors)
        mean_length, std_dev_length, max_length, min_length, median_length = statistics(lengths)

        print(f"{file_path}")
        print(f"平均向量长度: {mean_length}")
        print(f"向量长度标准差: {std_dev_length}")
        print(f"最大向量长度: {max_length}")
        print(f"最小向量长度: {min_length}")
        print(f"向量长度中位数: {median_length}\n")

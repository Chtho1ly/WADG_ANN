# 功能：
# 重用频率cdf绘制
# 查询距离-结果重合率散点图绘制
# 魔改轮廓系数计算
# 空间局部性实验

import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform, cdist, euclidean

# 读取Yandex DEEP、Yandex Text-to-Image数据集中的groundtruth文件
def load_yandex(file_path):
    with open(file_path, 'rb') as f:
        num_vectors, vector_dim = struct.unpack('II', f.read(8))
        vector_array = np.fromfile(f, dtype=np.int32, count=num_vectors * vector_dim)
        return vector_array.reshape((num_vectors, vector_dim))

# 读取Microsoft SPACEV数据集中的groundtruth文件
def load_ms(file_path):
    with open(file_path, 'rb') as f:
        t_count = struct.unpack('i', f.read(4))[0]  # 总查询数
        topk = struct.unpack('i', f.read(4))[0]    # 每个查询的最近邻个数
        # 读取真实的最近邻向量ID
        truth_vids = np.frombuffer(f.read(t_count * topk * 4), dtype=np.int32).reshape((t_count, topk))
        # 读取到对应的距离
        truth_distances = np.frombuffer(f.read(t_count * topk * 4), dtype=np.float32).reshape((t_count, topk))
    return truth_vids, truth_distances

# 读取sift、gist数据集中的ivecs文件
def load_ivecs(file_path):
    vectors = []
    with open(file_path, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector = np.fromfile(f, dtype=np.int32, count=dim)
            vectors.append(vector)
    return np.array(vectors)

# 读取sift、gist数据集中的fvecs文件
def load_fvecs(file_path, step=1):
    vectors = []
    cnt = 0
    with open(file_path, 'rb') as f:
        while True:
            dim_bytes = f.read(4)  # 读取维度
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector = np.fromfile(f, dtype=np.float32, count=dim)
            if cnt % step == 0:
                vectors.append(vector)
            cnt += 1
    return vectors

def cal_reuse_distances(vectors):
    last_seen = {}
    reuse_distances = []
    
    for index, vector in enumerate(vectors):
        for target_id in vector:
            if target_id in last_seen:
                distance = index - last_seen[target_id]
                reuse_distances.append(distance)
            last_seen[target_id] = index
    
    return reuse_distances, len(vectors)

def plot_reuse_distances_cdf(trace_name, reuse_distances, total_searches):
    values, counts = np.unique(reuse_distances, return_counts=True)
    frequencies = counts / counts.sum()
    cumulative_frequencies = np.cumsum(frequencies)

    # 归一化重用距离
    normalized_values = values / total_searches

    # 固定的次要坐标轴刻度
    fixed_normalized_ticks = np.arange(0, 1.05, 0.05)

    # 计算对应的主坐标轴刻度
    main_ticks = fixed_normalized_ticks * total_searches
    main_ticks = np.unique(np.round(main_ticks)).astype(int)  # 取整并去除重复值

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(values, cumulative_frequencies, drawstyle='steps-post', color='blue')
    ax.set_xlabel('Reuse Distance')
    ax.set_ylabel('Cumulative Frequency (CDF)')
    ax.set_title(f'Cumulative Distribution Function of Reuse Distances for {trace_name}')
    ax.set_xticks(main_ticks)  # 设置主轴的显示刻度

    # 添加次要横坐标轴
    ax2 = ax.secondary_xaxis('top')
    ax2.set_xlabel('Normalized Reuse Distance (Reuse Distance / Total Searches)')
    ax2.set_xticks(main_ticks)  # 使用计算得到的主轴刻度点
    ax2.set_xticklabels([f"{val:.2f}" for val in fixed_normalized_ticks])

    ax.grid(True)
    save_path = os.path.join('trace_analysis', 'figs', f'reuse_distance_cdf_{trace_name}.png')
    plt.savefig(save_path)
    plt.show()
    
def cal_euclidean_distances(vectors):
    distances = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean(vectors[i], vectors[j])
            distances.append(dist)
    return distances

def cal_overlap(ivecs):
    overlaps = []
    n = len(ivecs)
    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(set(ivecs[i]) & set(ivecs[j]))
            union = len(set(ivecs[i]) | set(ivecs[j]))
            if union > 0:  # Avoid division by zero
                overlap = intersection / union
                overlaps.append(overlap)
            else:
                overlaps.append(0)
    return overlaps

def plot_distances_vs_overlap(distances, overlaps, trace_name, window_size):
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, overlaps, alpha=0.2)
    plt.title(f'Euclidean Distance vs. Overlap Ratio for {trace_name} (Window Size: {window_size})')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Overlap Ratio')
    plt.grid(True)
    save_path = os.path.join('trace_analysis', 'figs', f'query_distance_vs_shared_result_{trace_name}_{window_size}.png')
    plt.savefig(save_path)
    plt.show()

def cal_alt_silhouette_coefficient(queries, bases, n_clusters=5):
    # Perform K-means clustering on query vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(queries)
    labels = kmeans.labels_
    
    # Compute intra-cluster distances (a)
    intra_distances = np.zeros(len(queries))
    for i in range(n_clusters):
        cluster_points = queries[labels == i]
        if len(cluster_points) > 1:
            distances = squareform(pdist(cluster_points, 'euclidean'))
            mean_distances = np.sum(distances, axis=1) / (len(cluster_points) - 1)
            intra_distances[labels == i] = mean_distances
        else:
            intra_distances[labels == i] = 0
    
    # Calculate distances to all base vectors (b)
    all_base_distances = cdist(queries, bases, metric='euclidean')
    mean_distances_to_bases = np.mean(all_base_distances, axis=1)
    
    # Calculate silhouette coefficients for each query vector
    silhouette_scores = (mean_distances_to_bases - intra_distances) / np.maximum(intra_distances, mean_distances_to_bases)
    
    # 计算整体queries的轮廓系数
    # average_silhouette = np.mean(silhouette_scores)
    # return average_silhouette
    
    # 计算每个簇的轮廓系数
    silhouette_coefficient_cluster = []
    cluster_size = []
    for i in range(n_clusters):
        # 排除返回值中大小为1或轮廓系数小于0.3的簇，以便于分析
        if len(queries[labels == i])>1 and np.mean(silhouette_scores[labels == i])>=0.3:
            silhouette_coefficient_cluster.append(round(np.mean(silhouette_scores[labels == i]),3))
            cluster_size.append(len(queries[labels == i]))
    
    return silhouette_coefficient_cluster, cluster_size

def cal_combined_ratios(queries, bases, n_clusters, cluster_method='k-means'):
    if cluster_method == 'k-means':
        # Execute K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(queries)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
    elif cluster_method == 'random':
        # Randomly select cluster centers from base vectors
        if n_clusters > len(bases):
            raise ValueError("Number of clusters cannot be more than the number of base vectors.")
        rng = np.random.default_rng(seed=0)
        center_indices = rng.choice(len(bases), size=n_clusters, replace=False)
        cluster_centers = bases[center_indices]
        labels = np.argmin(cdist(queries, cluster_centers), axis=1)

    # Calculate the centroid of the base vectors
    base_centroid = np.mean(bases, axis=0)

    # Initialize lists to collect distance ratios and cluster sizes
    ratio1_list, ratio2_list, ratio3_list, ratio4_list, cluster_sizes = [], [], [], [], []

    for i in range(n_clusters):
        # Select all points in the current cluster
        cluster_points = queries[labels == i]
        cluster_size = len(cluster_points)

        if cluster_size > 1:
            # Intra-cluster distances
            intra_cluster_distances = cdist([cluster_centers[i]], cluster_points, 'euclidean')
            average_intra_distance = np.mean(intra_cluster_distances)

            # Center to all base vectors distances
            center_to_bases_distances = cdist([cluster_centers[i]], bases, 'euclidean')
            average_center_to_base_distance = np.mean(center_to_bases_distances)

            # Cluster to all base vectors distances
            cluster_to_base_distances = cdist(cluster_points, bases, 'euclidean')
            average_cluster_to_base_distance = np.mean(cluster_to_base_distances)

            # Cluster to base centroid distances
            cluster_to_base_centroid_distances = cdist(cluster_points, [base_centroid], 'euclidean')
            average_cluster_to_base_centroid_distance = np.mean(cluster_to_base_centroid_distances)

            # 实验 4
            # the nearest random point index to the cluster centers
            nearest_center_to_base_distance_index = np.argmin(center_to_bases_distances)
            # Cluster to the nearest point distances
            cluster_to_nearest_center_distances = cdist(cluster_points, [bases[nearest_center_to_base_distance_index]], 'euclidean')
            average_cluster_to_nearest_center_distance = np.mean(cluster_to_nearest_center_distances)

            # Calculate ratios and round to three decimal places
            ratio1 = round(average_intra_distance / average_cluster_to_base_distance, 3)
            ratio2 = round(average_intra_distance / average_center_to_base_distance, 3)
            ratio3 = round(average_intra_distance / average_cluster_to_base_centroid_distance, 3)
            # 
            ratio4 = round(average_intra_distance / average_cluster_to_nearest_center_distance, 3)

            # Append each ratio to its respective list
            ratio1_list.append(ratio1)
            ratio2_list.append(ratio2)
            ratio3_list.append(ratio3)
            ratio4_list.append(ratio4)
            cluster_sizes.append(cluster_size)

    # Return the tuple of lists including cluster sizes
    return (ratio1_list, ratio2_list, ratio3_list, ratio4_list, cluster_sizes)


def cal_combined_ratios_56(kmeans:KMeans, curr_query: np.ndarray, groundtruths: np.ndarray, bases: np.ndarray, n_clusters, cluster_method='k-means'):
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Calculate the centroid of the base vectors
    base_centroid = np.mean(bases, axis=0)

    # 距离 curr_query 最近的 cluster_center 的 index
    curr_query_to_cluster_centers_distances = cdist([curr_query], cluster_centers, 'euclidean')
    nearest_cluster_center_to_curr_query_index = np.argmin(curr_query_to_cluster_centers_distances)
    nearest_cluster_center = cluster_centers[nearest_cluster_center_to_curr_query_index]

    # 最近聚类中心到最终结果距离
    nearest_cluster_center_to_groundtruth_distances = cdist([nearest_cluster_center], groundtruths, 'euclidean')
    avg_nearest_cluster_center_to_groundtruth_distance = np.mean(nearest_cluster_center_to_groundtruth_distances)

    # 实验 5
    # 计算 最近聚类中心到最终结果距离 / 质心到最终结果距离
    base_centroid_to_groundtruth_distances = cdist([base_centroid], groundtruths, 'euclidean')
    avg_base_centroid_to_groundtruth_distance = np.mean(base_centroid_to_groundtruth_distances)
    
    # 实验 6
    # 计算 最近聚类中心到最终结果距离 / 随机点到最终结果距离
    # Center to all base vectors distances
    nearest_cluster_center_to_bases_distances = cdist([nearest_cluster_center], bases, 'euclidean')
    random_point_index = np.argmin(nearest_cluster_center_to_bases_distances)
    random_point_to_groundtruth_distances = cdist([bases[random_point_index]], groundtruths, 'euclidean')
    avg_random_point_to_groundtruth_distance = np.mean(random_point_to_groundtruth_distances)

    ratio5 = round(avg_nearest_cluster_center_to_groundtruth_distance / avg_base_centroid_to_groundtruth_distance, 3)
    ratio6 = round(avg_nearest_cluster_center_to_groundtruth_distance / avg_random_point_to_groundtruth_distance, 3)

    return (ratio5, ratio6, len(groundtruths))


def cal_weighted_avg_ratio(ratio1_list, ratio2_list, ratio3_list, ratio4_list, cluster_sizes):
    # Initialize weighted sums and total weight counters
    weighted_sum1, weighted_sum2, weighted_sum3, weighted_sum4 = 0, 0, 0, 0
    total_weight = sum(cluster_sizes)  # Total weight is the sum of all cluster sizes

    # Calculate weighted sums for each ratio
    for size, ratio1, ratio2, ratio3, ratio4 in zip(cluster_sizes, ratio1_list, ratio2_list, ratio3_list, ratio4_list):
        weighted_sum1 += ratio1 * size
        weighted_sum2 += ratio2 * size
        weighted_sum3 += ratio3 * size
        weighted_sum4 += ratio4 * size

    # Calculate the weighted averages
    if total_weight > 0:
        weighted_average1 = weighted_sum1 / total_weight
        weighted_average2 = weighted_sum2 / total_weight
        weighted_average3 = weighted_sum3 / total_weight
        weighted_average4 = weighted_sum4 / total_weight
    else:
        # Handle case where total weight is zero (unlikely unless all clusters are empty)
        weighted_average1, weighted_average2, weighted_average3, weighted_average4 = None, None, None

    return (weighted_average1, weighted_average2, weighted_average3, weighted_average4)


def plot_ratios(x_coords, ratio1_list, ratio2_list, ratio3_list, ratio4_list, trace_name, cluster_ratio, cluster_method='k-means'):
    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot each set of ratios
    # plt.plot(x_coords, ratio1_list, label='center base ratios', marker='o', linestyle='-')
    # plt.plot(x_coords, ratio2_list, label='cluster base ratios', marker='o', linestyle='-')
    plt.plot(x_coords, ratio3_list, label='root cluster ratios', marker='o', linestyle='-')
    plt.plot(x_coords, ratio4_list, label='cluster nearest ratios', marker='o', linestyle='-')

    # Adding labels and title
    plt.xlabel('window size')
    plt.ylabel('ratios')
    plt.title(f'Weighted Average Ratios vs. window size {trace_name} {cluster_method}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    save_path = os.path.join('trace_analysis', 'figs_new', f'window_size_vs_ratios_{trace_name}_{cluster_ratio}_{cluster_method}.png')
    plt.savefig(save_path)
    # plt.show()


def plot_ratios_56(x_coords, ratio5_list, ratio6_list, trace_name, cluster_ratio, cluster_method = 'k-means'):
     # Create a plot
    plt.figure(figsize=(10, 6))

    plt.plot(x_coords, ratio5_list, label='cluster centroid ratios', marker=None, linestyle='-')
    plt.plot(x_coords, ratio6_list, label='cluster random ratios', marker=None, linestyle='-')

    # Adding labels and title
    plt.xlabel('queries')
    plt.ylabel('ratios')
    plt.title(f'Average Ratios {trace_name} {cluster_method}')
    plt.legend()

    # Show the plot
    plt.grid(True)
    save_path = os.path.join('trace_analysis', 'figs_new', f'queries_vs_ratios_{trace_name}_{cluster_ratio}_{cluster_method}.png')
    plt.savefig(save_path)


def process_files(trace_source_list, expts: list):
    for trace_source in trace_source_list:
        trace_name, source = os.path.splitext(os.path.basename(trace_source))
        if source == '.bigann':
            # 读取数据
            groundtruth_vectors = load_ivecs(os.path.join('dataset', trace_name, f'{trace_name}_groundtruth.ivecs'))
            query_vectors = load_fvecs(os.path.join('dataset', trace_name, f'{trace_name}_query.fvecs'))
            base_vectors = load_fvecs(os.path.join('dataset', trace_name, f'{trace_name}_base.fvecs')) # base向量过多，可以进行间隔采样以降低计算复杂度

            # 查询距离-结果重合率散点图
            # 魔改轮廓系数
            if (expts[0]):
                cluster_ratio = 5
                window_sizes = [20, 50, 200, 500, 1000]
                if trace_name == 'sift': # sift数据集query向量数量为10000，可以扩大窗口
                    window_sizes.extend([2000, 4000])
                for window_size in window_sizes:
                    cluster_num = window_size//cluster_ratio
                    print(f"{trace_name}: window_size={window_size}, cluster_num={cluster_num}")
                    query_vectors_window = query_vectors[:window_size]
                    groundtruth_vectors_window = groundtruth_vectors[:window_size]

                    # 查询距离-结果重合率散点图
                    distances = cal_euclidean_distances(query_vectors_window)
                    overlaps = cal_overlap(groundtruth_vectors_window)
                    plot_distances_vs_overlap(distances, overlaps, trace_name, window_size)

                    # 魔改轮廓系数
                    # 在命令行输出，不会保存到文件

                    alt_silhouette_coefficient, cluster_size = cal_alt_silhouette_coefficient(np.array(query_vectors_window), np.array(base_vectors), cluster_num)
                    print(f"Alternative Silhouette Coefficient for {trace_name}: {alt_silhouette_coefficient}")
                    print(f"                               Cluster Size: {cluster_size}")
            
            # 空间局部性实验 1, 2, 3, 4
            # 簇到base平均距离
            if (expts[1]):
                cluster_ratios = [10] # query向量数与聚类中心数之比
                cluster_methods = ['k-means', 'random'] # 聚类方法
                window_sizes = [] # 窗口大小
                window_sizes.append(10)
                window_sizes.append(20)
                window_sizes.extend(range(50, 300, 50))
                window_sizes.extend(range(300, 1000, 100))
                if trace_name == 'sift': # sift数据集query向量数量为10000，可以扩大窗口
                    window_sizes.extend(range(1000, 4200, 200))
                for cluster_method in cluster_methods:
                    for cluster_ratio in cluster_ratios:
                        avg_center_base_ratios, avg_cluster_base_ratios, avg_root_cluster_ratios, avg_cluster_nearest_ratios = [], [], [], [] # 使用不同窗口大小时的比例，用于做图
                        for window_size in window_sizes:
                            cluster_num = window_size//cluster_ratio
                            # query vectors window
                            query_vectors_window = query_vectors[:window_size]
                            print(f"{trace_name}: window_size={window_size}, cluster_num={cluster_num}")

                            # 各簇对应比例系数以及簇的大小
                            center_base_ratios, cluster_base_ratios, root_cluster_ratios, cluster_nearest_ratios, cluster_sizes = cal_combined_ratios(np.array(query_vectors_window), np.array(base_vectors), cluster_num, cluster_method)
                            # 以簇大小为权重求加权平均
                            avg_ratios = cal_weighted_avg_ratio(center_base_ratios, cluster_base_ratios, root_cluster_ratios, cluster_nearest_ratios, cluster_sizes)
                            avg_center_base_ratios.append(avg_ratios[0])
                            avg_cluster_base_ratios.append(avg_ratios[1])
                            avg_root_cluster_ratios.append(avg_ratios[2])
                            avg_cluster_nearest_ratios.append(avg_ratios[3])
                        plot_ratios(window_sizes, avg_center_base_ratios, avg_cluster_base_ratios, avg_root_cluster_ratios, avg_cluster_nearest_ratios, trace_name, cluster_ratio, cluster_method)
        
            # 补: 空间局部性实验 5, 6
            if (expts[2]):
                cluster_ratio = 10
                ratio5_list, ratio6_list, x_coords = [], [], []   
                # 要聚类的 N 个 queries
                queries_to_cluster = []
                # 存储 kmeans
                kmeans_list = []
                # 顺序处理每个 query
                # for i in range(len(query_vectors)):
                for i in range(400):
                    # 当前 query
                    curr_query = query_vectors[i]

                    # 当为 100 的倍数时计算聚类
                    if ((i + 1) % 100 == 0):
                        # 计算 KMeans
                        cluster_num = (i + 1) // 10
                        kmeans = KMeans(n_clusters = cluster_num)
                        kmeans.fit(np.array(queries_to_cluster))
                        kmeans_list.append(kmeans)

                    # 当聚类一次后开始计算
                    if (i >= 99):
                        # groundtruth: list
                        curr_groundtruths = np.array(base_vectors).take(groundtruth_vectors[i], axis = 0)
                        print(f"{trace_name}: curr_query={i}, cluster_num={cluster_num}")

                        ratio5, ratio6, groundtruth_size = cal_combined_ratios_56(kmeans_list[(i + 1) // 100 - 1], 
                                                                                   np.array(curr_query), 
                                                                                   np.array(curr_groundtruths), 
                                                                                   np.array(base_vectors), cluster_num)
                        ratio5_list.append(ratio5)
                        ratio6_list.append(ratio6)
                        x_coords.append(i + 1)

                    # 前 N-1 个queries 加入
                    queries_to_cluster.append(curr_query)
                # plot
                plot_ratios_56(x_coords, ratio5_list, ratio6_list, trace_name, cluster_ratio)


        # 没找到完整数据集，只分析了groundtruth里面的重用距离
        elif source == '.yandex':
            file_path = os.path.join('dataset', trace_name, f'{trace_name}_groundtruth.bin')
            groundtruth_vectors = load_yandex(file_path)  
        elif source == '.ms':
            file_path = os.path.join('dataset', trace_name, f'{trace_name}_groundtruth.bin')
            groundtruth_vectors,_ = load_ms(file_path)
        else:
            print(f"Unsupported file extension: {source}")
            continue
        
        # 重用距离cdf
        if (expts[0]):
            reuse_distances, total_searches = cal_reuse_distances(groundtruth_vectors)
            plot_reuse_distances_cdf(trace_name, reuse_distances, total_searches)



if __name__ == "__main__":
    trace_source_list = [
        # 'deep.yandex', 
        # 't2i.yandex',
        # 'spacev.ms',
        'gist.bigann',
        # 'sift.bigann',
        ]
    # [查询距离-结果重合率散点图 + 魔改轮廓系数 + 重用距离cdf, 空间局部性实验1234，空间局部性实验56]
    process_files(trace_source_list, [False, False, True])
/*
 * @Author: Chen Shi
 * @Date: 2024-03-08 14:50:58
 * @Description: implement of IndexWADG class
 */

#include <map>
#include "efanna2e/index_wadg.h"
// k-means 聚类库，性能好
#include "efanna2e/dkm.hpp"
#include <vector>
#include <algorithm>
#include <iomanip>

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>

#include <efanna2e/exceptions.h>
#include <efanna2e/parameters.h>

namespace efanna2e
{
  #define _CONTROL_NUM 100
  // @CS0522
  IndexWADG::IndexWADG(const size_t dimension, size_t n, Metric m, const unsigned K,
                       Index *initializer)
      : IndexNSG(dimension, n, m, initializer) 
      {
        // TODO 初始化 max_hot_points_num, 该值应该大于 K
        max_hot_points_num = K * 2;
        // TODO 初始化 window_size 搜索请求记录窗口大小
        window_size = K;
        // TODO 初始化 cluster_num 聚类中心数
        cluster_num = K;
        // TODO print
        std::cout << "max_hot_points_num: " << max_hot_points_num << std::endl 
                  << "window_size: " << window_size << std::endl
                  << "cluster_num: " << cluster_num << std::endl
                  << "K: " << K << std::endl; 
        // 初始化 LRU 队列
        // (key, value): (下标, 有效热点 id)
        hot_points_lru = new LRUCache(max_hot_points_num);
        // 刚开始时 LRU 队列随机选取节点
        for (unsigned i = 0; i < max_hot_points_num; i++)
        {
          hot_points_lru->put(rand() % nd_);
        }
      }

  // @CS0522
  IndexWADG::~IndexWADG()
  {
    if (distance_ != nullptr) {
        delete distance_;
        distance_ = nullptr;
    }
    if (initializer_ != nullptr) {
        delete initializer_;
        initializer_ = nullptr;
    }
  }
  
  // @CS0522
  void IndexWADG::Set_data(const float *x)
  {
    data_ = x;
  }

  // @CS0522
  // TODO 修改为多线程
  void IndexWADG::Search(const float *query, const Parameters &parameters, unsigned*& indices,
                          bool record_query_flag)
  {
    // TODO print
    // std::cout << (record_query_flag ? "主 Search 开始" : "聚类中心 Search 开始") << std::endl;

    // const unsigned L = parameters.Get<unsigned>("L_search");
    // const unsigned K = parameters.Get<unsigned>("K_search");
    auto L = parameters.Get<unsigned>("L_search");
    auto K = parameters.Get<unsigned>("K_search");

    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};

    // TODO 从 LRU 缓存中选取离搜索目标最近的 L 个点作为初始队列 init_ids
    // 不在 LRU 内部进行排序，因为需要 index->distance_
    // 上锁
    mtx_lru.lock();
    // TODO 需要额外拷贝？可以直接 lru->get(index)？
    std::vector<unsigned> lru_copy = hot_points_lru->get_cache();
    // 解锁
    mtx_lru.unlock();

    // 求前 L 个最小元素，构建容量为 L 的大顶堆
    // 选取 L 与 lru.size 中更小的那个
    unsigned pq_size = (L < lru_copy.size()) ? L : lru_copy.size();
    // 函数指针
    auto cmp = [this, query](unsigned a, unsigned b) -> bool {
        return (distance_->compare(data_ + dimension_ * a, query, (unsigned)dimension_)
                <
                distance_->compare(data_ + dimension_ * b, query, (unsigned)dimension_));
    };
    // 大顶堆
    std::priority_queue<unsigned, std::vector<unsigned>, decltype(cmp)> pq(cmp);
    // 先往大顶堆压入 pq_size 个元素
    for (unsigned i = 0; i < pq_size; ++i)
    {
        pq.push(lru_copy[i]);
    }
    // 循环比较剩余元素
    for (unsigned j = pq_size; j < lru_copy.size(); ++j)
    {
        // 如果当前的元素 a 小于大顶堆的最大元素 b，说明 a 要入，b 要出
        if (distance_->compare(data_ + dimension_ * lru_copy[j], query, (unsigned)dimension_)
            <
            distance_->compare(data_ + dimension_ * pq.top(), query, (unsigned)dimension_))
        {
            pq.pop();
            pq.push(lru_copy[j]);
        }
    }

    // 从大顶堆中选取所有点（点的个数可能小于 L）
    unsigned tmp_l = 0;
    while (!pq.empty() && tmp_l < pq_size)
    {
        // 大顶堆中的元素倒序加入
        init_ids[pq_size - tmp_l - 1] = pq.top();
        pq.pop();

        flags[init_ids[pq_size - tmp_l - 1]] = true;
        tmp_l++;
    }

    // for (; tmp_l < L && tmp_l < lru_copy.size(); tmp_l++)
    // {
    //  init_ids[tmp_l] = lru_copy[tmp_l];
    //  flags[init_ids[tmp_l]] = true;
    // }

    // 不足 L 个则随机选取节点，直至 init_ids 包括 L 个节点
     while (tmp_l < L)
     {
      unsigned id = rand() % nd_;
      if (flags[id])
      {
        continue;
      }
      flags[id] = true;
      init_ids[tmp_l] = id;
      tmp_l++;
     }

    // TODO 将初始队列中的第一个节点放到 LRU 缓存头部
    // lock
    mtx_lru.lock();
    hot_points_lru->put(init_ids[0]);
    // unlock
    mtx_lru.unlock();

    // TODO print the distance between init_ids[0] and search target
    std::cout << "init_ids[0]: " << init_ids[0] << std::endl;
    std::cout << "distance: " <<
    distance_->compare(data_ + dimension_ * init_ids[0], query, (unsigned)dimension_)
    << std::endl;

    // 将 init_ids 中的节点放入 retset 作为候选节点集
    for (unsigned i = 0; i < init_ids.size(); i++)
    {
      unsigned id = init_ids[i];
      float dist = 
          distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
      retset[i] = Neighbor(id, dist, true);
      // flags[id] = true;
    }

    std::sort(retset.begin(), retset.begin() + L);

    // greedy search
    int k = 0;
    while (k < (int)L)
    {
      int nk = L;
      
      if (retset[k].flag)
      {
        retset[k].flag = false;
        unsigned n = retset[k].id;

        for (unsigned m = 0; m < final_graph_[n].size(); ++m)
        {
          unsigned id = final_graph_[n][m];
          if (flags[id])
          {
            continue;
          }
          flags[id] = 1;
          float dist = 
              distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
          if (dist >= retset[L - 1].distance)
          {
            continue;
          }
          Neighbor nn(id, dist, true);
          int r = InsertIntoPool(retset.data(), L, nn);
          
          if (r < nk)
          {
            nk = r;
          }
        }
      }
      if (nk <= k)
      {
        k = nk;
      }
      else
      {
        ++k;
      }
    }
    // 记录搜索结果
    for (size_t i = 0; i < K; i++)
    {
      indices[i] = retset[i].id;
    }
    
    // 若 flag == true
    // 并未进行热点识别和热点更新
    if (record_query_flag == true)
    {
      record_query(query);
      // 如果 query_list 窗口已满
      if (query_list.size() >= window_size)
      {
        // TODO print
        // std::cout << "Search 中进入到了 window_size 判断条件" << std::endl;
        // std::cout << "Search 中创建热点识别和热点更新线程" << std::endl;
        // 多线程热点识别和热点更新
        std::thread t_tmp(&IndexWADG::identify_and_update, this,
                    query_list, std::ref(parameters), false);
        // 不能阻塞 Search 过程
        // 可能存在问题 Search 函数已经循环结束，热点识别还未结束
        t_tmp.detach();
        // t_tmp.join();
        // 清空 query_list
        query_list.clear();
        // TODO print
        // std::cout << "Search 中清空了 query_list" << std::endl;
      }
    }
  }

  
  // @CS0522
  // 用于多线程的热点识别和热点更新
  void IndexWADG::identify_and_update(std::vector<const float*> old_query_list, 
                                      const Parameters &parameters,
                                      bool record_query_flag)
  {
    // TODO print
    // std::cout << "新的线程 identify_and_update" << std::endl;

    //热点识别
    auto start_identify = std::chrono::high_resolution_clock::now();
    std::vector<float *> query_centroids = get_cluster_centers(old_query_list, parameters, query_list.size());
    // TODO print
    // std::cout << "K-Means 聚类结束" << std::endl;

    // const unsigned K = parameters.Get<unsigned>("K_search");
    auto K = parameters.Get<unsigned>("K_search");

    // search
    // TODO print
    // std::cout << "热点识别过程中聚类中心搜索开始" << std::endl;
    // 保存搜索聚类中心的所有搜索结果
    std::vector<std::vector<unsigned> > search_res;
    for (int i = 0; i < query_centroids.size(); i++)
    {
      std::vector<unsigned> tmp(K);
      search_res.push_back(tmp);
      unsigned *tmp_ = search_res[search_res.size() - 1].data();
      // 创建新线程进行聚类中心的搜索
      // 在热点搜索过程中不会记录搜索请求
      boost::asio::post(tp, boost::bind(&IndexWADG::Search, this, std::ref(query_centroids[i]),
                            std::ref(parameters),
                            std::ref(tmp_),
                            std::ref(record_query_flag)));
      // std::thread t_tmp(&IndexWADG::Search, this, std::ref(query_centroids[i]),
      //                    std::ref(parameters),
      //                    std::ref(tmp_),
      //                    std::ref(record_query_flag));
    }
    // 连接线程（等待 聚类中心search 多线程全部结束）
    tp.join();

    // TODO print
    // std::cout << "聚类中心搜索线程全部连接且结束" << std::endl;

    auto end_identify = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_identify = end_identify - start_identify;
    // TODO print
    // std::cout << "identify hot points time: " << diff_identify.count() << "\n";

    // 热点更新
    auto start_update = std::chrono::high_resolution_clock::now();
    update_hot_points(search_res);
    auto end_update = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_update = end_update - start_update;
    // TODO print
    // std::cout << "update hot points time: " << diff_update.count() << "\n";
  }


  // @CS0522
  // 记录搜索请求
  void IndexWADG::record_query(const float *query)
  {
    // 上锁
    mtx_query_list.lock();
    // 加入搜索请求
    query_list.push_back(query);
    // 解锁
    mtx_query_list.unlock();
    // print
    // std::cout << "query_list size: " << query_list.size() << std::endl;
  }

  // @CS0522
  void IndexWADG::update_hot_points(std::vector<std::vector<unsigned> > &search_res)
  {
    // 上锁
    mtx_lru.lock();
    for (int i = search_res.size() - 1; i >= 0; i--)
    {
      // 每次搜索结果中首个元素应该就是距离最近的点
      hot_points_lru->put(search_res[i][0]);
    }
    // 解锁
    mtx_lru.unlock();

    update_hot_points_count += 1;
  }

  // @CS0522
  // 通过 K-means 获取搜索请求的聚类中心
  // num: querys 长度
  // 需要优化，转化过程时间、空间复杂度较高
  std::vector<float *> IndexWADG::get_cluster_centers(
      std::vector<const float *> querys,
      const Parameters &parameters,
      unsigned num)
  {
    // TODO print
    // std::cout << "identify_and update 中进入到了 get_cluster_centers" << std::endl;
    const unsigned K = parameters.Get<unsigned>("K_search");

    // 调用 dkm 库
    // dkm 接收的数据格式修改为 std::vector<std::vector<T>>
    // 需要先将 const float* query 转化为 std::vector
    int querys_num = querys.size();
    // TODO print
    // std::cout << "querys_num: " << querys_num << std::endl;
    std::vector<std::vector<float> > querys_in_vector(querys_num, std::vector<float>(dimension_));
    for (int i = 0; i < querys_num; i++)
    {
      // 用 std::vector 表示一个 query
      std::vector<float> query(dimension_);
      for (int j = 0; j < dimension_; j++)
      {
        query[j] = querys[i][j];
      }
      // 加入 querys_in_array
      // querys_in_vector.push_back(query);
      querys_in_vector[i] = query;
    } // 转化完成

    // 调用 kmeans 函数
    // 返回的结果是一个Tuple   
    //Tuple[0]: 返回的是数据集聚类中心的列表 (长度为 K)   
    //Tuple[1]: 返回的是输入数据集对应的标签 (归属于哪一个点)
    auto cluster_data = dkm::kmeans_lloyd(querys_in_vector, K);
    std::vector<std::vector<float> > cluster_centers = std::get<0>(cluster_data);
    // 将 std::array 转化回 float *
    std::vector<float *> cluster_centers_result;
    for (int i = 0; i < K; i++)
    {
      float* cluster_center = new float[dimension_];
      for (int j = 0; j < dimension_; j++)
      {
        cluster_center[j] = cluster_centers[i][j];
      }
      // 加入 cluster_centers_result
      cluster_centers_result.push_back(cluster_center);
    } // 转化完成

    // 返回结果
    return cluster_centers_result;
  }
}

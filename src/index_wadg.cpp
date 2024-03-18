/*
 * @Author: Chen Shi
 * @Date: 2024-03-08 14:50:58
 * @Description: implement of IndexWADG class
 */

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
    std::cout << (record_query_flag ? "主 Search 开始" : "聚类中心 Search 开始") << std::endl;

    const unsigned L = parameters.Get<unsigned>("L_search");
    const unsigned K = parameters.Get<unsigned>("K_search");

    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};

    // 从 LRU 缓存中选取离搜索目标最近的 L 个点作为初始队列 init_ids
    unsigned tmp_l = 0;
    // 上锁
    mtx_lru.lock();
    for (; tmp_l < L && tmp_l < hot_points_lru->get_size(); tmp_l++)
    {
      // LRU 队列 get 后会移至 LRU 头部，所以 index 会变
      // 倒序加入
      init_ids[L - tmp_l - 1] = hot_points_lru->get(L - 1);
      flags[init_ids[tmp_l]] = true;
    }
    // 解锁
    mtx_lru.unlock();

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
        std::cout << "Search 中进入到了 window_size 判断条件" << std::endl;
        std::cout << "Search 中创建热点识别和热点更新线程" << std::endl;
        // 多线程热点识别和热点更新
        std::thread t_tmp(&IndexWADG::identify_and_update, this,  
                    query_list, std::ref(parameters), std::ref(indices), false);
        // 不能阻塞 Search 过程
        // 可能存在问题 Search 函数已经循环结束，热点识别还未结束
        t_tmp.detach();
        // t_tmp.join();
        // 清空 query_list
        query_list.clear();
        // TODO print
        std::cout << "Search 中清空了 query_list" << std::endl;
      }
    }
    // 若 flag == false
    // 热点识别和热点更新
    else
    {
      // 上锁
      mtx_search_res.lock();
      // 本次搜索结果的最近的点作为热点之一
      search_res.push_back(indices[0]);
      // 解锁
      mtx_search_res.unlock();
    }
  }

  
  // @CS0522
  // 用于多线程的热点识别和热点更新
  void IndexWADG::identify_and_update(std::vector<const float*> old_query_list, 
                                      const Parameters &parameters, 
                                      unsigned* &indices, 
                                      bool record_query_flag = false)
  {
    // TODO print
    std::cout << "新的线程 identify_and_update" << std::endl;

    //热点识别
    auto start_identify = std::chrono::high_resolution_clock::now();
    std::vector<float *> query_centroids = get_cluster_centers(old_query_list, parameters, query_list.size());
    // TODO print
    std::cout << "K-Means 聚类结束" << std::endl;

    const unsigned K = parameters.Get<unsigned>("K_search");

    // 储存线程
    std::vector<std::thread> thread_container;

    // search
    // TODO print
    std::cout << "热点识别过程中聚类中心搜索开始" << std::endl;
    for (int i = 0; i < query_centroids.size(); i++)
    {
      std::vector<unsigned> tmp(K);
      unsigned *tmp_ = tmp.data();
      // 创建新线程进行聚类中心的搜索
      // 在热点搜索过程中不会记录搜索请求
      std::thread t_tmp(&IndexWADG::Search, this, std::ref(query_centroids[i]), 
                          std::ref(parameters), 
                          std::ref(tmp_), 
                          std::ref(record_query_flag));
      // 加入 container，以便后续释放
      thread_container.push_back(std::move(t_tmp));
    }
    // 连接线程（等待 聚类中心search 多线程全部结束）
    for (int i = 0; i < thread_container.size(); i++)
    {
      thread_container[i].join();
    }

    // TODO print
    std::cout << "聚类中心搜索线程全部连接且结束" << std::endl;

    auto end_identify = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_identify = end_identify - start_identify;
    std::cout << "identify hot points time: " << diff_identify.count() << "\n";

    // 热点更新
    auto start_update = std::chrono::high_resolution_clock::now();
    update_hot_points();
    auto end_update = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_update = end_update - start_update;
    std::cout << "update hot points time: " << diff_update.count() << "\n";

    // thread_container.clear();

    // TODO print
    std::cout << "更新热点次数: " << update_hot_points_count << std::endl;
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
  void IndexWADG::update_hot_points()
  {
    // 上锁
    mtx_lru.lock();
    for (int i = search_res.size() - 1; i >= 0; i--)
    {
      hot_points_lru->put(search_res[i]);
    }
    // 解锁
    mtx_lru.unlock();

    // 上锁
    mtx_search_res.lock();
    // 清空 search_res
    search_res.clear();
    // 解锁
    mtx_search_res.unlock();

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
    std::cout << "identify_and update 中进入到了 get_cluster_centers" << std::endl; 
    const unsigned K = parameters.Get<unsigned>("K_search");

    // 调用 dkm 库
    // dkm 接收的数据格式修改为 std::vector<std::vector<T>>
    // 需要先将 const float* query 转化为 std::vector
    int querys_num = querys.size();
    std::cout << "querys_num: " << querys_num << std::endl;
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

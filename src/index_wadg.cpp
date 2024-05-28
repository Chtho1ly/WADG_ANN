/*
 * @Author: Chen Shi
 * @Date: 2024-03-08 14:50:58
 * @Description: implement of IndexWADG class
 */

#include <map>
#include <cstring>
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
  // @CS0522
  IndexWADG::IndexWADG(const size_t dimension, size_t n, Metric m, const unsigned K,
                       Index *initializer)
      : IndexNSG(dimension, n, m, initializer) 
      {
        // 初始化 max_hot_points_num
        max_hot_points_num = 200;
        // 初始化 window_size 搜索请求记录窗口大小
        // window_size >= cluster_num
        window_size = 100;
        // 初始化 cluster_num 聚类中心数
        // 主要影响因素
        cluster_num = 10;

        // DEBUG
        // 初始化 pre 数组
        if (DEBUG)
        {
          pre = (int *) malloc(sizeof(int) * 1000000);
          // 值为 -1
          memset(pre, 0b11111111, sizeof(int) * 1000000);
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
  void IndexWADG::Set_lru()
  {
      // 初始化 LRU 队列
      // (key, value): (下标, 有效热点 id)
      hot_points_lru = new LRUCache(max_hot_points_num);
      // 将导航点的全部邻居放入 LRU 队列
      unsigned ep = get_ep_();
      // DEBUG
      if (DEBUG)
      {
        std::cout << "\n===== DEBUG: LRU Init =====\n" << std::endl;
        std::cout << "Navigate node: " << std::endl;
        std::cout << "id: " << ep << std::endl << std::endl;
        std::cout << "Neighbor points of navigate node: " << std::endl;
      }
      for (unsigned i = 0; i < max_hot_points_num && i < final_graph_[ep].size(); i++)
      {
          hot_points_lru->put(final_graph_[ep][i]);
          // DEBUG
          if (DEBUG)
          {
            std::cout << final_graph_[ep][i] << " ";
          }
      }
      // DEBUG
      if (DEBUG)
      {
        std::cout << std::endl;
        // print lru
        std::cout << "\nInitial LRU: " << std::endl;
        hot_points_lru->print_lru_cache();
      }
      // 与 nsg 中 init_ids 加入的导航点一致
  }


  // @CS0522
  // 修改为多线程
  void IndexWADG::Search(const float *query, const Parameters &parameters, unsigned*& indices,
                          bool record_query_flag)
  {
    auto L = parameters.Get<unsigned>("L_search");
    auto K = parameters.Get<unsigned>("K_search");

    std::vector<Neighbor> retset;
    boost::dynamic_bitset<> flags{nd_, 0};

    // 记录最长搜索路径
    int *mlen;
    // DEBUG
    if (DEBUG)
    {
      mlen = (int *) malloc(sizeof(int) * 1000000);
      memset(mlen, 0b00000000, sizeof(int) * 1000000);
    }

    unsigned init_size = (L < hot_points_lru->get_size()) ? L : hot_points_lru->get_size();

    // lock
    mtx_lru.lock();
    // 获取 lru 数据副本
    auto lru_copy = hot_points_lru->get_cache();
    // unlock
    mtx_lru.unlock();

    /**
    * 从 LRU 缓存中选取离搜索目标最近的 init_size 个点作为初始队列
    * pair + priority_queue 实现
    * std::pair<unsigned, float>: (id, distance)
    * vector 存储 id 和该点到 query 的距离
    */
    std::vector<std::pair<unsigned, float> > pairs;
    for (int i = 0; i < lru_copy.size(); ++i)
    {
      unsigned id = lru_copy[i];
      pairs.emplace_back(std::make_pair(id, distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_)));
    }
    // 销毁 lru_copy
    lru_copy.clear();
    lru_copy.shrink_to_fit();
    // 求前 init_size 个最小元素，构建大顶堆
    // 函数指针
    auto cmp = [](std::pair<unsigned, float> &a, std::pair<unsigned, float> &b) -> bool {
          return a.second < b.second;
    };
    std::priority_queue<std::pair<unsigned, float>, std::vector<std::pair<unsigned, float> >, decltype(cmp)> pq(cmp);
    // 大顶堆中压入 init_size 个元素
    for (int i = 0; i < init_size; ++i)
    {
      pq.push(pairs[i]);
    }
    // 循环比较剩余元素
    for (int j = init_size; j < pairs.size(); ++j)
    {
      if (pairs[j].second < pq.top().second)
      {
        pq.pop();
        pq.push(pairs[j]);
      }
    }

    // 大根堆中的所有点加入 retset 中
    while (!pq.empty())
    {
      auto id = pq.top().first;
      auto dist = pq.top().second;
      // 此时 retset 顺序为 distance 从大到小
      retset.emplace_back(Neighbor(id, dist, true));
      pq.pop();

      // set flags
      flags[id] = true;
    }

    // 加入后 retset 顺序为 distance 从大到小，进行反转
    std::reverse(retset.begin(), retset.end());

    // 距离 query 最近的 id 放到 LRU 缓存头部
    // lock
    mtx_lru.lock();
    hot_points_lru->put(retset[0].id);
    // unlock
    mtx_lru.unlock();

    // DEBUG 且为主 Search
    if (DEBUG && record_query_flag == true)
    {
      std::cout << std::endl << "====== DEBUG: Search for query " 
                << this->search_points_counts.size() << " =====" << std::endl;
      std::cout << "\nHot points: " << std::endl;
      int num = hot_points_lru->get_size();
      for (int i = 0; i < num; ++i)
      {
        unsigned id = hot_points_lru->visit(i);
        float dist = distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
        std::cout << "id: " << std::setw(6) << id << ", dis: " << std::setw(6) << dist << std::endl;
      }
      std::cout << std::endl;
      std::cout << "Initial retset (length = " << retset.size() << "): " << std::endl;
      for (int i = 0; i < retset.size(); ++i)
      {
        std::cout << "id: " << std::setw(6) << retset[i].id << ", dis: " << std::setw(6) << retset[i].distance << std::endl;
        // retset 中的点的前驱为导航点，路径为 1
        pre[retset[i].id] = get_ep_();
        mlen[retset[i].id] = 1;
      }
      // std::cout << std::endl << std::endl << "开始检索...";
      std::cout << std::endl << "===== DEBUG: Greedy search =====\n" << std::endl;
    }

    // greedy search
    int k = 0;
    // 检索点数量
    // LRU 都计算了一遍
    int search_points_count = pairs.size();
    while (k < (int)L)
    {
        int len = (L < retset.size()) ? L : retset.size();
        int nk = len;

        if (retset[k].flag)
        {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            // DEBUG 且为主 Search
            if (DEBUG && record_query_flag == true)
            {
              float dist = distance_->compare(query, data_ + dimension_ * n, (unsigned)dimension_);
              std::cout << "Level: " << std::setw(2) << mlen[n] << " - "
                        << "id: " << std::setw(6) << n << ", dis: " 
                        << std::setw(6) << dist << ", pre: " << std::setw(6) << pre[n] << " " << std::endl; 
              // std::cout << "加入 retset 的邻居点: " << std::endl;
            }

            for (unsigned m = 0; m < final_graph_[n].size(); ++m)
            {
                unsigned id = final_graph_[n][m];

                // DEBUG 更新每个点的最长搜索路径
                if (DEBUG && record_query_flag == true)
                {
                  mlen[id] = std::max(mlen[id], mlen[n] + 1);
                }

                if (flags[id])
                {
                    continue;
                }
                flags[id] = 1;
                float dist =
                        distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);

                // DEBUG
                // 统计检索点数量
                // 开启热点识别且为主 Search 或 未开启热点识别
                if (DEBUG)
                {
                  if ((HOT_POINTS && record_query_flag == true) || !HOT_POINTS)
                  {
                    ++search_points_count;
                  }
                }

                if (dist >= retset[len - 1].distance)
                {
                    continue;
                }

                Neighbor nn(id, dist, true);
                auto r = InsertIntoPool(retset, len, nn);

                // DEBUG 且为主 Search
                if (DEBUG && record_query_flag == true)
                {
                  // 更新前驱
                  pre[id] = n;
                  // std::cout << "id: " << id << ", dis: " << dist << ", pre: "<< pre[id] <<", 插入位置: " << r << ", 插入后 retset 长度: " << retset.size() << std::endl;
                }

                // update len
                len = (L < retset.size()) ? L : retset.size();

                if (r < nk)
                {
                    // 在 r 位置插入了一个 nn
                    // 回溯
                    nk = r;
                }
            }
        }
        // 回溯
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
    for (auto i = 0; i < K; i++)
    {
      indices[i] = retset[i].id;
    }

    // DEBUG
    // 记录本次 Search 的最长搜索路径
    if (DEBUG && record_query_flag == true)
    {
      auto max_len = std::max_element(mlen, mlen + 1000000);
      // std::cout << std::endl << "Max search length of current query: " << *max_len << std::endl;
      std::cout << std::endl << *max_len << std::endl;
      this->max_search_lengths.push_back(*max_len);
    }

    // DEBUG
    // 统计检索点数量
    // 开启热点识别且为主 Search 或 未开启热点识别
    if (DEBUG)
    {
      if (!HOT_POINTS || (HOT_POINTS && record_query_flag == true))
      {
        // std::cout << "Search points count of current query: " << search_points_count << std::endl;
        std::cout << search_points_count << std::endl;
        this->search_points_counts.push_back(search_points_count);
      }
    }
    
    // 开启热点识别且为主 Search
    // 进行热点识别和热点更新
    if (HOT_POINTS && record_query_flag == true)
    {
      record_query(query);
      // 如果 query_list 窗口已满
      if (query_list.size() >= window_size)
      {
        // 多线程热点识别和热点更新
        std::thread t_tmp(&IndexWADG::identify_and_update, this,
                    query_list, std::ref(parameters), false);
        // 不能阻塞 Search 过程
        // 可能存在问题 Search 函数已经循环结束，热点识别还未结束
        t_tmp.detach();
        // t_tmp.join();
        // 清空 query_list
        query_list.clear();
        // window count + 1
        ++window_count;
      }
    }
  }


  // @CS0522
  // 用于多线程的热点识别和热点更新
  void IndexWADG::identify_and_update(std::vector<const float*> old_query_list, 
                                      const Parameters &parameters,
                                      bool record_query_flag)
  {
    //热点识别
    std::vector<float *> query_centroids = get_cluster_centers(old_query_list, parameters, query_list.size());

    auto K = parameters.Get<unsigned>("K_search");

    // search
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
    }
    // 连接线程（等待 聚类中心search 多线程全部结束）
    tp.join();

    // 热点更新
    update_hot_points(search_res);
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
    auto K = parameters.Get<unsigned>("K_search");

    // 调用 dkm 库
    // dkm 接收的数据格式修改为 std::vector<std::vector<T>>
    // 需要先将 const float* query 转化为 std::vector
    int querys_num = querys.size();
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
    auto cluster_data = dkm::kmeans_lloyd(querys_in_vector, cluster_num);
    auto cluster_centers = std::get<0>(cluster_data);
    auto labels = std::get<1>(cluster_data);

    // 将 std::array 转化回 float *
    std::vector<float *> cluster_centers_result;
    for (int i = 0; i < cluster_num; i++)
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

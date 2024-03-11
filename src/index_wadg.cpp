/*
 * @Author: Chen Shi
 * @Date: 2024-03-08 14:50:58
 * @Description: implement of IndexWADG class
 */

#include "efanna2e/index_wadg.h"
// k-means 聚类库，性能好
#include "efanna2e/dkm.hpp"
#include <vector>
#include <array>
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
  IndexWADG::IndexWADG(const size_t dimension, const size_t n, Metric m,
                       Index *initializer)
      : IndexNSG(dimension, n, m, initializer) 
      {
        // TODO 初始化 max_hot_points_num, 该值应该大于 K
        
        // TODO 初始化 window_size 搜索请求记录窗口大小
        
        // TODO 初始化 cluster_num 聚类中心数
        
        // TODO 初始化 query_list 窗口内搜索请求记录

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
  void IndexWADG::Search(const float *query, const Parameters &parameters, unsigned*& indices)
  {
    const unsigned L = parameters.Get<unsigned>("L_search");
    const unsigned K = parameters.Get<unsigned>("K_search");

    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};

    // 从 LRU 缓存中选取离搜索目标最近的 L 个点作为初始队列 init_ids
    unsigned tmp_l = 0;
    for (; tmp_l < L && tmp_l < hot_points_lru->get_size(); tmp_l++)
    {
      init_ids[tmp_l] = hot_points_lru->get(tmp_l);
      flags[init_ids[tmp_l]] = true;
    }

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
      search_res[i] = retset[i].id;
    }
  }

  // @CS0522
  // 记录搜索请求
  // 若请求记录窗口已满则进行热点更新，并删除旧的记录
  void IndexWADG::record_query(const float *query)
  {
    // 加入搜索请求
    query_list.push_back(query);
    // 如果 query_list 已满
    if (query_list.size() >= window_size)
    {
      // TODO 热点更新
      // TODO 需要放在这里，还是放在 test_index_wadg_search 中
      update_hot_points();
      // 删除旧的访问记录
      query_list.clear();
    }
  }

  // @CS0522
  void IndexWADG::update_hot_points()
  {
    // search_res 保存了记录的搜索结果
    // search_res 中应该是按照距离从近到远保存的搜索结果
    // K 个搜索结果放到 LRU 头部
    unsigned K = sizeof(search_res) / sizeof(search_res[0]);
    for (int i = K - 1; i >= 0; i--)
    {
      unsigned hot_id = search_res[i];
      hot_points_lru->put(hot_id);
    }
  }

  // @CS0522
  // 通过 K-means 获取搜索请求的聚类中心
  // num: querys 长度
  // 需要优化，转化过程时间、空间复杂度较高
  std::vector<float *> IndexWADG::get_cluster_centers(
      std::vector<float *> querys,
      const Parameters &parameters,
      unsigned num)
  {
    const unsigned K = parameters.Get<unsigned>("K_search");

    // 调用 dkm 库
    // dkm 接收的数据格式 std::vector< std::array<type, n> >
    // 需要先将 const float* query 转化为 std::array
    const unsigned dimension_of_query = sizeof(querys[0]) / sizeof(querys[0][0]);
    // 转化 querys
    std::vector<std::array<float, dimension_of_query> > querys_in_array;
    int querys_num = querys.size();
    // int querys_num = num;
    for (int i = 0; i < querys_num; i++)
    {
      // 用 std::array 表示一个 query
      std::array<float, dimension_of_query> query;
      for (int j = 0; j < dimension_of_query; j++)
      {
        query[j] = querys[i][j];
      }
      // 加入 querys_in_array
      querys_in_array.push_back(query);
    } // 转化完成

    // 调用 kmeans 函数
    // 返回的结果是一个Tuple   
    //Tuple[0]: 返回的是数据集聚类中心的列表 (长度为 K)   
    //Tuple[1]: 返回的是输入数据集对应的标签 (归属于哪一个点)
    auto cluster_data = dkm::kmeans_lloyd(querys_in_array, K);
    std::vector<std::array<float, dimension_of_query> > cluster_centers = std::get<0>(cluster_data);
    // 将 std::array 转化回 float *
    std::vector<float *> cluster_centers_result;
    for (int i = 0; i < K; i++)
    {
      float* cluster_center = new float[dimension_of_query];
      for (int j = 0; j < dimension_of_query; j++)
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

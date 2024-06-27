//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_INDEX_H
#define EFANNA2E_INDEX_H

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include "distance.h"
#include "parameters.h"

// DEBUG
#define DEBUG true
// print
#define PRINT_INFO true
// enable hot points
#define HOT_POINTS true
// nsg random points
#define NSG_RANDOM false

namespace efanna2e {

class Index {
 public:
  explicit Index(const size_t dimension, const size_t n, Metric metric);


  virtual ~Index();

  virtual void Build(size_t n, const float *data, const Parameters &parameters) = 0;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) = 0;

  virtual void Save(const char *filename) = 0;

  virtual void Load(const char *filename) = 0;

  inline bool HasBuilt() const { return has_built; }

  inline size_t GetDimension() const { return dimension_; };

  inline size_t GetSizeOfDataset() const { return nd_; }

  inline const float *GetDataset() const { return data_; }

  // @CS0522
  std::vector<std::pair<double, double> > get_search_times() { return search_times; }
  std::vector<std::pair<unsigned, float> > get_start_points() { return start_points; }


 protected:
  const size_t dimension_;
  const float *data_ = nullptr;
  size_t nd_;
  bool has_built;
  Distance* distance_ = nullptr;

  // @CS0522
  // 存储搜索时间
  // pair: (time before greedy search, time of greedy search)
  std::vector<std::pair<double, double> > search_times;
  // 存储起点的距离
  // pair: (id, distance)
  std::vector<std::pair<unsigned, float> > start_points;

  // 记录每次 Search 的检索点数量
  std::vector<int> search_points_counts;
  // 记录每次 Search 的最长搜索路径
  std::vector<int> max_search_lengths;
  // 记录前驱的 pre 数组
  // 这里想初始化值为 -1，所以用 int。int 范围应该是够 sift 的
  int *pre;
  // 记录最长搜索路径的 mlen 数组
  int *mlen;
};

}

#endif //EFANNA2E_INDEX_H

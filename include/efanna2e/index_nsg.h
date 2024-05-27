#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>

namespace efanna2e {

class IndexNSG : public Index {
 public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexNSG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;

  // @CS0522
  unsigned get_ep_() { return this->ep_; }

  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices);
  void OptimizeGraph(float* data);

  // @CS0522
  std::vector<int> get_try_enter_retset_points_counts()
  {
    return this->try_enter_retset_points_counts;
  }

  std::vector<int> get_max_search_lengths()
  {
      return this->max_search_lengths;
  }

  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_ = nullptr;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_ = nullptr;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;

    // @CS0522
    // 记录每次 Search 的检索点数量
    std::vector<int> try_enter_retset_points_counts;

    // 记录每次 Search 的最长搜索路径
    std::vector<int> max_search_lengths;

    // 记录前驱的 pre 数组
    // 这里想初始化值为 -1，所以用 int。int 范围应该是够 sift 的
    int *pre;
};
}

#endif //EFANNA2E_INDEX_NSG_H

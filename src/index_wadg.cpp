#include "efanna2e/index_wadg.h"

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
  IndexWADG::IndexWADG(const size_t dimension, const size_t n, Metric m,
                       Index *initializer)
      : IndexNSG(dimension, n, m, initializer) {}

  IndexWADG::~IndexWADG()
  {
  }
  void IndexWADG::Search(const float *query, const Parameters &parameters, unsigned *indices)
  {
  }
  void IndexWADG::Set_data(const float *x)
  {
  }
  void IndexWADG::record_query(const float *query)
  {
  }
  void IndexWADG::update_hot_points()
  {
  }
  std::vector<float *> IndexWADG::get_cluster_centers(
      std::vector<float *> querys,
      const Parameters &parameters,
      unsigned num)
  {
  }
}

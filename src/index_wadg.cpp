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
}

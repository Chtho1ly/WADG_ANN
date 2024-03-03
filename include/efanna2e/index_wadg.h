#ifndef EFANNA2E_INDEX_WADG_H
#define EFANNA2E_INDEX_WADG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index_nsg.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>

namespace efanna2e
{

    class IndexWADG : public IndexNSG
    {
    public:
        explicit IndexWADG(const size_t dimension, const size_t n, Metric m, Index *initializer);

        virtual ~IndexWADG();
        virtual void Search(
            const float *query,
            const float *x,
            size_t k,
            const Parameters &parameters,
            unsigned *indices) override;
    };
}

#endif // EFANNA2E_INDEX_WADG_H

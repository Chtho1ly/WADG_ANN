#ifndef EFANNA2E_INDEX_WADG_H
#define EFANNA2E_INDEX_WADG_H

// @CS0522
#include "efanna2e/lru_cache.h"
#include <tuple>

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index_nsg.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
// @CS0522
// 使用 boost 线程池库
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <queue>
#include <stack>
#include <mutex>
#include <thread>

namespace efanna2e
{
    // @CS0522
    // 线程池
    // 用于热点识别和热点更新中的多个聚类中心搜索
    // 主search 中的热点识别和热点更新的多线程没有使用线程池
    // 主search 使用线程池，会导致主search 等待线程池内的线程执行结束才继续执行
    // 设定线程池大小
    static boost::asio::thread_pool tp(8);

    class IndexWADG : public IndexNSG
    {
    public:
        explicit IndexWADG(const size_t dimension, const size_t n, Metric m, 
                        const unsigned K, Index *initializer);

        virtual ~IndexWADG();

        // 记录历史查询请求并实现热点的更新
        // 与NSG相比，使用parameters传递k_search，并且将_data的初始化放在了Set_data函数中
        
        /* 暂时没想到一个好的解决办法，
         * 来解决当 主Search 和 聚类Search 同时搜索时，
         * 如何判断哪个是主哪个是聚类的，
         * 使得聚类Search不能对query_list写而主Search可以。
         */ 
        virtual void Search(
            const float *query,
            const Parameters &parameters,
            unsigned* &indices,
            bool record_query_flag);

        virtual void Set_data(const float *x);

        // 需要在读取 final_graph 之后才能把导航点全部邻居放入 LRU 中
        virtual void Set_lru();

        int get_update_hot_points_count()
        {
            return this->update_hot_points_count;
        }

        int get_window_count()
        {
            return this->window_count;
        }

        std::tuple<unsigned, unsigned, unsigned> get_hyperparams()
        {
            // W, C, L
            return std::make_tuple(this->window_size, this->cluster_num, this->max_hot_points_num);
        }

        std::vector<int> get_try_enter_retset_points_counts()
        {
            return this->try_enter_retset_points_counts;
        }

        std::vector<int> get_max_search_lengths()
        {
            return this->max_search_lengths;
        }

    protected:
        // 记录搜索请求
        // 若请求记录窗口已满则进行热点更新，并删除旧的记录
        virtual void record_query(const float *query);
        // 更新热点
        virtual void update_hot_points(std::vector<std::vector<unsigned> > &search_res);
        // 通过K-means获取搜索请求的聚类中心
        virtual std::vector<float *> get_cluster_centers(
            std::vector<const float *> querys,
            const Parameters &parameters,
            unsigned num);
        // @CS0522
        // 多线程
        // 用于热点识别和热点更新
        /**
         * old_query_list 用于复制一份 query_list，
         * 防止 Search 和 get_cluster_centers 同时修改 query_list
        */
        virtual void identify_and_update(
            std::vector<const float*> old_query_list, 
            const Parameters &parameters,
            bool record_query_flag);

    private:
        unsigned max_hot_points_num;          // 最大热点数
        unsigned window_size;                 // 搜索请求记录窗口大小，远大于 K
        unsigned cluster_num;                 // 聚类中心数
        // std::vector<unsigned> hot_points_lru; // 包含全部有效热点id的LRU队列
        // @CS0522
        LRUCache *hot_points_lru;             // 包含全部有效热点id的LRU队列
        // @CS0522
        std::mutex mtx_lru;                   // LRU 队列的互斥锁
        // @CS0522
        // "float *" -> "const float *"
        std::vector<const float *> query_list;      // 窗口内搜索请求记录
        std::mutex mtx_query_list;

        // 记录更新热点次数
        int update_hot_points_count = 0;
        // 记录经过了几个时间窗口
        int window_count = 0;
        // 记录每次主 Search 的检索点数量
        std::vector<int> try_enter_retset_points_counts;
        // 记录每次主 Search 的最长搜索路径
        std::vector<int> max_search_lengths;

        // 记录前驱的 pre 数组
        // 这里想初始化值为 -1，所以用 int。int 范围应该是够 sift 的
        int *pre;
    };
}

#endif // EFANNA2E_INDEX_WADG_H

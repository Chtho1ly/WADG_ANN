/*
 * @Author: Chen Shi
 * @Date: 2024-03-10 00:03:44
 * @Description: implement of LRUCache class
 */
#include "efanna2e/lru_cache.h"

namespace efanna2e 
{
    #define LRU_SIZE 5
      
    LRUCache::LRUCache(unsigned capacity)
    {
        this->capacity = capacity;
    }

    unsigned LRUCache::get_capacity()
    {
        return this->capacity;
    }

    unsigned LRUCache::get_size()
    {
        return this->cache.size();
    }

    void LRUCache::add_to_head(unsigned id)
    {
        if (this->cache.size() < this->capacity)
        {
            this->cache.insert(cache.begin(), id);
        }
    }

    void LRUCache::remove_id(unsigned id)
    {
        // 查找指定 id
        std::vector<unsigned>::iterator it = std::find(cache.begin(), cache.end(), id);
        // 删除该节点
        if (it != cache.end())
        {
            cache.erase(it);
        }
    }
       
    void LRUCache::move_to_head(unsigned id)
    {
        // 删除该 id
        remove_id(id);
        // 在 head 位置添加该 id
        add_to_head(id);
    }
       
    unsigned LRUCache::remove_tail()
    {
        unsigned tail_id = this->cache.back();
        this->cache.pop_back();
        return tail_id;
    }

    void LRUCache::put(unsigned id)
    {
        // 判断是否已存在该 id
        std::vector<unsigned>::iterator it = find(cache.begin(), cache.end(), id);
        // 不存在该 id
        if (it == cache.end())
        {
            // 如果当前队列数量超过 LRU cache 容量
            if (this->cache.size() >= this->capacity)
            {
                // 释放 tail 节点
                unsigned tail_id = remove_tail();
            }
            // 插入到 head
            add_to_head(id);
        }
        // 存在该 id
        else
        {
            // 移至 head
            move_to_head(id);
        }
    }
       
    unsigned LRUCache::get(unsigned index)
    {
        index = index % this->capacity;
        // if (index >= this->capacity)
        // {
        //     return -1;
        // }
        unsigned get_id = this->cache[index];
        // 最近被访问过，移至 head
        move_to_head(get_id);
        // 返回该值
        return get_id;
    }

    void LRUCache::print_lru_cache()
    {
        for (unsigned i = 0; i < this->cache.size(); i++)
        {
            std::cout << "(" << i << ", " << this->cache[i] << ")"; 
        }
        std::cout << std::endl;
    }
}
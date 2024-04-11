/*
 * @Author: Chen Shi
 * @Date: 2024-03-10 00:03:44
 * @Description: LRUCache class
 */
#ifndef EFANNA2E_LRU_CACHE_H
#define EFANNA2E_LRU_CACHE_H

#include <iostream>
#include <vector>
#include <algorithm>

namespace efanna2e
{
    class LRUCache
    {
        private:
            unsigned capacity;
            std::vector<unsigned> cache;
        
        public: 
            /**
             * @name: LRUCache
             * @msg: constructor
             * @param {unsigned} capacity: LRU capacity
             * @return {*}
             */              
            LRUCache(unsigned capacity);

            /**
             * @name: get_capacity
             * @msg: get the capacity of LRU queue
             * @return {unsigned}: capacity
             */
            unsigned get_capacity();
        
            /**
             * @name: get_size
             * @msg: get the current size of LRU queue
             * @return {unsigned}: size
             */
            unsigned get_size();

            /**
             * @name: get_cache
             * @msg: get the cache of LRU queue
             * @return {unsigned}: cache
             */
             std::vector<unsigned> get_cache();

            /**
             * @name: add_to_head
             * @msg: add id to the head of LRU queue
             * @param {unsigned} id: hot id
             * @return {*}
             */
            void add_to_head(unsigned id);

            /**
             * @name: remove_id
             * @msg: remove target id in LRU queue
             * @param {unsigned} id: hot id
             * @return {*}
             */
            void remove_id(unsigned id);
       
            /**
             * @name: move_to_head
             * @msg: move target id to the head of LRU queue
             * @param {unsigned} id: hot id
             * @return {*}
             */
            void move_to_head(unsigned id);
      
            /**
             * @name: remove_tail
             * @msg: remove the tail of LRU queue
             * @return {*}
             */
            unsigned remove_tail();

            /**
             * @name: put
             * @msg: insert hot id
             * @param {unsigned} id: hot id
             * @return {*}
             */
            void put(unsigned id);
     
            /**
             * @name: get
             * @msg: get the id at the index position in LRU queue
             * @param {unsigned} index: target position
             * @return {*}
             */
            unsigned get(unsigned index);

            /**
            * @name: visit
            * @msg: visit the id at the index position in LRU queue
            *       without promoting it to the front
            * @param {unsigned} index: target position
            * @return {*}
            */
            unsigned visit(unsigned index);

            /**
             * @name: print_lru_cache
             * @msg: print the LRU queue
             * @return {*}
             */
            void print_lru_cache();
    };  
}

#endif
//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_GRAPH_H
#define EFANNA2E_GRAPH_H

#include <cstddef>
#include <vector>
#include <mutex>
#include <tuple>

namespace efanna2e {

struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

// @CS0522
struct Node
{
    Neighbor nn;
    Node *next;
    
    // constructor
    Node(Neighbor new_nn)
    {
        this->nn = new_nn;
        this->next = nullptr;
    }
    Node(Neighbor new_nn, Node *new_next)
    {
        this->nn = new_nn;
        this->next = new_next;
    }
};

// @CS0522
struct Neighbors
{
    Node *head;
    unsigned capacity;
    unsigned size;
    
    // constructor
    Neighbors(unsigned capacity_)
    {
        // 带头节点
        Node *header = new Node(Neighbor(-1, -1, false));
        this->head = header;
        this->capacity = capacity_;
        this->size = 0;
    }

    // get tail pointer
    Node* get_tail()
    {
        auto pointer = this->head;
        auto count = this->size;
        while (count > 0)
        {
            pointer = pointer->next;
            --count;
        }

        return pointer;
    }

    // insert at head
    void insert_head(Neighbor new_nn)
    {
        if (this->size >= this->capacity)
        {
            remove_tail();
        }

        Node *new_node = new Node(new_nn);
        new_node->next = this->head->next;
        this->head->next = new_node;
        
        ++this->size; 
    }

    // insert at tail
    void insert_tail(Neighbor new_nn)
    {
        if (this->size >= this->capacity)
        {
            remove_tail();
        }

        auto count = this->size;
        auto pointer = this->head;
        while (count > 0)
        {
            pointer = pointer->next;
            --count;
        }

        Node *new_node = new Node(new_nn);
        pointer->next = new_node;
        
        ++this->size; 
    }

    // insert at target position
    void insert(unsigned index, Neighbor new_nn)
    {
        if (index >= this->size)
        {
            return;
        }

        if( this->size >= this->capacity)
        {
            remove_tail();
        }

        auto pointer = this->head;
        // 在该节点后面插入
        while (index > 0)
        {
            pointer = pointer->next;
            --index;
        }

        Node *new_node = new Node(new_nn);
        new_node->next = pointer->next;
        pointer->next = new_node;

        ++this->size;
    }

    // remove the head
    Neighbor remove_head()
    {
        if (this->size <= 0)
        {
            return Neighbor();
        }
        Node *remove_node = this->head->next;
        this->head->next = remove_node->next;

        Neighbor remove_nn(remove_node->nn.id, remove_node->nn.distance, remove_node->nn.flag);
        delete remove_node;

        --this->size;

        return remove_nn;
    }

    // remove the tail
    Neighbor remove_tail()
    {
        if (this->size <= 0)
        {
            return Neighbor();
        }

        auto count = this->size;
        auto pointer = this->head;
        // 要删除的是 next
        while (count > 1)
        {
            pointer = pointer->next;
            --count;
        }

        Node *remove_node = pointer->next;
        pointer->next = nullptr;

        Neighbor remove_nn(remove_node->nn.id, remove_node->nn.distance, remove_node->nn.flag);
        delete remove_node;

        --this->size;

        return remove_nn;
    }

    // remove at target position
    Neighbor remove(unsigned index)
    {
        if (index >= this->size || this->size <= 0)
        {
            return Neighbor();
        }

        auto pointer = this->head;
        // 要删除的是 next
        while (index > 0)
        {
            pointer = pointer->next;
            --index;
        }

        Node *remove_node = pointer->next;
        pointer->next = remove_node->next;

        Neighbor remove_nn(remove_node->nn.id, remove_node->nn.distance, remove_node->nn.flag);
        delete remove_node;

        --this->size;

        return remove_nn;
    }

    // update
    void update(unsigned index, Neighbor new_nn)
    {
        if (index >= this->size || this->size > this->capacity)
        {
            return;
        }

        auto pointer = this->head->next;
        while (index > 0)
        {
            pointer = pointer->next;
            --index;
        }

        pointer->nn = new_nn;
    }

    // get
    Node* get(unsigned index)
    {
        if (index > this->size)
        {
            return nullptr;
        }

        auto pointer = this->head->next;
        while (index > 0)
        {
            pointer = pointer->next;
            --index;
        }

        return pointer;
    }


    // swap for sort
    void swap_nn(Node *a, Node *b)
    {
        Neighbor tmp = a->nn;
        a->nn = b->nn;
        b->nn = tmp;
    }

    // Quick Sort
    // 按 distance 从小到大排序
    void sort(Node *head, Node *tail)
    {
        // 递归出口
        if (head == tail || head == nullptr || head == tail->next)
        {
            return;
        }
	    Node* p, * q, * pre;
	    p = q = pre = head;
	    while (q != tail) 
        {
	    	q = q->next; // 对于一个链表只遍历一遍
	    	if (q->nn < head->nn) {
	    		// 如果q的 node 大于 base，则放在 p 左边
	    		pre = p;
	    		p = p->next;
	    		swap_nn(p, q);
	    	}
	    }
	    swap_nn(head, p);
	    sort(head, pre);
	    sort(p->next, tail);
    }

    // print
    void print()
    {
        int count = this->size;
        Node *pointer = this->head->next;
        while (count > 0)
        {
            std::cout << "(" << pointer->nn.id << ", " 
                        << pointer->nn.distance << ", "
                        << pointer->nn.flag << ")" << " ";
            pointer = pointer->next;
            --count;
        }
        std::cout << std::endl;
    }
};

typedef std::lock_guard<std::mutex> LockGuard;
struct nhood{
  std::mutex lock;
  std::vector<Neighbor> pool;
  unsigned M;

  std::vector<unsigned> nn_old;
  std::vector<unsigned> nn_new;
  std::vector<unsigned> rnn_old;
  std::vector<unsigned> rnn_new;

  nhood(){}
  nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N){
    M = s;
    nn_new.resize(s * 2);
    GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N);
    nn_new.reserve(s * 2);
    pool.reserve(l);
  }

  nhood(const nhood &other){
    M = other.M;
    std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
    nn_new.reserve(other.nn_new.capacity());
    pool.reserve(other.pool.capacity());
  }
  void insert (unsigned id, float dist) {
    LockGuard guard(lock);
    if (dist > pool.front().distance) return;
    for(unsigned i=0; i<pool.size(); i++){
      if(id == pool[i].id)return;
    }
    if(pool.size() < pool.capacity()){
      pool.push_back(Neighbor(id, dist, true));
      std::push_heap(pool.begin(), pool.end());
    }else{
      std::pop_heap(pool.begin(), pool.end());
      pool[pool.size()-1] = Neighbor(id, dist, true);
      std::push_heap(pool.begin(), pool.end());
    }

  }

  template <typename C>
  void join (C callback) const {
    for (unsigned const i: nn_new) {
      for (unsigned const j: nn_new) {
        if (i < j) {
          callback(i, j);
        }
      }
      for (unsigned j: nn_old) {
        callback(i, j);
      }
    }
  }
};

struct SimpleNeighbor{
  unsigned id;
  float distance;

  SimpleNeighbor() = default;
  SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance}{}

  inline bool operator<(const SimpleNeighbor &other) const {
      return distance < other.distance;
  }
};
struct SimpleNeighbors{
  std::vector<SimpleNeighbor> pool;
};

static inline int InsertIntoPool (Neighbor *addr, unsigned K, Neighbor nn) {
  // find the location to insert
  int left=0,right=K-1;
  if(addr[left].distance>nn.distance){
    memmove((char *)&addr[left+1], &addr[left],K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if(addr[right].distance<nn.distance){
    addr[K] = nn;
    return K;
  }
  // 二分查找插入点
  while(left<right-1){
    int mid=(left+right)/2;
    if(addr[mid].distance>nn.distance)right=mid;
    else left=mid;
  }
  //check equal ID

  while (left > 0){
    if (addr[left].distance < nn.distance) break;
    if (addr[left].id == nn.id) return K + 1;
    left--;
  }
  if(addr[left].id == nn.id||addr[right].id==nn.id)return K+1;
  memmove((char *)&addr[right+1], &addr[right],(K-right) * sizeof(Neighbor));
  addr[right]=nn;
  return right;
}

// @CS0522
// 重载，方便 nsg 和 wadg 都可用
static inline int InsertIntoPool(std::vector<Neighbor> &retset, unsigned K, Neighbor new_nn)
{
    // find the location to insert
    int left = 0, right = K - 1;
    if(retset[left].distance > new_nn.distance)
    {
        retset.insert(retset.begin() + left, new_nn);
        return left;
    }
    // 实际不会进入这个 if 分支
    if(retset[right].distance < new_nn.distance)
    {
        retset.insert(retset.begin() + K, new_nn);
        return K;
    }
    // 二分查找插入点
    while(left < right - 1)
    {
        int mid = (left + right) / 2;
        if(retset[mid].distance > new_nn.distance)
        {
            right = mid;
        }
        else
        {
            left = mid;
        }
    }
    //check equal ID

    while (left > 0)
    {
        if (retset[left].distance < new_nn.distance)
        {
            break;
        }
        if (retset[left].id == new_nn.id)
        {
            return K + 1;
        }
        left--;
    }
    if(retset[left].id == new_nn.id || retset[right].id == new_nn.id)
    {
        return K + 1;
    }
    retset.insert(retset.begin() + right, new_nn);
    return right;
}

// 修改为链表
// 返回值为插入的 index
static inline int InsertIntoPool_linklist(Neighbors *&retset, Neighbor new_nn)
{
    // 插入到带头节点有序链表中，且不会插到末尾
    Node *pointer = retset->head;
    unsigned index = 0;
    while (pointer->next != nullptr)
    {
        // 在该 index 处插入
        if (pointer->nn.distance <= new_nn.distance && pointer->next->nn.distance >= new_nn.distance)
        {
            // 存在相同 id
            if (pointer->nn.id == new_nn.id || pointer->next->nn.id == new_nn.id)
            {
                return retset->size + 1;
            }
            // 插入
            Node *new_node = new Node(new_nn);
            new_node->next = pointer->next;
            pointer->next = new_node;
            ++retset->size;
            // 超过容量要截断
            if (retset->size > retset->capacity)
            {
                retset->remove_tail();
            }
            break;
        }
        pointer = pointer->next;
        ++index;
    }
    return index;
}
}

#endif //EFANNA2E_GRAPH_H

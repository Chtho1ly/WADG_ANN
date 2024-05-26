//
// Created by 付聪 on 2017/6/21.
//

#include "efanna2e/index_wadg.h"
#include "efanna2e/util.h"

void load_data(char *filename, float *&data, unsigned &num,
               unsigned &dim)
{ // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open())
  {
    std::cout << filename << " open file error" << std::endl;
    exit(-1);
  }
  in.read((char *)&dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[num * dim * sizeof(float)];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++)
  {
    in.seekg(4, std::ios::cur);
    in.read((char *)(data + i * dim), dim * 4);
  }
  in.close();
}

void save_result(char *filename, std::vector<std::vector<unsigned>> &results)
{
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++)
  {
    unsigned GK = (unsigned)results[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

int main(int argc, char **argv)
{
  if (argc != 7)
  {
    std::cout << argv[0]
              << " data_file query_file nsg_path search_L search_K result_path"
              << std::endl;
    exit(-1);
  }
  float *data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  float *query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  assert(dim == query_dim);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  if (L < K)
  {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build query_load = efanna2e::data_align(query_load,
  // query_num, query_dim);
  // efanna2e::L2确定距离比较器，默认欧氏距离平方
  // @CS0522
  // 传入聚类中心数 K，用于初始化 IndexWADG 的成员变量
  efanna2e::IndexWADG index(dim, points_num, efanna2e::L2, K, nullptr);
  index.Load(argv[3]);
  index.Set_data(data_load);
  index.Set_lru();

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L); // 未找到使用P_search的地方
  paras.Set<unsigned>("K_search", K); // 向parameters中添加k_search

  auto s = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<unsigned>> res;

  for (unsigned i = 0; i < query_num; i++)
  {
    std::vector<unsigned> tmp(K);
    // @CS0522
    // 指向 vector 内部的指针
    unsigned *tmp_ = tmp.data();
    index.Search(query_load + i * dim, paras, tmp_, HOT_POINTS);
    res.push_back(tmp);
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "search time: " << diff.count() << "\n";

  // print infos
  if (PRINT_INFO)
  {
    printf("==========\n");
    std::cout << "更新窗口次数: " << index.get_window_count() << std::endl;
    std::cout << "更新热点次数: " << index.get_update_hot_points_count() << std::endl;

    // print hyper parameters
    auto hyperparams = index.get_hyperparams();
    std::cout << "超参数: "
              << "W = " << std::get<0>(hyperparams)
              << ", C = " << std::get<1>(hyperparams)
              << ", Q = " << L
              << ", L = " << std::get<2>(hyperparams)
              << ", K = " << K << std::endl;

    // DEBUG
    if (DEBUG)
    {
      std::cout << "主 Search 中尝试加入 retset 的点数量: " << std::endl;
      auto counts = index.get_try_enter_retset_points_counts();
      int total_counts = 0;
      for (int i = 0; i < counts.size(); i++)
      {
        total_counts += counts[i];
      }
      std::cout << "Total(for " << counts.size() << " queries): " << total_counts << std::endl;
    }
    printf("==========\n");
  }

  save_result(argv[6], res);

  return 0;
}

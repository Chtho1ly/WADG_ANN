#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>

// 读取ivecs文件
std::vector<std::vector<int>> readIvecs(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open())
    {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<int>> data;
    while (!input.eof())
    {
        int dim;
        input.read(reinterpret_cast<char *>(&dim), sizeof(int));
        std::vector<int> vec(dim);
        input.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(int));
        if (input)
        {
            data.push_back(std::move(vec));
        }
    }
    return data;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "test_cal_top_K_precision {search result path} {ground truth path} {K}" << std::endl;
        return 1;
    }

    std::string file1 = argv[1];
    std::string file2 = argv[2];
    int K = std::stoi(argv[3]);

    auto data1 = readIvecs(file1);
    auto data2 = readIvecs(file2);

    if (data1.size() != data2.size())
    {
        std::cout << "两个文件的行数不同。" << std::endl;
        return 0;
    }

    int overlap = 0;
    for (size_t i = 0; i < data1.size(); ++i)
    {
        if (data1[i].size() < K || data2[i].size() < K)
        {
            std::cout << "文件1向量数量为: " << data1[i].size() << ", 文件2向量数量为: " << data2[i].size() << std::endl;
            std::cout << "在第 " << i + 1 << " 行，文件" << (int)((data1[i].size() < K) ? 1 : 2) << "中的向量数量为" << data1[i].size() << "小于K。" << std::endl;
            return -1;
        }
        for (int j = 0; j < K; ++j)
        {
            if (std::find(data2[i].begin(), data2[i].begin() + K, data1[i][j]) != data2[i].begin() + K)
            {
                ++overlap;
            }
        }
    }

    double accuracy = static_cast<double>(overlap) / (data1.size() * K);
    std::cout << "Top-" << K << " 准确度: " << accuracy << std::endl;

    return 0;
}

/**
 * @file code_eunsoo_c.cpp
 * @author EunsooLim  (ies041196@gmail.com)
 * @brief implement K-means algorithm by c++
 * @version 0.1
 * @date 2022-02-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <array>
#include <chrono>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <random>

#define DATA_ACCOUNT 1000
#define K 2

void Estep(Eigen::MatrixXf &distances, Eigen::MatrixXf &k_centroid, Eigen::MatrixXi &datas, std::array<uint16_t, 2> &labels) {
    // cal distance
    for (int i = 0; i < DATA_ACCOUNT; i++) {
        for (int j = 0; j < K; j++) {
            double_t tmp_distance = std::sqrt(std::pow(datas(i, 0) - k_centroid(j, 0), 2) + std::pow(datas(i, 1) - k_centroid(j, 1), 2));
            distances(i, j) = tmp_distance;
        }
    }
    for (int i = 0; i < DATA_ACCOUNT; i++) {
        // passing according to label
        double_t min = distances(i, 0);
        int8_t label = 0;
        for (int j = 1; j < K; j++) {
            if (min > distances(i, j)) {
                label = j;
            }
        }
        // sum for cal mean value
        k_centroid(label, 0) += datas(i, 0);
        k_centroid(label, 1) += datas(i, 1);
        labels[label]++;
    }
}
void Mstep(std::array<uint16_t, 2> &labels, Eigen::MatrixXf &k_centroid) {
    // find optimized centroid value
    for (int i = 0; i < K; i++) {
        if (labels[i] != 0) {
            k_centroid(i, 0) = k_centroid(i, 0) / labels[i];
            k_centroid(i, 1) = k_centroid(i, 1) / labels[i];
        }
    }
}

int main(int argc, const char **argv) {
    /**** generate random initial value ****/
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(1.0, 10.0);
    double number = distribution(generator);
    Eigen::MatrixXi datas = Eigen::MatrixXi::Zero(1000, 2);
    for (int i = 0; i < DATA_ACCOUNT; i++) {
        datas(i, 0) = (int)distribution(generator);
        datas(i, 1) = (int)distribution(generator);
    }

    Eigen::MatrixXf k_centroid = Eigen::MatrixXf::Zero(2, 2);
    Eigen::MatrixXf distances = Eigen::MatrixXf::Zero(1000, 2);
    for (int i = 0; i < K; i++) {
        k_centroid(i, 0) = distribution(generator);
        k_centroid(i, 1) = distribution(generator);
    }
    Eigen::MatrixXf old_k_centroid = Eigen::MatrixXf::Zero(2, 2);
    std::array<uint16_t, 2> labels = {0, 0};

    bool sat = false;
    int16_t cnt = 0;
    // loop, until cnt is maximum 20 or when value will be saturation
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    while (!sat && cnt++ < 20) {
        // init
        labels = {0, 0};
        distances = Eigen::MatrixXf::Zero(1000, 2);

        Estep(distances, k_centroid, datas, labels);
        Mstep(labels, k_centroid);

        if (old_k_centroid == k_centroid) {
            sat = true;
        }

        old_k_centroid = k_centroid;
        std::cout << "iteration : " << cnt << std::endl;
        for (int i = 0; i < K; i++) {
            std::cout << "(" << i + 1 << "'s centroid:" << k_centroid(i, 0) << "," << k_centroid(i, 1) << ") ";
        }
        std::cout << std::endl;
    }
    std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
    std::cout << "Avg duration : " << sec.count() / cnt << " seconds" << std::endl;

    return 0;
}

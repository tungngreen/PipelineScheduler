#include <misc.h>
#include <fstream>
#include <torch/torch.h>
#include <boost/algorithm/string.hpp>
#include <random>
#include <cmath>
#include <chrono>
#include "indevicemessages.grpc.pb.h"
#include "controlmessages.grpc.pb.h"

#ifndef PIPEPLUSPLUS_MICRO_OPTIMIZATION_H
#define PIPEPLUSPLUS_MICRO_OPTIMIZATION_H

using indevicemessages::FlData;
using indevicemessages::CrlUtilityWeights;
using T = torch::Tensor;

enum threadingAction {
    NoMultiThreads = 0,
    MultiPreprocess = 1,
    MultiPostprocess = 2,
    BothMultiThreads = 3
};

const std::unordered_map<std::string, torch::Dtype> DTYPE_MAP = {
        {"float", torch::kFloat32},
        {"double", torch::kDouble},
        {"half", torch::kFloat16},
        {"int", torch::kInt32},
        {"long", torch::kInt64},
        {"short", torch::kInt16},
        {"char", torch::kInt8},
        {"byte", torch::kUInt8},
        {"bool", torch::kBool}
};

class ExperienceBuffer {
public:
    ExperienceBuffer() = default;

    ExperienceBuffer(size_t capacity)
            :  timestamps(capacity), states(capacity), log_probs(capacity), values(capacity), rewards(capacity),
               capacity(capacity), current_index(0), await_reward(false), valid_history(false), is_full(false) {}

    virtual void add(const T& state, const T& log_prob, const T& value) {
        if (is_full) {
            double distance = distance_metric(state, states[current_index]);

            if (distance > 0.5) {
                spdlog::trace("Distance: {}", distance);
                //set current index to the index of the oldest timestamp
                current_index = std::distance(timestamps.begin(),
                                              std::min_element(timestamps.begin(), timestamps.end()));
                await_reward = true;
            } else {
                return;
            }
        }
        timestamps[current_index] = std::chrono::system_clock::now();
        states[current_index] = state;

        log_probs[current_index] = log_prob;
        values[current_index] = value;
        valid_history = false;
    }

    virtual void add_reward(const double x){
        if (!is_full) {
            rewards[current_index] = x;
            current_index = (current_index + 1) % capacity;
            if (current_index == 0) is_full = true;
        }

        if (await_reward) {
            rewards[current_index] = x;
            await_reward = false;
        }
    }

    [[nodiscard]] std::vector<T> get_states() const {
        if (is_full)  return states;
        return {states.begin(), states.begin() + current_index - 1};
    }

    [[nodiscard]] std::vector<T> get_log_probs() const {
        if (is_full) return log_probs;
        return {log_probs.begin(), log_probs.begin() + current_index - 1};
    }

    [[nodiscard]] std::vector<T> get_values() const {
        if (is_full) return values;
        return {values.begin(), values.begin() + current_index  - 1};
    }

    [[nodiscard]] virtual std::vector<double> get_rewards() const {
        if (is_full) return rewards;
        return {rewards.begin(), rewards.begin() + current_index  - 1};
    }

    void clear() {
        current_index = 0;
        is_full = false;
    }

protected:
    double distance_metric(const T& state, const T& log_prob) {
        if (!valid_history) {
            historical_states = torch::stack(states);
            T mean = historical_states.mean(0);
            T centered_states = historical_states - mean;
            T covariance_matrix = (centered_states.transpose(0, 1).mm(centered_states))
                                  / (static_cast<int64_t>(states.size()) - 1);
            T epsilon = torch::eye(covariance_matrix.size(0)) * 1e-6; // Small value added to the diagonal
            covariance_inv = torch::inverse(covariance_matrix + epsilon);
        }

        T diff = historical_states - state;
        T mahalanobis_distances = torch::sqrt((diff.matmul(covariance_inv).mul(diff)).sum(1));

        T kl_divergences = torch::kl_div(log_prob, torch::stack(log_probs), torch::Reduction::None);

        return 0.5 * mahalanobis_distances.mean().item<double>() + 0.5 * kl_divergences.mean().item<double>();
    }

    std::vector<ClockType> timestamps;
    std::vector<T> states, log_probs, values;
    T historical_states, covariance_inv;
    std::vector<double> rewards;

    size_t capacity;
    size_t current_index;
    bool await_reward;
    bool valid_history;
    bool is_full;
};

#endif //PIPEPLUSPLUS_MICRO_OPTIMIZATION_H

using indevicemessages::FlData;
using T = torch::Tensor;

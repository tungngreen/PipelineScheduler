#include "misc.h"
#include <fstream>
#include <torch/torch.h>
#include <boost/algorithm/string.hpp>
#include <random>
#include <cmath>
#include <chrono>
#include "indevicemessages.grpc.pb.h"
#include "controlmessages.grpc.pb.h"

#ifndef PIPEPLUSPLUS_BATCH_LEARNING_H
#define PIPEPLUSPLUS_BATCH_LEARNING_H

using indevicemessages::FlData;
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
        :  timestamps(capacity), states(capacity), log_probs(capacity), values(capacity),
          timeouts(capacity), batchings(capacity), scalings(capacity), rewards(capacity),
          capacity(capacity), current_index(0), await_reward(false), valid_history(false), is_full(false) {}

    void add(const T& state, const T& log_prob, const T& value,
             int timeout_action, int batching_action, int scaling_action) {
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
        timeouts[current_index] = timeout_action;
        batchings[current_index] = batching_action;
        scalings[current_index] = scaling_action;
        valid_history = false;
    }

    void add_reward(const double x){
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
        return {states.begin(), states.begin() + current_index};
    }

    [[nodiscard]] std::vector<T> get_log_probs() const {
        if (is_full) return log_probs;
        return {log_probs.begin(), log_probs.begin() + current_index};
    }

    [[nodiscard]] std::vector<T> get_values() const {
        if (is_full) return values;
        return {values.begin(), values.begin() + current_index};
    }

    [[nodiscard]] std::vector<int> get_timeout() const {
        if (is_full) return timeouts;
        return {timeouts.begin(), timeouts.begin() + current_index};
    }

    [[nodiscard]] std::vector<int> get_batching() const {
        if (is_full)  return batchings;
        return {batchings.begin(), batchings.begin() + current_index};
    }

    [[nodiscard]] std::vector<int> get_scaling() const {
        if (is_full)  return scalings;
        return {scalings.begin(), scalings.begin() + current_index};
    }

    [[nodiscard]] std::vector<double> get_rewards() const {
        if (is_full) return rewards;
        return {rewards.begin(), rewards.begin() + current_index};
    }

    void clear() {
        current_index = 0;
        is_full = false;
    }

private:

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
    std::vector<int> timeouts, batchings, scalings;
    std::vector<double> rewards;

    size_t capacity;
    size_t current_index;
    bool await_reward;
    bool valid_history;
    bool is_full;
};

struct MultiPolicyNet: torch::nn::Module {
    MultiPolicyNet(int state_size, int action1_size, int action2_size, int action3_size) {
        shared_layer1 = register_module("shared_layer", torch::nn::Linear(state_size, 64));
        shared_layer2 = register_module("shared_layer2", torch::nn::Linear(64, 48));
        value_head = register_module("value_head", torch::nn::Linear(48, 1));
        norm_layer = register_module("norm_layer", torch::nn::LayerNorm(torch::nn::LayerNormOptions({48})));
        policy_head1 = register_module("policy_head1", torch::nn::Linear(48, action1_size));
        policy_head2 = register_module("policy_head2", torch::nn::Linear(48 + action1_size, action2_size));
        policy_head3 = register_module("policy_head3", torch::nn::Linear(48 + action1_size, action3_size));
    }

    std::tuple<T, T, T, T> forward(T state) {
        T x = torch::relu(shared_layer1->forward(state));
        x = torch::relu(shared_layer2->forward(x));
        x = norm_layer->forward(x);
        T policy1_output = torch::softmax(policy_head1->forward(x), -1);
        T combined_input = torch::cat({x, policy1_output.clone()}, -1);
        T policy2_output = torch::softmax(policy_head2->forward(combined_input), -1);
        T policy3_output = torch::softmax(policy_head3->forward(combined_input), -1);
        T value = value_head->forward(x);
        return std::make_tuple(policy1_output, policy2_output, policy3_output, value);
    }

    torch::nn::Linear shared_layer1{nullptr};
    torch::nn::Linear shared_layer2{nullptr};
    torch::nn::Linear value_head{nullptr};
    torch::nn::LayerNorm norm_layer{nullptr};
    torch::nn::Linear policy_head1{nullptr};
    torch::nn::Linear policy_head2{nullptr};
    torch::nn::Linear policy_head3{nullptr};
};

struct SinglePolicyNet: torch::nn::Module {
    SinglePolicyNet(int state_size, int action1_size, int action2_size, int action3_size) {
        shared_layer1 = register_module("shared_layer", torch::nn::Linear(state_size, 64));
        shared_layer2 = register_module("shared_layer2", torch::nn::Linear(64, 48));
        policy_head = register_module("policy_head", torch::nn::Linear(48, action1_size * action2_size * action3_size));
        value_head = register_module("value_head", torch::nn::Linear(48, 1));
    }

    std::tuple<T, T> forward(T state) {
        T x = torch::relu(shared_layer1->forward(state));
        x = torch::relu(shared_layer2->forward(x));
        T policy_output = torch::softmax(policy_head->forward(x), -1);
        T value = value_head->forward(x);
        return std::make_tuple(policy_output, value);
    }

    T combine_actions(std::vector<int> timeout, std::vector<int> batching, std::vector<int> scaling, int action2_size, int action3_size) {
        T action = torch::zeros({static_cast<long>(timeout.size()), 1});
        for (unsigned int i = 0; i < timeout.size(); i++) {
            action[i] = timeout[i] * action2_size * action3_size + batching[i] * action3_size + scaling[i];
        }
        return action;
    }

    std::tuple<int, int, int> interpret_action(int action, int action2_size, int action3_size) {
        int timeout = action / (action2_size * action3_size);
        int remainder = action % (action2_size * action3_size);
        int batching = remainder / action3_size;
        int scaling = remainder % action3_size;
        return std::make_tuple(timeout, batching, scaling);
    }

    torch::nn::Linear shared_layer1{nullptr};
    torch::nn::Linear shared_layer2{nullptr};
    torch::nn::Linear policy_head{nullptr};
    torch::nn::Linear value_head{nullptr};
};

class CHEISAgent {
public:
    CHEISAgent(std::string& cont_name, uint state_size, uint timeout_size, uint max_batch, uint scaling_size,
             socket_t *socket, BatchInferProfileListType profile, int base_batch, torch::Dtype precision = torch::kF64,
             uint update_steps = 60, uint update_steps_inc = 5, uint federated_steps = 5, double lambda = 0.95,
             double gamma = 0.99, double clip_epsilon = 0.2, double penalty_weight = 0.1, double theta = 1.0,
             double sigma = 10.0, double phi = 2.0, double rho = 1.0, int seed = 42);

    ~CHEISAgent() {
        torch::save(model, path + "/latest_model.pt");
        out.close();
    }

    std::tuple<int, int, int> runStep();
    void rewardCallback(double throughput, double latency_penalty, double oversize_penalty, double memory_use);
    void setState(double curr_timeout, double curr_batch, double curr_scaling,  double arrival, double pre_queue_size,
                  double inf_queue_size, double post_queue_size, double slo, double memory_use);
    void federatedUpdateCallback(const std::string &msg);

private:
    void update();
    void federatedUpdate(double loss);
    void reset() {
        cumu_reward = 0.0;
        first = true;
        experiences.clear();
    }
    void selectAction();
    T computeCumuRewards() const;
    T computeGae() const;

    std::mutex model_mutex;
    std::shared_ptr<MultiPolicyNet> model;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    torch::Dtype precision;
    T state, log_prob, value;
    double last_slo;
    int timeout, batching, last_batching, scaling;
    ExperienceBuffer experiences;

    bool first = true;
    bool penalty = true;
    BatchInferProfileListType batchInferProfileList;

    std::ofstream out;
    std::string path;
    std::string cont_name;
    socket_t *device_socket;

    // PPO Hyperparameters
    double lambda;
    double gamma;
    double clip_epsilon;
    double cumu_reward;
    double last_avg_reward = 0.0;
    double penalty_weight;

    // weights for reward function
    double theta;
    double sigma;
    double phi;
    double rho;

    uint state_size;
    uint timeout_size;
    uint max_batch;
    int base_batch;
    uint scaling_size;

    uint steps_counter = 0;
    uint update_steps;
    uint update_steps_inc;
    uint federated_steps_counter = 1;
    uint federated_steps;
    ClockType  federatedStartTime;
};

class CHEISServer {
public:
    CHEISServer(std::string run_name, nlohmann::json parameters, uint16_t clusterCount, socket_t *mq, uint state_size = 9,
               torch::Dtype precision = torch::kF32);
    ~CHEISServer() {
        int i = 0;
        for (auto model: models) {
            torch::save(model, path + "/latest_model" + std::to_string(i++) + ".pt");
        }
        out.close();
    }

    bool addClient(FlData &data);

    void incrementClientCounter() { client_counter++; }

    void decrementClientCounter() { client_counter--; }

    void updateCluster(uint16_t id, std::vector<std::tuple<std::string, std::string, nlohmann::json>> devices);

    void stop() { run = false; }

    nlohmann::json getConfig() {
        std::string p = torch::toString(precision);
        return {
                {"state_size", state_size},
                {"lambda", lambda},
                {"gamma", gamma},
                {"clip_epsilon", clip_epsilon},
                {"penalty_weight", penalty_weight},
                {"theta", theta},
                {"sigma", sigma},
                {"phi", phi},
                {"rho", rho},
                {"precision", boost::algorithm::to_lower_copy(p)},
                {"update_steps", update_steps},
                {"update_step_incs", update_step_incs},
                {"federated_steps", federated_steps},
                {"seed", seed}
        };
    }

private:
    struct ClientModel{
        FlData data;
        std::shared_ptr<MultiPolicyNet> model;
    };

    void proceed();

    void returnFLModel(ClientModel &client);

    void sendClusterModel(std::string name, std::string queue, nlohmann::json conf);

    std::vector<std::shared_ptr<MultiPolicyNet>> models;
    std::vector<std::map<std::vector<int64_t>, T>> action_heads_by_size;
    uint16_t clusterID;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    torch::Dtype precision;
    std::vector<ClientModel> federated_clients;
    uint client_counter = 0;

    std::mt19937 re;
    std::ofstream out;
    std::string path;
    std::atomic<bool> run;
    socket_t *message_queue;

    // PPO Hyperparameters
    double lambda;
    double gamma;
    double clip_epsilon;
    double penalty_weight;
    uint state_size;

    // weights for reward function
    double theta;
    double sigma;
    double phi;
    double rho;


    // Parameters for Learning Loop
    int update_steps;
    int update_step_incs;
    int federated_steps;
    int seed;
};


#endif //PIPEPLUSPLUS_BATCH_LEARNING_H

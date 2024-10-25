#ifndef PIPEPLUSPLUS_BATCH_LEARNING_H
#define PIPEPLUSPLUS_BATCH_LEARNING_H

#include "misc.h"
#include <fstream>
#include <torch/torch.h>
#include <random>
#include <cmath>
#include <chrono>
#include "indevicecommands.grpc.pb.h"
#include "indevicemessages.grpc.pb.h"
#include "controlcommands.grpc.pb.h"

using controlcommands::ControlCommands;
using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using indevicemessages::InDeviceMessages;
using indevicecommands::FlData;
using EmptyMessage = google::protobuf::Empty;
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

template <typename Experience>
class ExperienceBuffer {
public:

    ExperienceBuffer(size_t capacity)
        : buffer(capacity), capacity(capacity), current_index(0), is_full(false) {}

    size_t add(const Experience& experience, float (*distance_metric)(const Experience&, const Experience&)) {
        if (!is_full) {
            size_t inserted_index = current_index;
            add(experience);
            return inserted_index;
        }

        auto min_it = std::min_element(buffer.begin(), buffer.end(),
                                       [&experience, &distance_metric](const Experience& a, const Experience& b) {
                                           return distance_metric(experience, a) < distance_metric(experience, b);
                                       });

        size_t removed_index = std::distance(buffer.begin(), min_it);

        if (distance_metric(experience, *min_it) > 0.1) {
            *min_it = experience;
        }

        return removed_index;
    }

    void add_at(const Experience& experience, size_t index) {
        if (index >= capacity) {
            throw std::out_of_range("Index out of range.");
        }
        buffer[index] = experience;
    }

    // Retrieve all experiences (for debugging or analysis)
    std::vector<Experience> get_all() const {
        if (is_full) {
            std::vector<Experience> ordered(buffer.begin() + current_index, buffer.end());
            ordered.insert(ordered.end(), buffer.begin(), buffer.begin() + current_index);
            return ordered;
        }
        return std::vector<Experience>(buffer.begin(), buffer.begin() + current_index);
    }

private:

    void add(const Experience& experience) {
        buffer[current_index] = experience;
        current_index = (current_index + 1) % capacity;
        if (current_index == 0) is_full = true;
    }

    std::vector<Experience> buffer;
    size_t capacity;
    size_t current_index;
    bool is_full;
};

struct ActorCriticNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, policy_head{nullptr}, value_head{nullptr};

    ActorCriticNet(int state_size, int action_size) {
        fc1 = register_module("fc1", torch::nn::Linear(state_size, 64));
        policy_head = register_module("policy_head", torch::nn::Linear(64, action_size));
        value_head = register_module("value_head", torch::nn::Linear(64, 1));
    }

    std::pair<T, T> forward(T state) {
        T x = torch::relu(fc1->forward(state));
        T policy_logits = policy_head->forward(x);
        T policy = torch::softmax(policy_logits, -1); // Softmax to obtain action probabilities
        T value = value_head->forward(x); // State value
        return {policy, value};
    }
};

struct MultiPolicyNetwork: torch::nn::Module {
    torch::nn::Linear shared_layer1{nullptr};
    torch::nn::Linear shared_layer2{nullptr};
    torch::nn::Linear policy_head1{nullptr};
    torch::nn::Linear policy_head2{nullptr};
    torch::nn::Linear policy_head3{nullptr};
    torch::nn::Linear value_head{nullptr};

    MultiPolicyNetwork(int state_size, int action1_size, int action2_size, int action3_size) {
        shared_layer1 = register_module("shared_layer", torch::nn::Linear(state_size, 64));
        shared_layer2 = register_module("shared_layer2", torch::nn::Linear(64, 48));
        policy_head1 = register_module("policy_head1", torch::nn::Linear(48, action1_size));
        policy_head2 = register_module("policy_head2", torch::nn::Linear(48 + action1_size, action2_size));
        policy_head3 = register_module("policy_head3", torch::nn::Linear(48 + action1_size, action3_size));
        value_head = register_module("value_head", torch::nn::Linear(48, 1));
    }

    std::tuple<T, T, T, T> forward(T state) {
        T x = torch::nn::functional::relu(shared_layer1(state));
        T y = torch::nn::functional::relu(shared_layer2(x));
        T policy1_output = torch::nn::functional::softmax(policy_head1(y), -1);
        T combined_input1 = torch::cat({y, policy1_output.clone()}, -1);
        T policy2_output = torch::nn::functional::softmax(policy_head2(combined_input1), -1);
        T combined_input2 = torch::cat({y, policy1_output.clone()}, -1);
        T policy3_output = torch::nn::functional::softmax(policy_head3(combined_input2), -1);
        T value = value_head(y);
        return std::make_tuple(policy1_output, policy2_output, policy3_output, value);
    }
};

class FCPOAgent {
public:
    FCPOAgent(std::string& cont_name, uint state_size, uint resolution_size, uint max_batch, uint threading_size,
             CompletionQueue *cq, std::shared_ptr<InDeviceMessages::Stub> stub, torch::Dtype precision,
             uint update_steps = 60, uint update_steps_inc = 5, uint federated_steps = 5, double lambda = 0.95,
             double gamma = 0.99, double clip_epsilon = 0.2, double penalty_weight = 0.1);

    ~FCPOAgent() {
        torch::save(model, path + "/latest_model.pt");
        out.close();
    }
    std::tuple<int, int, int> runStep();
    void rewardCallback(double throughput, double drops, double latency_penalty, double oversize_penalty);
    void setState(double curr_batch, double curr_resolution_choice,  double arrival, double pre_queue_size,
                  double inf_queue_size);
    void federatedUpdateCallback(FlData &response);

private:
    void update();
    void federatedUpdate();
    void reset() {
        cumu_reward = 0.0;
        states.clear();
        values.clear();
        resolution_actions.clear();
        batching_actions.clear();
        scaling_actions.clear();
        rewards.clear();
        log_probs.clear();
    }
    std::tuple<int, int, int> selectAction();
    T computeCumuRewards() const;
    T computeGae() const;

    std::mutex model_mutex;
    std::shared_ptr<MultiPolicyNetwork> model;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    torch::Dtype precision;
    T state, log_prob, value;
    std::vector<T> states, log_probs, values;
    std::tuple<int, int, int> actions;
    std::vector<int> resolution_actions;
    std::vector<int> batching_actions;
    std::vector<int> scaling_actions;
    std::vector<double> rewards;

    bool first = true;
    std::mt19937 re;
    std::ofstream out;
    std::string path;
    std::string cont_name;
    CompletionQueue *cq;
    std::shared_ptr<InDeviceMessages::Stub> stub;

    double lambda;
    double gamma;
    double clip_epsilon;
    double cumu_reward;
    double penalty_weight;
    uint state_size;
    uint resolution_size;
    uint max_batch;
    uint threading_size;

    uint steps_counter = 0;
    uint update_steps;
    uint update_steps_inc;
    uint federated_steps_counter = 1;
    uint federated_steps;
};

class FCPOServer {
public:
    FCPOServer(std::string run_name, uint state_size, torch::Dtype precision);
    ~FCPOServer() {
        torch::save(model, path + "/latest_model.pt");
        out.close();
    }

    void addClient(FlData &data, std::shared_ptr<ControlCommands::Stub> stub, CompletionQueue *cq);

    void stop() { run = false; }

    nlohmann::json getConfig() {
        return {
                {"state_size", state_size},
                {"lambda", lambda},
                {"gamma", gamma},
                {"clip_epsilon", clip_epsilon},
                {"penalty_weight", penalty_weight},
                {"precision", torch::toString(precision)},
                {"update_steps", 60},
                {"update_step_incs", 5},
                {"federated_steps", 5}
        };
    }

private:
    struct ClientModel{
        FlData data;
        std::shared_ptr<MultiPolicyNetwork> model;
        std::shared_ptr<ControlCommands::Stub> stub;
        CompletionQueue *cq;
    };

    void proceed();

    void returnFLModel(ClientModel &client);

    std::shared_ptr<MultiPolicyNetwork> model;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    torch::Dtype precision;
    std::vector<ClientModel> federated_clients;
    uint client_counter = 0;

    std::mt19937 re;
    std::ofstream out;
    std::string path;
    std::atomic<bool> run;

    double lambda;
    double gamma;
    double clip_epsilon;
    double penalty_weight;
    uint state_size;
};


#endif //PIPEPLUSPLUS_BATCH_LEARNING_H

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

struct ActorCriticNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, policy_head{nullptr}, value_head{nullptr};

    ActorCriticNet(int state_size, int action_size) {
        fc1 = register_module("fc1", torch::nn::Linear(state_size, 64));
        policy_head = register_module("policy_head", torch::nn::Linear(64, action_size));
        value_head = register_module("value_head", torch::nn::Linear(64, 1));
    }

    std::pair<T, T> forward(T state) {
        auto x = torch::relu(fc1->forward(state));
        auto policy_logits = policy_head->forward(x);
        auto policy = torch::softmax(policy_logits, -1); // Softmax to obtain action probabilities
        auto value = value_head->forward(x); // State value
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
        shared_layer2 = register_module("shared_layer2", torch::nn::Linear(64, 64));
        policy_head1 = register_module("policy_head1", torch::nn::Linear(64, action1_size));
        policy_head2 = register_module("policy_head2", torch::nn::Linear(64 + action1_size, action2_size));
        policy_head3 = register_module("policy_head3", torch::nn::Linear(action2_size, action3_size));
        value_head = register_module("value_head", torch::nn::Linear(64, 1));
    }

    std::tuple<T, T, T, T> forward(T x) {
        auto shared_features = torch::relu(shared_layer1(x));
        shared_features = torch::relu(shared_layer2(shared_features));
        auto policy1_output = torch::nn::functional::softmax(policy_head1(shared_features), -1);
        auto combined_input = torch::cat({shared_features, policy1_output}, -1);
        auto policy2_output = torch::nn::functional::glu(combined_input,-1);
        auto policy3_output = torch::nn::functional::glu(policy2_output,-1);
        auto value_output = value_head(shared_features);
        return std::make_tuple(policy1_output, policy2_output, policy3_output, value_output);
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
    T computeCumuRewards(double last_value = 0.0) const;
    T computeGae(double last_value = 0.0) const;

    std::mutex model_mutex;
    std::shared_ptr<MultiPolicyNetwork> model;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    torch::Dtype precision;
    T state;
    std::tuple<int, int, int> actions;
    std::vector<T> states, log_probs, values;
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

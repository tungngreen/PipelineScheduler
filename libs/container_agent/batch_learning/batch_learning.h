#include "misc.h"
#include <fstream>
#include <torch/torch.h>
#include <random>
#include <cmath>
#include <chrono>

#ifndef PIPEPLUSPLUS_BATCH_LEARNING_H
#define PIPEPLUSPLUS_BATCH_LEARNING_H

// Network model for Proximal Policy Optimization
struct ActorCriticNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, policy_head{nullptr}, value_head{nullptr};

    ActorCriticNet(int state_size, int action_size) {
        fc1 = register_module("fc1", torch::nn::Linear(state_size, 64));
        policy_head = register_module("policy_head", torch::nn::Linear(64, action_size));
        value_head = register_module("value_head", torch::nn::Linear(64, 1));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state) {
        auto x = torch::relu(fc1->forward(state));
        auto policy_logits = policy_head->forward(x);
        auto policy = torch::softmax(policy_logits, -1); // Softmax to obtain action probabilities
        auto value = value_head->forward(x);             // State value
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

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
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

// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPOAgent {
public:
    PPOAgent(std::string& cont_name, uint state_size, uint max_batch, uint resolution_size,
             uint update_steps = 256, double lambda = 0.95, double gamma = 0.99, const std::string& model_save = "");

    ~PPOAgent() {
        torch::save(model, path + "/latest_model.pt");
        out.close();
    }
    std::tuple<int, int, int> runStep();
    void rewardCallback(double throughput, double drops, double latency_penalty, double oversize_penalty);
    void setState(double curr_batch, double curr_resolution_choice,  double arrival, double pre_queue_size,
                  double inf_queue_size);

private:
    void update();
    std::tuple<int, int, int> selectAction(torch::Tensor state);
    torch::Tensor compute_cumu_rewards(double last_value = 0.0);
    torch::Tensor compute_gae(double last_value = 0.0);

    std::shared_ptr<MultiPolicyNetwork> model;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    std::vector<torch::Tensor> states, log_probs, values;
    std::vector<int> resolution_actions;
    std::vector<int> batching_actions;
    std::vector<int> scaling_actions;
    std::vector<double> rewards;

    std::mt19937 re;
    std::ofstream out;
    std::string path;
    uint max_batch;

    uint counter = 0;
    uint update_steps;
    double lambda;
    double gamma;
    double clip_epsilon;
    double avg_reward;
    double penalty_weight;
};


#endif //PIPEPLUSPLUS_BATCH_LEARNING_H

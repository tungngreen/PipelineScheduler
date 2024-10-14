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

    // Layers
    torch::nn::Linear fc1{nullptr}, policy_head{nullptr}, value_head{nullptr};
};

// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPOAgent {
public:
    PPOAgent(std::string& cont_name, uint max_batch, uint update_steps = 256, double lambda = 0.95, double gamma = 0.99,
             const std::string& model_save = "");

    ~PPOAgent() {
        torch::save(actor_critic_net, path + "/latest_model.pt");
        out.close();
    }
    int runStep();
    void rewardCallback(double throughput, double drops, double latency_penalty, double oversize_penalty);
    void setState(double curr_batch, double arrival, double pre_queue_size, double inf_queue_size);

private:
    void update();
    int selectAction(torch::Tensor state);
    torch::Tensor compute_cumu_rewards(double last_value = 0.0);
    torch::Tensor compute_gae(double last_value = 0.0);

    std::shared_ptr<ActorCriticNet> actor_critic_net;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    std::vector<torch::Tensor> states, log_probs, values;
    std::vector<int> actions;
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
};


#endif //PIPEPLUSPLUS_BATCH_LEARNING_H

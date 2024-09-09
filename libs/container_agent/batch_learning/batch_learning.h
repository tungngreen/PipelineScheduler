#include "misc.h"
#include <fstream>
#include <torch/torch.h>
#include <random>
#include <math.h>

#ifndef PIPEPLUSPLUS_BATCH_LEARNING_H
#define PIPEPLUSPLUS_BATCH_LEARNING_H

// Vector of tensors.
using VT = std::vector<torch::Tensor>;

// Network model for Proximal Policy Optimization
struct ActorCriticImpl : public torch::nn::Module{
    // Actor.
    torch::nn::Linear a_lin1_, a_lin2_, a_lin3_;
    torch::Tensor mu_, log_std_;
    // Critic.
    torch::nn::Linear c_lin1_, c_lin2_, c_lin3_, c_val_;

    ActorCriticImpl(int64_t n_in, int64_t n_out, double std = 2e-2) :
            a_lin1_(torch::nn::Linear(n_in, 16)), a_lin2_(torch::nn::Linear(16, 32)),
            a_lin3_(torch::nn::Linear(32, n_out)),
            mu_(torch::full(n_out, 0.)), log_std_(torch::full(n_out, std)),
            c_lin1_(torch::nn::Linear(n_in, 16)), c_lin2_(torch::nn::Linear(16, 32)),
            c_lin3_(torch::nn::Linear(32, n_out)),
            c_val_(torch::nn::Linear(n_out, 1)) {
        // Register the modules.
        register_module("a_lin1", a_lin1_);
        register_module("a_lin2", a_lin2_);
        register_module("a_lin3", a_lin3_);
        register_parameter("log_std", log_std_);
        register_module("c_lin1", c_lin1_);
        register_module("c_lin2", c_lin2_);
        register_module("c_lin3", c_lin3_);
        register_module("c_val", c_val_);
    }

    // Forward pass.
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // Actor.
        mu_ = torch::relu(a_lin1_->forward(x));
        mu_ = torch::relu(a_lin2_->forward(mu_));
        mu_ = torch::tanh(a_lin3_->forward(mu_));
        // Critic.
        torch::Tensor val = torch::relu(c_lin1_->forward(x));
        val = torch::relu(c_lin2_->forward(val));
        val = torch::tanh(c_lin3_->forward(val));
        val = c_val_->forward(val);

        torch::NoGradGuard no_grad;
        torch::Tensor action = at::normal(mu_, log_std_.exp().expand_as(mu_));
        return std::make_tuple(action, val);
    }

    void normal(double mu, double std) {
        torch::NoGradGuard no_grad;
        for (auto& p: this->parameters()) {
            p.normal_(mu,std);
        }
    }

    torch::Tensor entropy() { // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
        return 0.5 + 0.5*log(2*M_PI) + log_std_;
    }

    torch::Tensor log_prob(torch::Tensor action) { // Logarithmic probability of taken action, given the current distribution.
        torch::Tensor var = (log_std_+log_std_).exp();
        return -((action - mu_)*(action - mu_))/(2*var) - log_std_ - log(sqrt(2*M_PI));
    }
};
TORCH_MODULE(ActorCritic);


// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPO {
public:
    PPO(ActorCritic& ac, uint update_steps = 2048, uint mini_batch_size = 512, uint epochs = 4,
        double lambda = 0.95, double gamma = 0.99);

    ~PPO() {
        torch::save(ac, "../models/batch_learning/latest_model.pt");
        out.close();
    }
    double runStep(torch::Tensor& state);
    void rewardCallback(double throughput, double drops, double oversize_penalty);
    torch::Tensor buildState(int curr_batch, int arrival, int target_latency, int pre_queue_size, int inf_queue_size, int mem);

    static ActorCritic initActorCritic(int64_t n_in, int64_t n_out, double std, std::string model_save = "") {
        ActorCritic ac = ActorCritic(n_in, n_out, std);
        ac->to(torch::kF64);
        ac->normal(0., std);
        if (model_save != "") torch::load(ac, model_save);
        return ac;
    }

private:
    VT returns(); // Generalized advantage estimate, https://arxiv.org/abs/1506.02438
    void update(double beta = 1e-3, double clip_param = 0.2);

    ActorCritic ac;
    torch::optim::Adam* opt;
    VT states, actions, rewards, dones, log_probs, return_vals, values;

    std::mt19937 re;
    std::ofstream out;

    uint counter = 0;
    uint update_steps;
    uint mini_batch_size;
    uint epochs;
    double lambda;
    double gamma;
    double avg_reward;
};


#endif //PIPEPLUSPLUS_BATCH_LEARNING_H

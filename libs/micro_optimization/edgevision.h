#include "micro_optimization.h"

#ifndef PIPEPLUSPLUS_EDGEVISION_H
#define PIPEPLUSPLUS_EDGEVISION_H

// * Please note that this code is a simplified version of the original EdgeVision only focused on offloading.
// * The original code contains many other functionalities that are not included here.
// * For instance choosing Resolution and Model Type are not included as not all low-end edge devices support loading multiple models.
struct EdgeViActorNet: torch::nn::Module {
    EdgeViActorNet(int state_size, int action_size) {
        mlp_layer1 = register_module("mlp_layer1", torch::nn::Linear(state_size, 128));
        layer_norm1 = register_module("layer_norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
        mlp_layer2 = register_module("mlp_layer2", torch::nn::Linear(128, 128));
        layer_norm2 = register_module("layer_norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
        action_head = register_module("action_head", torch::nn::Linear(128, action_size));
    }

    T forward(T state) {
        T x = layer_norm1->forward(torch::relu(mlp_layer1->forward(state)));
        x = layer_norm2->forward(torch::relu(mlp_layer2->forward(x)));
        T actions = action_head->forward(x);
        return actions;
    }

    torch::nn::Linear mlp_layer1{nullptr};
    torch::nn::LayerNorm layer_norm1{nullptr};
    torch::nn::Linear mlp_layer2{nullptr};
    torch::nn::LayerNorm layer_norm2{nullptr};
    torch::nn::Linear action_head{nullptr};
};

struct EdgeViCriticNet: torch::nn::Module {
    EdgeViCriticNet(int state_size) {
        embedding_layer = register_module("embedding_layer", torch::nn::Linear(state_size, 8));
        multihead_attention = register_module("multihead_attention", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(8, 8)));
        // The MLP input size depends on seq_len, so we delay initialization
    }

    void initialize_critic(int seq_len) {
        int fc_input_dim = seq_len * 8;

        // Only initialize once or if sequence length has changed
        if (!mlp_layer1 || mlp_layer1->weight.sizes()[1] != fc_input_dim) {
            mlp_layer1 = register_module("mlp_layer1", torch::nn::Linear(fc_input_dim, 128));
            layer_norm1 = register_module("layer_norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
            mlp_layer2 = register_module("mlp_layer2", torch::nn::Linear(128, 128));
            layer_norm2 = register_module("layer_norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
            critic_head = register_module("critic_head", torch::nn::Linear(128, 1));
        }
    }

    T forward(T states) {
        auto batch_size = states.size(0);
        auto seq_len = states.size(1);
        auto x_flat = states.view({-1, states.size(2)});
        auto embedded = embedding_layer->forward(x_flat);

        embedded = embedded.view({batch_size, seq_len, 8}).transpose(0, 1);
        auto [attn_output, _] = multihead_attention->forward(embedded, embedded, embedded);
        auto attn_flat = attn_output.transpose(0, 1).reshape({batch_size, -1});

        initialize_critic(seq_len);
        auto x = torch::relu(layer_norm1->forward(mlp_layer1->forward(attn_flat)));
        x = torch::relu(layer_norm2->forward(mlp_layer2->forward(x)));
        return critic_head->forward(x); // [batch_size, 1]
    }

    torch::nn::Linear embedding_layer{nullptr};
    torch::nn::MultiheadAttention multihead_attention{nullptr};
    torch::nn::Linear mlp_layer1{nullptr};
    torch::nn::LayerNorm layer_norm1{nullptr};
    torch::nn::Linear mlp_layer2{nullptr};
    torch::nn::LayerNorm layer_norm2{nullptr};
    torch::nn::Linear critic_head{nullptr};
};

class EdgeVisionAgent {
public:
    EdgeVisionAgent(std::string& dev_name, int offloading_candidates, torch::Dtype precision = torch::kF32, uint update_steps = 100,
                double lambda = 0.1, double gamma = 0.1, double clip = 0.2, double omega = 0.1, double sigma = 0.01, int F = 1);

    ~EdgeVisionAgent(){
        torch::save(actor, path + "/latest_actor.pt");
        torch::save(critic, path + "/latest_critic.pt");
    }

    int runStep();
    void rewardCallback(int throughput, int drops, double avg_latency, double accuracy = 0.763);
    void setState(double req_arrival_rate, int queueLength, int offloadingQueue, std::vector<double> bandwidths);
private:
    void update();
    void reset() {
        states.clear();
        targeted_devices.clear();
        rewards.clear();
        log_probs.clear();
    }
    void selectAction();
    T computeCumuRewards() const;
    T computeGae() const;

    std::mutex model_mutex;
    std::shared_ptr<EdgeViActorNet> actor;
    std::unique_ptr<torch::optim::Optimizer> actor_optimizer;
    std::shared_ptr<EdgeViCriticNet> critic;
    std::unique_ptr<torch::optim::Optimizer> critic_optimizer;
    torch::Dtype precision;
    T state, log_prob, values;
    std::vector<T> states, log_probs;
    int target_device;
    std::vector<int> targeted_devices;
    std::vector<double> rewards;

    std::string path;
    std::string dev_name;

    double lambda;
    double gamma;
    double clip;
    double omega;
    double sigma;
    int F;

    uint steps_counter = 0;
    uint update_steps;
};

struct EdgeVisionDwnstrmInfo {
    std::string name;
    std::string offloading_ip;
    int offloading_port;
    int bandwidth_id;
};

#endif //PIPEPLUSPLUS_EDGEVISION_H

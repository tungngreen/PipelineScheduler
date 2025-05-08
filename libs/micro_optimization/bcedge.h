#include "micro_optimization.h"

#ifndef PIPEPLUSPLUS_BCEDGE_H
#define PIPEPLUSPLUS_BCEDGE_H

struct BCEdgeNet: torch::nn::Module {
    BCEdgeNet(int state_size, int action1_size, int action2_size, int action3_size) {
        shared_layer1 = register_module("shared_layer", torch::nn::Linear(state_size, 256));
        shared_layer2 = register_module("shared_layer2", torch::nn::Linear(256, 128));
        policy_layer1 = register_module("policy_layer1", torch::nn::Linear(128, 64));
        policy_head1 = register_module("policy_head1", torch::nn::Linear(64, action1_size));
        policy_layer2 = register_module("policy_layer2", torch::nn::Linear(128, 64));
        policy_head2 = register_module("policy_head2", torch::nn::Linear(64, action2_size));
        policy_layer3 = register_module("policy_layer3", torch::nn::Linear(128, 64));
        policy_head3 = register_module("policy_head3", torch::nn::Linear(64, action3_size));
        value_layer = register_module("value_layer", torch::nn::Linear(128, 64));
        value_head = register_module("value_head", torch::nn::Linear(64, 1));
    }

    std::tuple<T, T, T, T> forward(T state) {
        T x = torch::relu(shared_layer1->forward(state));
        x = torch::relu(shared_layer2->forward(x));
        T policy1_output = torch::softmax(policy_head1->forward(torch::relu(policy_layer1->forward(x))), -1);
        T policy2_output = torch::softmax(policy_head2->forward(torch::relu(policy_layer2->forward(x))), -1);
        T policy3_output = torch::softmax(policy_head3->forward(torch::relu(policy_layer3->forward(x))), -1);
        T value = value_head->forward(torch::relu(value_layer->forward(x)));
        return std::make_tuple(policy1_output, policy2_output, policy3_output, value);
    }

    torch::nn::Linear shared_layer1{nullptr};
    torch::nn::Linear shared_layer2{nullptr};
    torch::nn::Linear policy_layer1{nullptr};
    torch::nn::Linear policy_head1{nullptr};
    torch::nn::Linear policy_layer2{nullptr};
    torch::nn::Linear policy_head2{nullptr};
    torch::nn::Linear policy_layer3{nullptr};
    torch::nn::Linear policy_head3{nullptr};
    torch::nn::Linear value_layer{nullptr};
    torch::nn::Linear value_head{nullptr};
};

class BCEdgeAgent {
public:
    BCEdgeAgent(std::string& dev_name, double max_memory, torch::Dtype precision = torch::kF32, uint update_steps = 200,
                double lambda = 0.1, double gamma = 0.1, double clip_epsilon = 0.9);

    ~BCEdgeAgent(){
        torch::save(model, path + "/latest_model.pt");
        out.close();
    }

    std::tuple<int, int, int> runStep();
    void rewardCallback(double throughput, double latency, MsvcSLOType slo, double memory_usage);
    void setState(ModelType model_type, std::vector<int> data_shape, MsvcSLOType slo);
private:
    void update();
    void reset() {
        cumu_reward = 0.0;
        states.clear();
        values.clear();
        batching_actions.clear();
        scaling_actions.clear();
        memory_actions.clear();
        rewards.clear();
        log_probs.clear();
    }
    void selectAction();
    T computeCumuRewards() const;
    T computeGae() const;

    std::mutex model_mutex;
    std::shared_ptr<BCEdgeNet> model;
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    torch::Dtype precision;
    T state, log_prob, value;
    std::vector<T> states, log_probs, values;
    int batching, scaling, memory;
    std::vector<int> batching_actions;
    std::vector<int> scaling_actions;
    std::vector<int> memory_actions;
    std::vector<double> rewards;

    std::ofstream out;
    std::string path;
    std::string dev_name;
    double max_memory;

    double lambda;
    double gamma;
    double clip_epsilon;
    double cumu_reward;

    uint steps_counter = 0;
    uint update_steps;
};

#endif //PIPEPLUSPLUS_BCEDGE_H

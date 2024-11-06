#include "bcedge.h"

BCEdgeAgent::BCEdgeAgent(std::string& dev_name, uint state_size, uint max_batch, uint scaling_size, uint memory_size,
                         CompletionQueue *cq, std::shared_ptr<InDeviceMessages::Stub> stub, torch::Dtype precision,
                         uint update_steps, double lambda, double gamma, double clip_epsilon)
                         : precision(precision), dev_name(dev_name), lambda(lambda), gamma(gamma),
                           clip_epsilon(clip_epsilon), state_size(state_size), max_batch(max_batch),
                           scaling_size(scaling_size), memory_size(memory_size), update_steps(update_steps) {
    path = "../models/bcedge/" + dev_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log.csv");

    model = std::make_shared<BCEdgeNet>(state_size, max_batch, scaling_size, memory_size);
    std::string model_save = path + "/latest_model.pt";
    if (std::filesystem::exists(model_save)) torch::load(model, model_save);
    model->to(precision);
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-3));

    cumu_reward = 0.0;
    states = {};
    batching_actions = {};
    scaling_actions = {};
    memory_actions = {};
    rewards = {};
    log_probs = {};
    values = {};
}

void BCEdgeAgent::selectAction() {
    std::unique_lock<std::mutex> lock(model_mutex);
    auto [policy1, policy2, policy3, val] = model->forward(state);

    T action_dist = torch::multinomial(policy1, 1);  // Sample from policy (discrete distribution)
    batching = action_dist.item<int>();  // Convert tensor to int action
    action_dist = torch::multinomial(policy2, 1);
    scaling = action_dist.item<int>();
    action_dist = torch::multinomial(policy3, 1);
    memory = action_dist.item<int>();


    log_prob = torch::log(policy1[batching]) + torch::log(policy2[scaling]) + torch::log(policy3[memory]);
    states.push_back(state);
    log_probs.push_back(log_prob);
    values.push_back(val);
    batching_actions.push_back(batching);
    scaling_actions.push_back(scaling);
    memory_actions.push_back(memory);
}

T BCEdgeAgent::computeCumuRewards() const {
    std::vector<double> discounted_rewards;
    double cumulative = 0.0;
    for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
        cumulative = *it + gamma * cumulative;
        discounted_rewards.insert(discounted_rewards.begin(), cumulative);
    }
    return torch::tensor(discounted_rewards).to(precision);
}

T BCEdgeAgent::computeGae() const {
    std::vector<double> advantages;
    double gae = 0.0;
    double next_value = 0.0;
    for (int t = rewards.size() - 1; t >= 0; --t) {
        double delta = rewards[t] + gamma * next_value - values[t].item<double>();
        gae = delta + gamma * lambda * gae;
        advantages.insert(advantages.begin(), gae);
        next_value = values[t].item<double>();
    }
    return torch::tensor(advantages).to(precision);
}
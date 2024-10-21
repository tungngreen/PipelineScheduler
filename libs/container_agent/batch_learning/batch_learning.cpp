#include "batch_learning.h"

PPOAgent::PPOAgent(std::string& cont_name, uint state_size, uint max_batch, uint resolution_size,
                   uint update_steps, double lambda, double gamma, const std::string& model_save)
                   : max_batch(max_batch), update_steps(update_steps), lambda(lambda), gamma(gamma) {
    path = "../models/batch_learning/" + cont_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log.csv");

    std::random_device rd;
    re = std::mt19937(rd());
    model = std::make_shared<MultiPolicyNetwork>(state_size, resolution_size, max_batch, 2);
    model->to(torch::kF64);
    if (!model_save.empty()) torch::load(model, model_save);

    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-3));

    avg_reward = 0.0;
    penalty_weight = 0.1;
    states = {};
    resolution_actions = {};
    batching_actions = {};
    scaling_actions = {};
    rewards = {};
    log_probs = {};
    values = {};
}

void PPOAgent::update() {
    spdlog::info("Updating reinforcement learning batch size model!");
    Stopwatch sw;
    sw.start();

    auto [policy1, policy2, policy3, v] = model->forward(torch::stack(states));
    auto action1_probs = torch::softmax(policy1, -1);
    auto action1_log_probs = torch::log(action1_probs.gather(-1, torch::tensor(resolution_actions).view({-1, 1, 1})).squeeze(-1));
    auto action2_probs = torch::softmax(policy2, -1);
    auto action2_log_probs = torch::log(action2_probs.gather(-1, torch::tensor(batching_actions).view({-1, 1, 1})).squeeze(-1));
    auto action3_probs = torch::softmax(policy3, -1);
    auto action3_log_probs = torch::log(action3_probs.gather(-1, torch::tensor(batching_actions).view({-1, 1, 1})).squeeze(-1));
    auto new_log_probs = action1_log_probs + action2_log_probs + action3_log_probs;

    auto ratio = torch::exp(new_log_probs - torch::stack(log_probs));
    auto clipped_ratio = torch::clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon);
    torch::Tensor advantages = compute_gae();

    auto policy_loss = -torch::min(ratio * advantages, clipped_ratio * advantages).to(torch::kF64).mean();
    auto value_loss = torch::mse_loss(v, compute_cumu_rewards()).to(torch::kF64);
    auto policy1_penalty = penalty_weight * torch::mean(torch::clamp(torch::tensor(resolution_actions), 0, 1)).to(torch::kF64);
    auto policy3_penalty = penalty_weight * torch::mean(torch::clamp(torch::tensor(scaling_actions), 0, 1)).to(torch::kF64);
    auto loss = policy_loss + 0.5 * value_loss + policy1_penalty + policy3_penalty;

    // Backpropagation
    optimizer->zero_grad();
    loss.to(torch::kF64).backward();
    optimizer->step();
    sw.stop();

    std::cout << "Training: " << sw.elapsed_microseconds() << "," << counter << "," << avg_reward << std::endl;

    counter = 0;
    avg_reward = 0.0;
    states.clear();
    resolution_actions.clear();
    batching_actions.clear();
    scaling_actions.clear();
    rewards.clear();
    log_probs.clear();
    values.clear();
}

void PPOAgent::rewardCallback(double throughput, double drops, double latency_penalty, double oversize_penalty) {
    rewards.push_back(2 * throughput - drops - latency_penalty + (1 - oversize_penalty));
}

void PPOAgent::setState(double curr_batch, double curr_resolution_choice, double arrival, double pre_queue_size, double inf_queue_size) {
    states.push_back(torch::tensor({{curr_batch / max_batch, curr_resolution_choice, arrival, pre_queue_size,
                                     inf_queue_size}}, torch::kF64));
}

std::tuple<int, int, int> PPOAgent::selectAction(torch::Tensor state) {
    auto [policy1, policy2, policy3, value] = model->forward(state);

    torch::Tensor action_dist = torch::multinomial(policy1, 1);  // Sample from policy (discrete distribution)
    int resolution = action_dist.item<int>();  // Convert tensor to int action
    action_dist = torch::multinomial(policy2, 1);  // Sample from policy (discrete distribution)
    int batching = action_dist.item<int>();  // Convert tensor to int action
    action_dist = torch::multinomial(policy3, 1);  // Sample from policy (discrete distribution)
    int scaling = action_dist.item<int>();  // Convert tensor to int action

    log_probs.push_back(torch::log(policy1.squeeze(0)[resolution] + torch::log(policy2.squeeze(0)[batching]) + torch::log(policy3.squeeze(0)[scaling])));
    values.push_back(value);
    return std::make_tuple(resolution, batching, scaling);
}

torch::Tensor PPOAgent::compute_cumu_rewards(double last_value) {
    std::vector<double> discounted_rewards;
    double cumulative = last_value;
    for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
        cumulative = *it + gamma * cumulative;
        discounted_rewards.insert(discounted_rewards.begin(), cumulative);
    }
    return torch::tensor(discounted_rewards).to(torch::kF64);
}

torch::Tensor PPOAgent::compute_gae(double last_value) {
    std::vector<double> advantages;
    double gae = 0.0;
    double next_value = last_value;
    for (int t = rewards.size() - 1; t >= 0; --t) {
        double delta = rewards[t] + gamma * next_value - values[t].item<double>();
        gae = delta + gamma * lambda * gae;
        advantages.insert(advantages.begin(), gae);
        next_value = values[t].item<double>();
    }
    return torch::tensor(advantages).to(torch::kF64);
}

std::tuple<int, int, int> PPOAgent::runStep() {
    Stopwatch sw;
    sw.start();
    std::tuple<int, int, int> action = selectAction(states.back());
    avg_reward += rewards[counter] / update_steps;

    counter++;
    resolution_actions.push_back(std::get<0>(action));
    batching_actions.push_back(std::get<1>(action));
    scaling_actions.push_back(std::get<2>(action));
    sw.stop();
    out << sw.elapsed_microseconds() << "," << counter << "," << avg_reward << "," << std::get<0>(action) << "," << std::get<1>(action) << "," << std::get<2>(action) << std::endl;

    if (counter%update_steps == 0) {
        std::thread t(&PPOAgent::update, this);
        t.detach();
    }
    return std::make_tuple(std::get<0>(action) + 1, std::get<1>(action) + 1, std::get<2>(action) + 1);
}
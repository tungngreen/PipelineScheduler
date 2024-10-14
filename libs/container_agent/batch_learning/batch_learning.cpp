#include "batch_learning.h"

PPOAgent::PPOAgent(std::string& cont_name, uint max_batch, uint update_steps, double lambda, double gamma,
                   const std::string& model_save) : max_batch(max_batch), update_steps(update_steps), lambda(lambda), gamma(gamma) {
    path = "../models/batch_learning/" + cont_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log.csv");

    std::random_device rd;
    re = std::mt19937(rd());
    actor_critic_net = std::make_shared<ActorCriticNet>(4, max_batch);
    actor_critic_net->to(torch::kF64);
    if (!model_save.empty()) torch::load(actor_critic_net, model_save);

    optimizer = std::make_unique<torch::optim::Adam>(actor_critic_net->parameters(), torch::optim::AdamOptions(1e-3));

    avg_reward = 0.0;
    states = {};
    actions = {};
    rewards = {};
    log_probs = {};
    values = {};
}

void PPOAgent::update() {
    spdlog::info("Updating reinforcement learning batch size model!");
    Stopwatch sw;
    sw.start();


    auto [policy, v] = actor_critic_net->forward(torch::stack(states));
    auto a = torch::tensor(actions).view({-1, 1, 1});
    auto new_log_probs = torch::log(policy.gather(-1, a).squeeze(-1));

    auto ratio = torch::exp(new_log_probs - torch::stack(log_probs));
    auto clipped_ratio = torch::clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon);
    torch::Tensor advantages = compute_gae();

    auto policy_loss = -torch::min(ratio * advantages, clipped_ratio * advantages).to(torch::kF64).mean();
    auto value_loss = torch::mse_loss(v, compute_cumu_rewards()).to(torch::kF64);
    auto loss = policy_loss + 0.5 * value_loss;

    // Backpropagation
    optimizer->zero_grad();
    loss.to(torch::kF64).backward();
    optimizer->step();
    sw.stop();

    std::cout << "Training: " << sw.elapsed_microseconds() << "," << counter << "," << avg_reward << std::endl;

    counter = 0;
    avg_reward = 0.0;
    states.clear();
    actions.clear();
    rewards.clear();
    log_probs.clear();
    values.clear();
}

void PPOAgent::rewardCallback(double throughput, double drops, double latency_penalty, double oversize_penalty) {
    rewards.push_back(throughput - drops - latency_penalty + (1 - oversize_penalty));
}

void PPOAgent::setState(double curr_batch, double arrival, double pre_queue_size, double inf_queue_size) {
    states.push_back(torch::tensor({{curr_batch / max_batch, arrival, pre_queue_size, inf_queue_size}}, torch::kF64));
}

int PPOAgent::selectAction(torch::Tensor state) {
    auto [policy, value] = actor_critic_net->forward(state);

    torch::Tensor action_dist = torch::multinomial(policy, 1);  // Sample from policy (discrete distribution)
    int action = action_dist.item<int>();  // Convert tensor to int action

    log_probs.push_back(torch::log(policy.squeeze(0)[action]));
    values.push_back(value);
    return action;
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

int PPOAgent::runStep() {
    Stopwatch sw;
    sw.start();
    int action = selectAction(states.back());
    avg_reward += rewards[counter] / update_steps;

    counter++;
    actions.push_back(action);
    sw.stop();
    out << sw.elapsed_microseconds() << "," << counter << "," << avg_reward << "," << action << std::endl;

    if (counter%update_steps == 0) {
        std::thread t(&PPOAgent::update, this);
        t.detach();
    } else {
    }

    return action + 1;
}
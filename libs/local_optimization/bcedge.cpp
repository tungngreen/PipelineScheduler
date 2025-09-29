#include "bcedge.h"

BCEdgeAgent::BCEdgeAgent(std::string path, double max_memory, torch::Dtype precision,
                         uint update_steps, double lambda, double gamma, double clip_epsilon)
                         : precision(precision), path(path), max_memory(max_memory), lambda(lambda),
                           gamma(gamma), clip_epsilon(clip_epsilon), update_steps(update_steps) {
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log_" + getTimestampString() + ".csv");

    model = std::make_shared<BCEdgeNet>(5, 64, 2, 2);
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

void BCEdgeAgent::update() {
    std::unique_lock<std::mutex> lock(model_mutex);
    spdlog::get("container_agent")->info("Locally training RL agent at cumulative Reward {}!", cumu_reward);
    Stopwatch sw;
    sw.start();

    auto [policy1, policy2, policy3, val] = model->forward(torch::stack(states));
    T action1_probs = torch::softmax(policy1, -1);
    T action1_log_probs = torch::log(action1_probs.gather(-1, torch::tensor(batching_actions).reshape({-1, 1})).squeeze(-1));
    T action2_probs = torch::softmax(policy2, -1);
    T action2_log_probs = torch::log(action2_probs.gather(-1, torch::tensor(scaling_actions).reshape({-1, 1})).squeeze(-1));
    T action3_probs = torch::softmax(policy3, -1);
    T action3_log_probs = torch::log(action3_probs.gather(-1, torch::tensor(memory_actions).reshape({-1, 1})).squeeze(-1));
    T new_log_probs = (action1_log_probs + action2_log_probs + action3_log_probs).squeeze(-1);

    T ratio = torch::exp(new_log_probs - torch::stack(log_probs));

    if (rewards.size() < states.size()) {
        ratio = ratio.slice(0, 1, ratio.size(0), 1);
        val = val.slice(0, 1, val.size(0), 1);
    }

    T clipped_ratio = torch::clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon);
    T advantages = computeGae();
    T policy_loss = -torch::min(ratio * advantages, clipped_ratio * advantages).to(precision).mean();

    T value_loss = torch::mse_loss(val.squeeze(), computeCumuRewards());
    T loss = (policy_loss + 0.5 * value_loss);

    // Backpropagation
    optimizer->zero_grad();
    loss.backward();
    optimizer->step();
    sw.stop();

    double avg_reward = cumu_reward / (double) update_steps;
    out << "episodeEnd," << sw.elapsed_microseconds() << "," << 0 << "," << steps_counter
        << "," << cumu_reward << "," << avg_reward << "," << loss.item<double>() << "," << policy_loss.item<double>() << "," << value_loss.item<double>() << std::endl;
    steps_counter = 0;
    reset();
}

void BCEdgeAgent::rewardCallback(double throughput, double latency, MsvcSLOType slo, double memory_usage) {
    if (update_steps > 0 ) {
        double tmp_reward;
        if (latency <= slo && memory_usage <= max_memory) {
            tmp_reward = log(throughput / (latency / slo));
        } else {
            tmp_reward = exp(-latency * (memory_usage));
        }
        if (std::isnan(tmp_reward)) tmp_reward = 0.0;
        rewards.push_back(std::min(25.0, std::max(-25.0, tmp_reward)) / 25.0); // Normalize reward to be in the range of [-1, 1] for better training
    }
}

void BCEdgeAgent::setState(ModelType model_type, std::vector<int> data_shape, MsvcSLOType slo) {
    state = torch::tensor({(double)model_type / (double)ModelType::End, (double) data_shape[0] / 3.0,(double) data_shape[1] / 1000.0, (double)data_shape[2] / 1000.0, (double) slo / (double)TIME_PRECISION_TO_SEC}, precision);
    if (torch::any(torch::isnan(state)).item<bool>()) {
        state = torch::nan_to_num(state);
    }
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

    if (update_steps > 0 ) {
        log_prob = torch::log(policy1[batching]) + torch::log(policy2[scaling]) + torch::log(policy3[memory]);
        states.push_back(state);
        log_probs.push_back(log_prob);
        values.push_back(val);
        batching_actions.push_back(batching);
        scaling_actions.push_back(scaling);
        memory_actions.push_back(memory);
    }
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

std::tuple<int, int, int> BCEdgeAgent::runStep() {
    Stopwatch sw;
    sw.start();
    selectAction();
    cumu_reward += (steps_counter) ? rewards[steps_counter - 1] : 0;

    if (update_steps > 0 ) steps_counter++;
    sw.stop();
    out << "step," << sw.elapsed_microseconds() << "," << 0 << "," << steps_counter << "," << cumu_reward  << "," << batching << "," << scaling << "," << memory << std::endl;

    if (update_steps > 0 ) {
        if (steps_counter % update_steps == 0) {
            update();
        }
    }
    return std::make_tuple(batching + 1, scaling + 1, memory);
}
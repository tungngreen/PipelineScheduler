#include "PAHC.h"

PAHC::PAHC(const std::string& exp_name, uint state_size, uint weights_size, torch::Dtype precision, uint update_steps,
           int seed) : precision(precision), state_size(state_size), weights_size(weights_size),
           update_steps(update_steps) {
    path = "../models/apis_PAHC/" + exp_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log_" + getTimestampString() + ".csv");

    actor = std::make_shared<PAHC_Actor>(state_size, weights_size);
    std::string model_save = path + "/latest_actor.pt";
    if (std::filesystem::exists(model_save)) {
        torch::load(actor, model_save);
    } else {
        torch::manual_seed(seed);
        for (auto& p : actor->named_parameters()) {
            if(p.key().find("norm") != std::string::npos) continue;
            if (p.key().find("weight") != std::string::npos) {
                torch::nn::init::xavier_uniform_(p.value());
            } else if (p.key().find("bias") != std::string::npos) {
                torch::nn::init::constant_(p.value(), 0);
            }
        }
    };
    actor->to(precision);

    critic1 = std::make_shared<PAHC_Critic>(state_size, weights_size);
    model_save = path + "/latest_critic1.pt";
    if (std::filesystem::exists(model_save)) {
        torch::load(critic1, model_save);
    } else {
        torch::manual_seed(seed);
        for (auto& p : critic1->named_parameters()) {
            if(p.key().find("norm") != std::string::npos) continue;
            if (p.key().find("weight") != std::string::npos) {
                torch::nn::init::xavier_uniform_(p.value());
            } else if (p.key().find("bias") != std::string::npos) {
                torch::nn::init::constant_(p.value(), 0);
            }
        }
    };
    critic1->to(precision);

    critic2 = std::make_shared<PAHC_Critic>(state_size, weights_size);
    model_save = path + "/latest_critic2.pt";
    if (std::filesystem::exists(model_save)) {
        torch::load(critic2, model_save);
    } else {
        torch::manual_seed(seed);
        for (auto& p : critic2->named_parameters()) {
            if(p.key().find("norm") != std::string::npos) continue;
            if (p.key().find("weight") != std::string::npos) {
                torch::nn::init::xavier_uniform_(p.value());
            } else if (p.key().find("bias") != std::string::npos) {
                torch::nn::init::constant_(p.value(), 0);
            }
        }
    };
    critic2->to(precision);

    critic1_target = critic1->clone();
    critic1_target->to(precision);
    critic2_target = critic2->clone();
    critic2_target->to(precision);

    optimizer_actor = std::make_unique<torch::optim::AdamW>(actor->parameters(), torch::optim::AdamWOptions(1e-3));
    optimizer_critic1 = std::make_unique<torch::optim::AdamW>(critic1->parameters(), torch::optim::AdamWOptions(1e-3));
    optimizer_critic2 = std::make_unique<torch::optim::AdamW>(critic2->parameters(), torch::optim::AdamWOptions(1e-3));

    log_alpha_meta = torch::zeros({1}, torch::requires_grad(true));
    optimizer_alpha = std::make_unique<torch::optim::Adam>(std::vector<torch::Tensor>{log_alpha_meta}, torch::optim::AdamOptions(1e-4));

    cumu_reward = 0.0;
    steps_counter = 0;
    experiences = PAHC_ExperienceBuffer(1000);
}

std::vector<float> PAHC::runStep() {
    Stopwatch sw;
    sw.start();
    T output = selectAction();
    weights.clear();
    sw.stop();

    out << "step," << sw.elapsed_microseconds() << "," << steps_counter << "," << cumu_reward  << "," ;
    for (unsigned int i = 0; i < weights_size; i++) {
        out << output[i].item<float>();
        weights.emplace_back(output[i].item<float>());
        if (i < weights_size - 1) out << ",";
    }
    out << std::endl;

    if (update_steps > 0 ) {
        steps_counter++;
    }
    return weights;
}

void PAHC::rewardCallback(double throughput, double latency, double memory_use) {
    if (update_steps > 0 ) {
        if (first) {
            return;
        }
        double reward = (throughput + latency + memory_use) / 3.0;
        spdlog::get("container_agent")->trace("RL Agent Reward: throughput: {}, latency_penalty: {}, memory_use: {}",
                                              throughput, latency, memory_use);
        experiences.add_reward(reward);
        cumu_reward += reward;
    }
}

void PAHC::setState(double agentID, double agentType, double pipeType, double pipeLatency, double localLatency,
                    double theta, double pipeThroughput, double sigma, double memoryUse, double phi, double rho) {
    spdlog::get("container_agent")->trace("RL Agent State: agentID: {}, agentType: {}, pipeType: {}, pipeLatency: {}, "
                                          "localLatency: {}, theta: {}, pipeThroughput: {}, sigma: {}, memoryUse: {}, "
                                          "phi: {}, rho: {}",
                                          agentID, agentType, pipeType, pipeLatency, localLatency, theta, pipeThroughput,
                                          sigma, memoryUse, phi, rho);
    state = torch::tensor({agentID, agentType, pipeType, pipeLatency, localLatency, theta, pipeThroughput, sigma,
                           memoryUse, phi, rho}, precision);
    new_states.push_back(state);
}

T PAHC::selectAction() {
    auto [mean, log_std] = actor->forward(state);
    auto std = torch::exp(log_std);
    auto noise = torch::randn_like(mean);
    auto raw_W = mean + noise * std;
    auto W = torch::tanh(raw_W);

    if (update_steps > 0 ) {
        experiences.add(state, log_std, mean, W);
    }
    return W;
}

void PAHC::update() {
    if (steps_counter < update_steps)
        return;
    spdlog::get("container_agent")->info("Training ApisRL Agent at cumulative Reward {}!", cumu_reward);
    Stopwatch sw;
    sw.start();
    T states, actions, log_stds, means, rewards;
    try {
        std::tie(states, actions, log_stds, means, rewards) = experiences.sample(steps_counter);
    } catch (const c10::Error& e) {
        spdlog::get("container_agent")->error("Error sampling experiences for PAHC update: {}", e.what());
        reset();
        experiences.clear();
        return;
    }
    rewards.to(precision);

    size_t buffer_size = rewards.size(0);
    if (buffer_size < new_states.size()) {
        new_states.erase(new_states.begin(), new_states.end() - buffer_size);
    } else if (buffer_size > new_states.size()) {
        spdlog::get("container_agent")->warn("Not enough new states collected for PAHC update! Expected: {}, Got: {}. Skipping update.",
                                             buffer_size, new_states.size());
        buffer_size = new_states.size();
        states = states.slice(0, 0, buffer_size, 1);
        actions = actions.slice(0, 0, buffer_size, 1);
        log_stds = log_stds.slice(0, 0, buffer_size, 1);
        means = means.slice(0, 0, buffer_size, 1);
        rewards = rewards.slice(0, 0, buffer_size, 1);
    }
    T next_states = torch::stack(new_states).to(precision);

    // --- CRITIC UPDATE (L_Q) ---
    auto [mean, log_std] = actor->forward(next_states);
    auto next_std = torch::exp(log_std);
    auto next_noise = torch::randn_like(mean);
    auto next_raw_W = mean + next_noise * next_std;
    auto next_actions = torch::tanh(next_raw_W);
    auto next_log_prob = (-0.5 * ((next_raw_W - mean) / next_std).pow(2) - log_std - 0.5 * std::log(2 * M_PI) - torch::log(1.0 - next_actions.pow(2) + 1e-6)).sum(1, true);

    auto target_q1 = critic1_target->forward(next_states, next_actions);
    auto target_q2 = critic2_target->forward(next_states, next_actions);
    double alpha_meta = std::exp(log_alpha_meta.item<double>());

    // Soft Q target: Y^m = R^m + gamma * [ min(Q_target) - alpha_meta * log(pi(W'|S')) ]
    auto min_target_q = torch::min(target_q1, target_q2);
    auto next_q_value = min_target_q - alpha_meta * next_log_prob;
    auto target_q_value = rewards + 0.99 * next_q_value;

    auto current_q1 = critic1->forward(states, actions);
    auto current_q2 = critic2->forward(states, actions);

    auto critic1_loss = torch::mse_loss(current_q1, target_q_value.detach());
    auto critic2_loss = torch::mse_loss(current_q2, target_q_value.detach());

    optimizer_critic1->zero_grad();
    torch::autograd::backward({critic1_loss}, {}, /*retain_graph=*/true);
    optimizer_critic1->step();
    torch::save(critic1, path + "/latest_critic1.pt");
    optimizer_critic2->zero_grad();
    torch::autograd::backward({critic2_loss}, {}, /*retain_graph=*/true);
    optimizer_critic2->step();
    torch::save(critic2, path + "/latest_critic2.pt");

    // --- ACTOR UPDATE (L_pi) ---
    auto q1_pi = critic1->forward(states, actions);
    auto q2_pi = critic2->forward(states, actions);
    auto min_q_pi = torch::min(q1_pi, q2_pi);

    // NOTE: In the multi-objective case, min_q_pi should be scalarized using V.
    // Assuming reward tensor 'reward' is scalarized by V to get the single R used above.
    // To strictly implement the suggested loss:
    // L_pi = E[ alpha*log(pi(W|S)) - V * min(Q(S,W)) ]

    auto raw_W = actions.atanh(); // Inverse of tanh to get raw actions
    auto std = torch::exp(log_stds);
    auto log_probs = (-0.5 * ((raw_W - means) / std).pow(2) - log_stds - 0.5 * std::log(2 * M_PI) - torch::log(1.0 - actions.pow(2) + 1e-6)).sum(0, true);
    auto actor_loss = (alpha_meta * log_probs - min_q_pi).mean();

    optimizer_actor->zero_grad();
    torch::autograd::backward({actor_loss}, {}, /*retain_graph=*/true);
    optimizer_actor->step();
    torch::save(actor, path + "/latest_actor.pt");

    // --- ALPHA (Temperature) UPDATE (L_alpha) ---
    auto alpha_loss = -(log_alpha_meta * (log_probs.detach() + (-static_cast<double>(weights_size)))).mean();

    optimizer_alpha->zero_grad();
    torch::autograd::backward({alpha_loss}, {}, /*retain_graph=*/true);
    optimizer_alpha->step();
    soft_update_targets();

    double avg_reward = cumu_reward / (double) update_steps;
    out << "episodeEnd," << sw.elapsed_microseconds() << "," << steps_counter << "," << cumu_reward << ","
        << avg_reward << std::endl;
    reset();
}

void PAHC::soft_update_targets() {
    torch::NoGradGuard no_grad;
    constexpr double tau = 0.005;
    constexpr double one_minus_tau = 1.0 - tau;

    for (const auto& pair : critic1->named_parameters()) {
        const auto& name = pair.key();
        const auto& param = pair.value();
        auto target_param = critic1_target->named_parameters()[name];

        target_param.data().copy_(target_param.data() * one_minus_tau + param.data() * tau);
    }

    for (const auto& pair : critic2->named_parameters()) {
        const auto& name = pair.key();
        const auto& param = pair.value();
        auto target_param = critic2_target->named_parameters()[name];

        target_param.data().copy_(target_param.data() * one_minus_tau + param.data() * tau);
    }
}
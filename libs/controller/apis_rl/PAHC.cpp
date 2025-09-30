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
    for (int i = 0; i < weights_size; i++) {
        out << output[0][i].item<float>();
        weights.emplace_back(output[0][i].item<float>());
        if (i < weights_size - 1) out << ",";
    }
    out << std::endl;

    if (update_steps > 0 ) {
        steps_counter++;
        if (steps_counter % update_steps == 0) {
            update();
        }
    }
    return weights;
}

void PAHC::rewardCallback(double throughput, double latency, double memory_use) {
    if (update_steps > 0 ) {
        if (first) { // First reward is not valid and needs to be discarded
            first = false;
            return;
        }
        double reward = (throughput + latency + memory_use) / 3.0;
        spdlog::get("container_agent")->trace("RL Agent Reward: throughput: {}, latency_penalty: {}, memory_use: {}",
                                              throughput, latency, memory_use);
        experiences.add_reward(reward);
        cumu_reward += reward;
    }
}

void PAHC::setState(double agentID) {
    spdlog::get("container_agent")->trace("RL Agent State: ");
    state = torch::tensor({agentID}, precision);
    new_states.push_back(state);
}

T PAHC::selectAction() {
    auto [mean, log_std] = actor->forward(state);
    auto std = torch::exp(log_std);
    auto noise = torch::randn_like(mean);
    auto raw_W = mean + noise * std;
    auto W = torch::tanh(raw_W);

    if (update_steps > 0 ) {
        log_prob = (-0.5 * ((raw_W - mean) / std).pow(2) - log_std - 0.5 * std::log(2 * M_PI) - torch::log(1.0 - W.pow(2) + 1e-6)).sum(1, true);
        experiences.add(state, log_prob, mean, W);
    }
    return W;
}

void PAHC::update() {
    steps_counter = 0;
    spdlog::get("container_agent")->info("Locally training RL Agent at cumulative Reward {}!", cumu_reward);
    Stopwatch sw;
    sw.start();
    auto states = torch::stack(experiences.get_states());
    auto actions = torch::stack(experiences.get_weights());
    auto log_probs = torch::stack(experiences.get_log_probs());

    if (experiences.get_rewards().size() < experiences.get_states().size()) {
        states.slice(0, 1, states.size(0), 1);
        actions.slice(0, 1, actions.size(0), 1);
        log_probs.slice(0, 1, log_probs.size(0), 1);
    }

    // --- CRITIC UPDATE (L_Q) ---
    auto [mean, log_std] = actor->forward(torch::stack(new_states));
    auto std = torch::exp(log_std);
    auto noise = torch::randn_like(mean);
    auto raw_W = mean + noise * std;
    auto next_actions = torch::tanh(raw_W);
    auto next_log_prob = (-0.5 * ((raw_W - mean) / std).pow(2) - log_std - 0.5 * std::log(2 * M_PI) - torch::log(1.0 - next_actions.pow(2) + 1e-6)).sum(1, true);

    auto target_q1 = critic1_target->forward(states, next_actions);
    auto target_q2 = critic2_target->forward(states, next_actions);
    double alpha_meta = std::exp(log_alpha_meta.item<double>());

    // Soft Q target: Y^m = R^m + gamma * [ min(Q_target) - alpha_meta * log(pi(W'|S')) ]
    auto min_target_q = torch::min(target_q1, target_q2);
    auto next_q_value = min_target_q - alpha_meta * next_log_prob;
    auto target_q_value = torch::tensor(experiences.get_rewards()).to(precision) + 0.99 * next_q_value;

    auto current_q1 = critic1->forward(states, actions);
    auto current_q2 = critic2->forward(states, actions);

    auto critic1_loss = torch::mse_loss(current_q1, target_q_value.detach());
    auto critic2_loss = torch::mse_loss(current_q2, target_q_value.detach());

    optimizer_critic1->zero_grad();
    critic1_loss.backward();
    optimizer_critic1->step();
    optimizer_critic2->zero_grad();
    critic2_loss.backward();
    optimizer_critic2->step();

    // --- ACTOR UPDATE (L_pi) ---
    auto q1_pi = critic1->forward(states, actions);
    auto q2_pi = critic2->forward(states, actions);
    auto min_q_pi = torch::min(q1_pi, q2_pi);

    // NOTE: In the multi-objective case, min_q_pi should be scalarized using V.
    // Assuming reward tensor 'reward' is scalarized by V to get the single R used above.
    // To strictly implement the suggested loss:
    // L_pi = E[ alpha*log(pi(W|S)) - V * min(Q(S,W)) ]
    auto actor_loss = (alpha_meta * log_probs - min_q_pi).mean();

    optimizer_actor->zero_grad();
    actor_loss.backward();
    optimizer_actor->step();

    // --- ALPHA (Temperature) UPDATE (L_alpha) ---
    auto alpha_loss = -(log_alpha_meta * (log_prob.detach() + (-static_cast<double>(weights_size)))).mean();

    optimizer_alpha->zero_grad();
    alpha_loss.backward();
    optimizer_alpha->step();
    soft_update_targets();

    double avg_reward = cumu_reward / (double) update_steps;
    out << "episodeEnd," << sw.elapsed_microseconds() << "," << steps_counter << "," << cumu_reward << ","
        << avg_reward << std::endl;

    reset();
}

void PAHC::soft_update_targets() {
    torch::NoGradGuard no_grad; // Ensure no gradients are tracked during update

    // Update Critic 1 Target
    for (const auto& pair : critic1->named_parameters()) {
        auto name = pair.key();
        auto param = pair.value();
        auto target_param = critic1_target->named_parameters()[name];
        target_param.copy_(target_param * (1.0 - 0.005) + param * 0.005);
    }

    // Update Critic 2 Target
    for (const auto& pair : critic2->named_parameters()) {
        auto name = pair.key();
        auto param = pair.value();
        auto target_param = critic2_target->named_parameters()[name];
        target_param.copy_(target_param * (1.0 - 0.005) + param * 0.005);
    }
}
#include "fcpo_learning.h"

// -------------------------------------------------------------------------
// FCPO Agent Implementation (MOMAPPO Style, No FL)
// -------------------------------------------------------------------------

FCPOAgent::FCPOAgent(std::string& cont_name, uint state_size, uint timeout_size, uint max_batch,  uint scaling_size,
                     socket_t *socket, BatchInferProfileListType profile, int base_batch, torch::Dtype precision,
                     uint update_steps, uint update_steps_inc, uint federated_steps, double lambda, double gamma,
                     double clip_epsilon, double penalty_weight, double theta, double sigma, double phi, double rho,
                     int seed)
        : precision(precision), batchInferProfileList(profile), cont_name(cont_name),
          lambda(lambda), gamma(gamma), clip_epsilon(clip_epsilon), penalty_weight(penalty_weight), theta(theta),
          sigma(sigma), phi(phi), rho(rho), state_size(state_size), timeout_size(timeout_size), max_batch(max_batch),
          base_batch(base_batch), scaling_size(scaling_size), update_steps(update_steps),
          update_steps_inc(update_steps_inc) {

    path = "../models/momappo_fcpo_learning/" + cont_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log.csv");

    model = std::make_shared<MultiPolicyNet>(state_size, max_batch, timeout_size, scaling_size);
    std::string model_save = path + "/latest_model.pt";
    if (std::filesystem::exists(model_save)) {
        torch::load(model, model_save);
    } else {
        torch::manual_seed(seed);
        for (auto& p : model->named_parameters()) {
            if(p.key().find("norm") != std::string::npos) continue;
            if (p.key().find("weight") != std::string::npos) {
                torch::nn::init::xavier_uniform_(p.value());
            } else if (p.key().find("bias") != std::string::npos) {
                torch::nn::init::constant_(p.value(), 0);
            }
        }
    };
    model->to(precision);
    optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(1e-3));

    cumu_reward = 0.0;
    last_batching = 0;
    last_slo = 0.0;
    steps_counter = 0;
    experiences = FCPO_ExperienceBuffer(200);
}

void FCPOAgent::update() {
    steps_counter = 0;

    // 1. Prepare Data
    auto states = torch::stack(experiences.get_states()).to(precision);
    auto batching_actions = torch::tensor(experiences.get_batching()).to(torch::kInt64).reshape({-1, 1}).to(model->parameters()[0].device());
    auto timeout_actions = torch::tensor(experiences.get_timeout()).to(torch::kInt64).reshape({-1, 1}).to(model->parameters()[0].device());
    auto scaling_actions = torch::tensor(experiences.get_scaling()).to(torch::kInt64).reshape({-1, 1}).to(model->parameters()[0].device());
    auto old_log_probs = torch::stack(experiences.get_log_probs()).to(precision);
    auto returns = computeCumuRewards().detach();

    // 2. Forward Pass
    auto [policy1, policy2, policy3, val] = model->forward(states);

    // 3. Calculate Entropy
    T action1_probs = torch::softmax(policy1, -1);
    T action2_probs = torch::softmax(policy2, -1);
    T action3_probs = torch::softmax(policy3, -1);

    T entropy1 = -(action1_probs * torch::log(action1_probs + 1e-10)).sum(-1);
    T entropy2 = -(action2_probs * torch::log(action2_probs + 1e-10)).sum(-1);
    T entropy3 = -(action3_probs * torch::log(action3_probs + 1e-10)).sum(-1);
    T entropy = (entropy1 + entropy2 + entropy3).mean();

    // 4. Calculate New Log Probs
    T action1_log_probs = torch::log(action1_probs.gather(-1, batching_actions).squeeze(-1) + 1e-10);
    T action2_log_probs = torch::log(action2_probs.gather(-1, timeout_actions).squeeze(-1) + 1e-10);
    T action3_log_probs = torch::log(action3_probs.gather(-1, scaling_actions).squeeze(-1) + 1e-10);
    T new_log_probs = (action1_log_probs + action2_log_probs + action3_log_probs);

    // Safety checks for buffer alignment
    if (returns.size(0) < new_log_probs.size(0)) {
        new_log_probs = new_log_probs.slice(0, 0, returns.size(0));
        old_log_probs = old_log_probs.slice(0, 0, returns.size(0));
        val = val.slice(0, 0, returns.size(0));
    }

    // 5. Calculate Ratio and Advantages
    T ratio = torch::exp(new_log_probs - old_log_probs);
    T advantages = computeGae().detach();

    // Normalize Advantages (Standard MOMAPPO)
    if (advantages.size(0) > 1) {
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
    }

    T loss = torch::tensor(0.0), policy_loss = torch::tensor(0.0), value_loss = torch::tensor(0.0);
    try {
        // 6. Surrogate Loss (PPO Clip)
        T surr1 = ratio * advantages;
        T surr2 = torch::clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages;
        policy_loss = -torch::min(surr1, surr2).mean();

        // 7. Value Loss (MSE)
        value_loss = torch::nn::functional::mse_loss(val.squeeze(-1), returns);

        // 8. Total Loss (Policy + VF - Entropy)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy;

        std::cout << "MOMAPPO Policy Loss: " << policy_loss.item<double>()
                  << " Value Loss: " << value_loss.item<double>()
                  << " Entropy: " << entropy.item<double>() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error in loss computation: " << e.what() << std::endl;
        reset();
        return;
    }

    // Backpropagation
    std::unique_lock<std::mutex> lock(model_mutex);
    optimizer->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5);
    optimizer->step();

    double avg_reward = cumu_reward / (double) update_steps;
    out << "update," << steps_counter << "," << cumu_reward << "," << avg_reward << "," << loss.item<double>() << std::endl;

    if (last_avg_reward < avg_reward) {
        last_avg_reward = avg_reward;
        torch::save(model, path + "/latest_model.pt");
    }

    reset();
}

void FCPOAgent::rewardCallback(double throughput, double latency, double oversize, double memory_use) {
    if (update_steps > 0 ) {
        if (first) {
            first = false;
            return;
        }

        // Handle case where profile is empty (from simulation script)
        double max_mem = 1.0;
        if (!batchInferProfileList.empty() && batchInferProfileList.size() > max_batch) {
            max_mem = (double)(batchInferProfileList[max_batch].gpuMemUsage + 1.0);
        } else {
            // Fallback if no profile provided: approximate normalization
            max_mem = memory_use > 0 ? memory_use * 1.5 : 1000.0;
        }

        // MOMAPPO: Linear Scalarization of Objectives
        // R = w * r (Using theta, sigma, phi, rho as the weight vector w)
        double reward = theta * throughput
                        - sigma * latency
                        - phi * oversize
                        - rho * (memory_use / max_mem);

        // Clipping to maintain stability
        reward = std::max(-1.0, std::min(1.0, reward));

        experiences.add_reward(reward);
        cumu_reward += reward;
    }
}

void FCPOAgent::setState(double curr_timeout, double curr_batch, double curr_scaling,  double arrival,
                         double pre_queue_size, double inf_queue_size, double post_queue_size, double slo,
                         double memory_use) {
    last_slo = slo;
    double maxMemory;
    if (batchInferProfileList.empty())
        maxMemory = memory_use * 2.0 + 1;
    else
        maxMemory = (double) batchInferProfileList[max_batch].gpuMemUsage + 1.0;

    state = torch::tensor({curr_batch / max_batch, curr_timeout / timeout_size,
                           curr_scaling / scaling_size, arrival, pre_queue_size / 100.0, inf_queue_size / 100.0,
                           post_queue_size / 100.0, slo / 100000.0,
                           memory_use / maxMemory}, precision);
}

void FCPOAgent::selectAction() {
    std::unique_lock<std::mutex> lock(model_mutex);
    auto [policy1, policy2, policy3, val] = model->forward(state);

    T action_dist = torch::multinomial(policy1, 1);
    batching = action_dist.item<int>();

    action_dist = torch::multinomial(policy2, 1);
    timeout = action_dist.item<int>();

    action_dist = torch::multinomial(policy3, 1);
    scaling = action_dist.item<int>();

    // Safety check for empty profiles in simulation mode
    if (!batchInferProfileList.empty() && batching > 0 && batchInferProfileList[batching].p95inferLat > last_slo) {
        penalty = true;
        batching = std::max(last_batching, base_batch);
    }

    last_batching = batching;
    if (update_steps > 0 ) {
        log_prob = torch::log(policy1[batching] + 1e-10) + torch::log(policy2[timeout] + 1e-10) + torch::log(policy3[scaling] + 1e-10);
        experiences.add(state, log_prob, val, timeout, batching, scaling);
    }
}

T FCPOAgent::computeCumuRewards() const {
    std::vector<double> discounted_rewards, rewards = experiences.get_rewards();
    double cumulative = 0.0;
    for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
        cumulative = *it + gamma * cumulative;
        discounted_rewards.insert(discounted_rewards.begin(), cumulative);
    }
    return torch::tensor(discounted_rewards).to(precision);
}

T FCPOAgent::computeGae() const {
    std::vector<double> advantages, rewards = experiences.get_rewards();
    std::vector<T> values = experiences.get_values();
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

std::tuple<int, int, int> FCPOAgent::runStep() {
    selectAction();

    if (update_steps > 0 ) steps_counter++;

    out << "step," << steps_counter << "," << cumu_reward  << "," << timeout << "," << batching << "," << scaling << std::endl;

    if (update_steps > 0 ) {
        if (steps_counter % update_steps == 0) {
            update();
        }
    }
    return std::make_tuple((timeout + 1) % (timeout_size), batching + 1, scaling);
}

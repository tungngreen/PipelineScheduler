#include "fcpo_learning.h"
#include <torch/torch.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

// ============================================================================
// IPPO Replay Buffer Implementation (Standard IPPO)
// ============================================================================
class IPPOBuffer {
public:
    std::vector<torch::Tensor> states;
    std::vector<int64_t> timeouts;
    std::vector<int64_t> batchings;
    std::vector<int64_t> scalings;
    std::vector<double> log_probs;
    std::vector<double> rewards;
    std::vector<double> values;
    std::vector<bool> terminals;

    size_t capacity;
    size_t ptr = 0;
    bool is_full = false;

    IPPOBuffer(size_t size) : capacity(size) {
        states.reserve(size);
        timeouts.reserve(size);
        batchings.reserve(size);
        scalings.reserve(size);
        log_probs.reserve(size);
        rewards.reserve(size);
        values.reserve(size);
        terminals.reserve(size);
    }

    void add(const torch::Tensor& state, int64_t timeout, int64_t batching, int64_t scaling,
             double log_prob, double value, double reward, bool terminal) {
        if (states.size() < capacity) {
            states.push_back(state);
            timeouts.push_back(timeout);
            batchings.push_back(batching);
            scalings.push_back(scaling);
            log_probs.push_back(log_prob);
            rewards.push_back(reward);
            values.push_back(value);
            terminals.push_back(terminal);
        } else {
            // Overwrite if full (though IPPO usually clears after update)
            states[ptr] = state;
            timeouts[ptr] = timeout;
            batchings[ptr] = batching;
            scalings[ptr] = scaling;
            log_probs[ptr] = log_prob;
            rewards[ptr] = reward;
            values[ptr] = value;
            terminals[ptr] = terminal;
        }
        ptr = (ptr + 1) % capacity;
        if (ptr == 0) is_full = true;
    }

    void clear() {
        states.clear();
        timeouts.clear();
        batchings.clear();
        scalings.clear();
        log_probs.clear();
        rewards.clear();
        values.clear();
        terminals.clear();
        ptr = 0;
        is_full = false;
    }

    size_t size() const { return states.size(); }

    // Compute Generalized Advantage Estimation (GAE)
    // Returns: {Advantages, Returns (Targets)}
    std::pair<torch::Tensor, torch::Tensor> compute_gae(double gamma, double gae_lambda, double last_value) {
        size_t n = rewards.size();
        std::vector<double> advantages(n);
        std::vector<double> returns(n);
        double gae = 0.0;
        double next_value = last_value;

        for (int i = n - 1; i >= 0; --i) {
            double mask = terminals[i] ? 0.0 : 1.0;
            double delta = rewards[i] + gamma * next_value * mask - values[i];
            gae = delta + gamma * gae_lambda * mask * gae;
            advantages[i] = gae;
            next_value = values[i];
            returns[i] = gae + values[i];
        }

        auto adv_t = torch::tensor(advantages, torch::kFloat32);
        auto ret_t = torch::tensor(returns, torch::kFloat32);
        return {adv_t, ret_t};
    }
};

// ============================================================================
// FCPO Agent Implementation
// ============================================================================

// Initialize the new IPPO Buffer
std::unique_ptr<IPPOBuffer> IPPO_buffer = std::make_unique<IPPOBuffer>(5000);

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

    path = "../models/fcpo_learning/" + cont_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log_" + getTimestampString() + ".csv");

    model = std::make_shared<MultiPolicyNet>(state_size, max_batch, timeout_size, scaling_size);
    std::string model_save = path + "/latest_model.pt";
    if (std::filesystem::exists(model_save)) {
        torch::load(model, model_save);
    } else {
        torch::manual_seed(seed);
        for (auto& p : model->named_parameters()) {
            if(p.key().find("norm") != std::string::npos) continue;
            // Initialize weights and biases
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
}

void FCPOAgent::update() {
    steps_counter = 0;

    if (IPPO_buffer->size() == 0) {
        spdlog::get("container_agent")->trace("Buffer empty, skipping update.");
        return;
    }

    spdlog::get("container_agent")->info("Locally training RL Agent (IPPO) at cumulative Reward {}!", cumu_reward);
    Stopwatch sw;
    sw.start();

    double last_val_pred = IPPO_buffer->values.back();
    auto [advantages, returns] = IPPO_buffer->compute_gae(gamma, lambda, last_val_pred);

    auto states = torch::stack(IPPO_buffer->states).to(precision);
    auto timeouts = torch::tensor(IPPO_buffer->timeouts, torch::kInt64).to(model->parameters()[0].device());
    auto batchings = torch::tensor(IPPO_buffer->batchings, torch::kInt64).to(model->parameters()[0].device());
    auto scalings = torch::tensor(IPPO_buffer->scalings, torch::kInt64).to(model->parameters()[0].device());
    auto old_log_probs = torch::tensor(IPPO_buffer->log_probs, precision).to(model->parameters()[0].device());

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
    advantages = advantages.to(model->parameters()[0].device());
    returns = returns.to(model->parameters()[0].device());

    int n_epochs = 4;      // Standard IPPO epochs
    int batch_size = 64;   // Minibatch size
    int64_t dataset_size = states.size(0);

    T total_loss = torch::tensor(0.0);
    T total_policy_loss = torch::tensor(0.0);
    T total_value_loss = torch::tensor(0.0);

    model->train();

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        auto rand_indices = torch::randperm(dataset_size, torch::kInt64);

        for (int i = 0; i < dataset_size; i += batch_size) {
            int end = std::min((int)dataset_size, i + batch_size);
            auto idx = rand_indices.slice(0, i, end);

            // Minibatch Data
            auto mb_states = states.index_select(0, idx);
            auto mb_timeouts = timeouts.index_select(0, idx);
            auto mb_batchings = batchings.index_select(0, idx);
            auto mb_scalings = scalings.index_select(0, idx);
            auto mb_old_log_probs = old_log_probs.index_select(0, idx);
            auto mb_advantages = advantages.index_select(0, idx);
            auto mb_returns = returns.index_select(0, idx);
            auto mb_old_values = torch::tensor(IPPO_buffer->values).to(precision).to(model->parameters()[0].device()).index_select(0, idx);

            // Forward Pass
            auto [policy1, policy2, policy3, val] = model->forward(mb_states);

            auto p1 = torch::softmax(policy1, -1);
            auto log_p1 = torch::log(p1.gather(-1, mb_batchings.unsqueeze(-1)).squeeze(-1) + 1e-10);

            auto p2 = torch::softmax(policy2, -1);
            auto log_p2 = torch::log(p2.gather(-1, mb_timeouts.unsqueeze(-1)).squeeze(-1) + 1e-10);

            auto p3 = torch::softmax(policy3, -1);
            auto log_p3 = torch::log(p3.gather(-1, mb_scalings.unsqueeze(-1)).squeeze(-1) + 1e-10);

            auto new_log_probs = log_p1 + log_p2 + log_p3;

            // Entropy
            auto ent1 = -(p1 * torch::log(p1 + 1e-10)).sum(-1).mean();
            auto ent2 = -(p2 * torch::log(p2 + 1e-10)).sum(-1).mean();
            auto ent3 = -(p3 * torch::log(p3 + 1e-10)).sum(-1).mean();
            auto entropy = ent1 + ent2 + ent3;

            // Ratio
            auto ratio = torch::exp(new_log_probs - mb_old_log_probs);

            auto surr1 = ratio * mb_advantages;
            auto surr2 = torch::clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages;
            auto policy_loss = -torch::min(surr1, surr2).mean();

            auto val_pred = val.squeeze(-1);
            auto v_clipped = mb_old_values + torch::clamp(val_pred - mb_old_values, -clip_epsilon, clip_epsilon);
            auto v_loss_1 = torch::pow(val_pred - mb_returns, 2);
            auto v_loss_2 = torch::pow(v_clipped - mb_returns, 2);
            auto value_loss = 0.5 * torch::max(v_loss_1, v_loss_2).mean();

            // Total Loss
            // Coefficients matching ippon defaults
            double ent_coef = 0.01;
            double vf_coef = 0.8;

            auto loss = policy_loss + vf_coef * value_loss - ent_coef * entropy;

            std::unique_lock<std::mutex> lock(model_mutex);
            optimizer->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5);
            optimizer->step();

            total_loss = loss;
            total_policy_loss = policy_loss;
            total_value_loss = value_loss;
        }
    }

    sw.stop();

    double avg_reward = cumu_reward / (double) update_steps;
    out << "episodeEnd," << sw.elapsed_microseconds() << "," << 0 << "," << steps_counter
        << "," << cumu_reward << "," << avg_reward
        << "," << total_loss.item<double>()
        << "," << total_policy_loss.item<double>()
        << "," << total_value_loss.item<double>() << std::endl;

    if (last_avg_reward < avg_reward) {
        last_avg_reward = avg_reward;
        torch::save(model, path + "/latest_model.pt");
    }

    IPPO_buffer->clear();
    reset();
}

void FCPOAgent::rewardCallback(double throughput, double latency, double oversize, double memory_use) {
    if (update_steps > 0 ) {
        if (first) {
            first = false;
            return;
        }
        double reward = theta * throughput - sigma * latency - phi * oversize
                        - rho * (memory_use / (double) (batchInferProfileList[max_batch].gpuMemUsage + 1.0));

        spdlog::get("container_agent")->trace("RL Agent Reward: {}, Penalties: Lat {}, Size {}, Mem {}",
                                              reward, latency, oversize, memory_use);

        reward = std::max(-1.0, std::min(1.0, reward));
        if (penalty) reward *= (reward < 0) ? 1.1 : -0.9;

        if (IPPO_buffer->size() > 0) {
            IPPO_buffer->rewards.back() = reward;
            cumu_reward += reward;
        }
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

    T action_dist = torch::multinomial(torch::softmax(policy1, -1), 1);
    batching = action_dist.item<int>();

    action_dist = torch::multinomial(torch::softmax(policy2, -1), 1);
    timeout = action_dist.item<int>();

    action_dist = torch::multinomial(torch::softmax(policy3, -1), 1);
    scaling = action_dist.item<int>();

    // Validity check
    if (batching > 0 && batchInferProfileList[batching].p95inferLat > last_slo) {
        penalty = true;
        batching = std::max(last_batching, base_batch);
    }
    last_batching = batching;

    if (update_steps > 0 ) {
        auto l1 = torch::log(torch::softmax(policy1, -1)[batching] + 1e-10);
        auto l2 = torch::log(torch::softmax(policy2, -1)[timeout] + 1e-10);
        auto l3 = torch::log(torch::softmax(policy3, -1)[scaling] + 1e-10);

        double current_log_prob = (l1 + l2 + l3).item<double>();
        double current_val = val.item<double>();

        // Add to IPPO Buffer
        IPPO_buffer->add(state.clone(), timeout, batching, scaling, current_log_prob, current_val, 0.0, false);
    }
}

// Deprecated in favor of IPPOBuffer, keeping for compatibility signature
T FCPOAgent::computeCumuRewards() const {
    return torch::zeros({1});
}
T FCPOAgent::computeGae() const {
    return torch::zeros({1});
}

std::tuple<int, int, int> FCPOAgent::runStep() {
    Stopwatch sw;
    sw.start();
    selectAction();

    if (update_steps > 0 ) steps_counter++;
    sw.stop();

    if (update_steps > 0 ) {
        if (steps_counter % update_steps == 0) {
            update();
        }
    }
    return std::make_tuple((timeout + 1) % (timeout_size), batching + 1, scaling);
}
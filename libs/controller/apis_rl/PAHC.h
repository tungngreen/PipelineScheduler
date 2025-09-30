#include <local_optimization.h>

#ifndef PIPEPLUSPLUS_PAHC_H
#define PIPEPLUSPLUS_PAHC_H

class PAHC_ExperienceBuffer : public ExperienceBuffer {
public:
    PAHC_ExperienceBuffer() = default;

    PAHC_ExperienceBuffer(size_t capacity)
            : ExperienceBuffer(capacity), weights(capacity) {}

    void add(const T& state, const T& log_prob, const T& value, const T& weight) {
        if (is_full) {
            double distance = distance_metric(state, states[current_index]);

            if (distance > 0.5) {
                spdlog::trace("Distance: {}", distance);
                //set current index to the index of the oldest timestamp
                current_index = std::distance(timestamps.begin(),
                                              std::min_element(timestamps.begin(), timestamps.end()));
                await_reward = true;
            } else {
                return;
            }
        }
        timestamps[current_index] = std::chrono::system_clock::now();
        states[current_index] = state;

        log_probs[current_index] = log_prob;
        values[current_index] = value;
        weights[current_index] = weight;
        valid_history = false;
    }

    [[nodiscard]] std::vector<T> get_weights() const {
        if (is_full) return weights;
        return {weights.begin(), weights.begin() + current_index};
    }

    void clear_partially() { // remove first 20% of experiences
        current_index = capacity * 0.8;
        std::rotate(timestamps.begin(), timestamps.begin() + capacity * 0.2, timestamps.end());
        std::rotate(states.begin(), states.begin() + capacity * 0.2, states.end());
        std::rotate(log_probs.begin(), log_probs.begin() + capacity * 0.2, log_probs.end());
        std::rotate(values.begin(), values.begin() + capacity * 0.2, values.end());
        std::rotate(weights.begin(), weights.begin() + capacity * 0.2, weights.end());
        valid_history = false;
        is_full = false;
    }

private:
    std::vector<T> weights;
};


class PAHC_Actor : public torch::nn::Module {
public:
    PAHC_Actor(int state_dim, int action_dim) :
            fc1(torch::nn::LinearOptions(state_dim, 128)),
            fc2(torch::nn::LinearOptions(128, 128)),
            fc3(torch::nn::LinearOptions(128, 128)),
            fc_mean(torch::nn::LinearOptions(128, action_dim)),
            fc_log_std(torch::nn::LinearOptions(128, action_dim))
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("fc_mean", fc_mean);
        register_module("fc_log_std", fc_log_std);
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state) {
        auto x = torch::relu(fc1(state));
        x = torch::relu(fc2(x));
        x = torch::relu(fc3(x));

        auto mean = fc_mean(x);
        // Clamp log_std to a reasonable range for stability (e.g., [-2, 2])
        auto log_std = torch::clamp(fc_log_std(x), -2.0, 2.0);

        return {mean, log_std};
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::Linear fc_mean{nullptr}, fc_log_std{nullptr};
};


class PAHC_Critic : public torch::nn::Module {
public:
    PAHC_Critic(int state_dim, int action_dim) :
            fc1(torch::nn::LinearOptions(state_dim + action_dim, 128)),
            fc2(torch::nn::LinearOptions(128, 128)),
            fc3(torch::nn::LinearOptions(128, 64)),
            // Output size is 1 (the scalar Q-value for a given objective)
            fc_out(torch::nn::LinearOptions(64, 1))
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("fc_out", fc_out);
    }

    // Takes state S and utility weights W, outputs Q-value
    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = torch::cat({state, action}, /*dim=*/1);
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = torch::relu(fc3(x));
        return fc_out(x);
    }

    std::shared_ptr<PAHC_Critic> clone() const {
        auto cloned = std::make_shared<PAHC_Critic>(
                fc1->options.in_features() - fc2->options.in_features(), // state_dim
                fc1->options.in_features() - (fc1->options.in_features() - fc2->options.in_features()) // action_dim
        );
        torch::NoGradGuard no_grad;
        cloned->parameters() = this->parameters();
        return cloned;
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc_out{nullptr};
};


class PAHC {
public:
    PAHC(const std::string& exp_name, uint state_size, uint weights_size, torch::Dtype precision, uint update_steps,
         int seed = 42);
    ~PAHC() {
        torch::save(actor, path + "/latest_actor.pt");
        torch::save(critic1, path + "/latest_critic1.pt");
        torch::save(critic2, path + "/latest_critic2.pt");
        out.close();
    }

    std::vector<float> runStep();
    void rewardCallback(double throughput, double latency, double memory_use);
    void setState(double agentID);
private:
    T selectAction();
    void update();
    void soft_update_targets();

    void reset() {
        cumu_reward = 0.0;
        first = true;
        new_states.clear();
        experiences.clear_partially();
    }

    std::shared_ptr<PAHC_Actor> actor;
    std::shared_ptr<PAHC_Critic> critic1, critic2, critic1_target, critic2_target;
    std::unique_ptr<torch::optim::Optimizer> optimizer_actor, optimizer_critic1, optimizer_critic2, optimizer_alpha;
    torch::Dtype precision;

    T state, log_prob, value, log_alpha_meta;
    std::vector<T> new_states;
    std::vector<float> weights;
    PAHC_ExperienceBuffer experiences;

    std::ofstream out;
    std::string path;

    uint state_size;
    uint weights_size;

    bool first = true;
    uint steps_counter = 0;
    uint update_steps;
    double cumu_reward;
};

#endif //PIPEPLUSPLUS_PAHC_H

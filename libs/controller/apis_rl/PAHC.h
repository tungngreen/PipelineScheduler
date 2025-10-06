#include <local_optimization.h>

#ifndef PIPEPLUSPLUS_PAHC_H
#define PIPEPLUSPLUS_PAHC_H

class PAHC_ExperienceBuffer : public ExperienceBuffer {
public:
    PAHC_ExperienceBuffer() = default;

    PAHC_ExperienceBuffer(size_t capacity)
            : ExperienceBuffer(capacity), weights(capacity), log_stds(capacity), means(capacity) {}

    virtual void add(const T& state, const T& log_std, const T& mean, const T& weight) final {
        timestamps[current_index] = std::chrono::system_clock::now();
        states[current_index] = state;
        log_stds[current_index] = log_std;
        means[current_index] = mean;
        weights[current_index] = weight;

        current_index = (current_index + 1) % capacity;
    }

    virtual void add_reward(const double x) final {
        if (reward_index == capacity) return;
        rewards[reward_index] = x;
        reward_index = (reward_index + 1) % capacity;
        if (reward_index == 0) is_full = true;
    }

    [[nodiscard]] virtual std::vector<double> get_rewards() const final {
        if (is_full) return rewards;
        return {rewards.begin(), rewards.begin() + reward_index - 1};
    }

    [[nodiscard]] std::vector<T> get_weights() const {
        if (is_full) return weights;
        return {weights.begin(), weights.begin() + current_index - 1};
    }

    std::tuple<T, T, T, T, T> sample(uint n) {
        std::vector<int> indices;
        if (is_full)
            indices =  std::vector<int>(capacity);
        else {
            if (n > reward_index - 1) n = reward_index - 1;
            indices = std::vector<int>(reward_index - 1);
        }
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
        indices.resize(n);
        std::sort(indices.begin(), indices.end());
        std::vector<T> sampled_states, sampled_actions, sampled_log_stds, sampled_means;
        std::vector<double> sampled_rewards;

        for (auto i : indices) {
            if (states[i].size(0) == 0 || log_stds[i].size(0) == 0 || weights[i].size(0) == 0 || means[i].size(0) == 0) {
                spdlog::get("container_agent")->trace("Skipping empty experience at index {}", i);
                continue;
            }
            sampled_states.push_back(states[i]);
            sampled_actions.push_back(weights[i]);
            sampled_log_stds.push_back(log_stds[i]);
            sampled_means.push_back(means[i]);
            sampled_rewards.push_back(rewards[i]);
        }

        return std::make_tuple(torch::stack(sampled_states), torch::stack(sampled_actions),
                               torch::stack(sampled_log_stds), torch::stack(sampled_means), torch::tensor(sampled_rewards));
    };

    virtual void clear() final {
        reward_index = 0;
        current_index = 0;
        is_full = false;
    }
private:
    int reward_index;
    std::vector<T> weights, log_stds, means;
};


class PAHC_Actor : public torch::nn::Module {
public:
    PAHC_Actor(int state_dim, int action_dim) :
            fc1(torch::nn::LinearOptions(state_dim, 128)),
            fc2(torch::nn::LinearOptions(128, 128)),
            fc3(torch::nn::LinearOptions(128, 64)),
            fc_mean(torch::nn::LinearOptions(64, action_dim)),
            fc_log_std(torch::nn::LinearOptions(64, action_dim))
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

    void finish_step() {
        first = false;
        try {
            update();
        } catch (const c10::Error& e) {
            spdlog::get("container_agent")->error("Error during PAHC update: {0}", e.what());
            reset();
            experiences.clear();
        }
    }
    std::vector<float> runStep();
    void rewardCallback(double throughput, double latency, double memory_use);
    void setState(double agentID, double agentType, double pipeType, double pipeLatency, double localLatency,
                  double theta, double pipeThroughput, double sigma, double memoryUse, double phi, double rho);
private:
    T selectAction();
    void update();
    void soft_update_targets();

    void reset() {
        steps_counter = 0;
        cumu_reward = 0.0;
        first = true;
        new_states.clear();
        experiences.clear();
    }

    std::shared_ptr<PAHC_Actor> actor;
    std::shared_ptr<PAHC_Critic> critic1, critic2, critic1_target, critic2_target;
    std::unique_ptr<torch::optim::Optimizer> optimizer_actor, optimizer_critic1, optimizer_critic2, optimizer_alpha;
    torch::Dtype precision;

    T state, value, log_alpha_meta;
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

#include "edgevision.h"

EdgeVisionAgent::EdgeVisionAgent(std::string& dev_name, int offloading_candidates, torch::Dtype precision,
                         uint update_steps, double lambda, double gamma, double clip, double omega, double sigma, int F)
        : precision(precision), dev_name(dev_name), lambda(lambda), gamma(gamma), clip(clip),
          omega(omega), sigma(sigma), F(F), update_steps(update_steps) {
    path = "../../pipe/models/edgevision" ;
    std::filesystem::create_directories(std::filesystem::path(path));

    actor = std::make_shared<EdgeViActorNet>(offloading_candidates+3, offloading_candidates);
    critic = std::make_shared<EdgeViCriticNet>(5);
    std::string model_save = path + "/latest_actor.pt";
    if (std::filesystem::exists(model_save)) torch::load(actor, model_save);
    model_save = path + "/latest_critic.pt";
    if (std::filesystem::exists(model_save)) torch::load(critic, model_save);
    actor->to(precision);
    actor_optimizer = std::make_unique<torch::optim::Adam>(actor->parameters(), torch::optim::AdamOptions(0.0005));
    critic_optimizer = std::make_unique<torch::optim::Adam>(critic->parameters(), torch::optim::AdamOptions(0.0005));

    states = {};
    targeted_devices = {};
    rewards = {};
    log_probs = {};
}

void EdgeVisionAgent::update() {
    std::unique_lock<std::mutex> lock(model_mutex);
    spdlog::get("container_agent")->info("Locally training RL agent!");

    auto obs_tensor = torch::stack(states);
    auto actions_tensor = torch::tensor(targeted_devices, precision);

    values = critic->forward(obs_tensor).squeeze();
    auto advantages = computeGae();

    auto advantages_tensor = torch::stack(advantages).detach();
    auto returns_tensor = advantages_tensor + values;

    auto new_dist = actor->forward(obs_tensor);
    auto new_log_probs = torch::log(torch::gather(new_dist, 1, actions_tensor.unsqueeze(-1)).squeeze());
    auto ratio = torch::exp(new_log_probs - torch::stack(log_probs));

    auto surr1 = ratio * advantages_tensor;
    auto surr2 = torch::clamp(ratio, 1 - clip, 1 + clip) * advantages_tensor;

    auto entropy = -(new_dist * new_log_probs).sum(-1).mean();
    auto actor_loss = -torch::mean(torch::min(surr1, surr2)) - sigma * entropy;

    auto critic_pred = critic->forward(obs_tensor).squeeze();
    auto clipped_value = torch::clamp(critic_pred, critic_pred - clip, critic_pred + clip);
    auto critic_loss = torch::mean(torch::max(
            (critic_pred - returns_tensor).pow(2),
            (clipped_value - returns_tensor).pow(2)));

    actor_optimizer->zero_grad();
    actor_loss.backward();
    actor_optimizer->step();

    critic_optimizer->zero_grad();
    critic_loss.backward();
    critic_optimizer->step();

    steps_counter = 0;
    reset();
}

void EdgeVisionAgent::rewardCallback(int throughput, int drops, double avg_latency, double accuracy) {
    if (update_steps > 0 ) {
        double tmp_reward = drops * (-omega * F) + throughput * (accuracy - omega * avg_latency * 1000);
        if (std::isnan(tmp_reward)) tmp_reward = 0.0;
        rewards.push_back(tmp_reward);
    }
}

void EdgeVisionAgent::setState(double req_arrival_rate, int queueLength, int offloadingQueue, std::vector<double> bandwidths) {
    std::vector<double> tmp = {req_arrival_rate, static_cast<double>(queueLength), static_cast<double>(offloadingQueue)};
    tmp.insert(tmp.end(), bandwidths.begin(), bandwidths.end());
    state = torch::tensor(tmp, precision);
    if (torch::any(torch::isnan(state)).item<bool>()) {
        state = torch::nan_to_num(state);
    }
}

T EdgeVisionAgent::computeGae() const {
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

void EdgeVisionAgent::selectAction() {
    std::unique_lock<std::mutex> lock(model_mutex);
    T probs = actor->forward(state);
    T action = torch::multinomial(probs, 1);
    log_prob = torch::log(probs.gather(-1, action));
    target_device = action.item<int>();

    if (update_steps > 0 ) {
        states.push_back(state);
        log_probs.push_back(log_prob);
        targeted_devices.push_back(target_device);
    }
}

int EdgeVisionAgent::runStep() {
    selectAction();

    if (update_steps > 0 ) {
        steps_counter++;
        if (steps_counter % update_steps == 0) {
            update();
        }
    }
    return target_device;
}
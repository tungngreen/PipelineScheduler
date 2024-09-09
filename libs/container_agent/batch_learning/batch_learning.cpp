#include "batch_learning.h"

PPO::PPO(ActorCritic& ac, uint update_steps, uint mini_batch_size, uint epochs, double lambda, double gamma)
    : ac(ac), update_steps(update_steps), mini_batch_size(mini_batch_size), epochs(epochs), lambda(lambda), gamma(gamma) {
    std::filesystem::create_directories(std::filesystem::path("../models/batch_learning"));
    out.open("../models/batch_learning/latest_log.csv");

    std::random_device rd;
    re = std::mt19937(rd());
    torch::optim::Adam tmp_opt = torch::optim::Adam(ac->parameters(), 1e-3);
    opt = &tmp_opt;
    avg_reward = 0.0;
}

VT PPO::returns() {
    torch::Tensor gae = torch::zeros({1}, torch::kFloat64);
    VT returns(rewards.size(), torch::zeros({1}, torch::kFloat64));

    for (uint i=rewards.size();i-- >0;) { // inverse for loops over unsigned: https://stackoverflow.com/questions/665745/whats-the-best-way-to-do-a-reverse-for-loop-with-an-unsigned-index/665773
        auto delta = rewards[i] + gamma*values[i+1]*(1-dones[i]) - values[i];
        gae = delta + gamma*lambda*(1-dones[i])*gae;
        returns[i] = gae + values[i];
    }
    return returns;
}

void PPO::update(double beta, double clip_param) {
    values.push_back(std::get<1>(ac->forward(states[counter-1])));
    return_vals = PPO::returns();
    torch::Tensor t_log_probs = torch::cat(log_probs).detach();
    torch::Tensor t_return_vals = torch::cat(return_vals).detach();
    torch::Tensor t_values = torch::cat(values).detach();
    torch::Tensor t_states = torch::cat(states);
    torch::Tensor t_actions = torch::cat(actions);
    torch::Tensor advantages = t_return_vals - t_values.slice(0, 0, update_steps);

    for (uint e=0;e<epochs;e++) {
        // Generate random indices.
        torch::Tensor cpy_sta = torch::zeros({mini_batch_size, t_states.size(1)}, t_states.options());
        torch::Tensor cpy_act = torch::zeros({mini_batch_size, t_actions.size(1)}, t_actions.options());
        torch::Tensor cpy_log = torch::zeros({mini_batch_size, t_log_probs.size(1)}, t_log_probs.options());
        torch::Tensor cpy_ret = torch::zeros({mini_batch_size, t_return_vals.size(1)}, t_return_vals.options());
        torch::Tensor cpy_adv = torch::zeros({mini_batch_size, advantages.size(1)}, advantages.options());

        for (uint b=0;b<mini_batch_size;b++) {
            uint idx = std::uniform_int_distribution<uint>(0, update_steps-1)(re);
            cpy_sta[b] = t_states[idx];
            cpy_act[b] = t_actions[idx];
            cpy_log[b] = t_log_probs[idx];
            cpy_ret[b] = t_return_vals[idx];
            cpy_adv[b] = advantages[idx];
        }

        auto av = ac->forward(cpy_sta); // action value pairs
        auto action = std::get<0>(av);
        auto entropy = ac->entropy().mean();
        auto new_log_prob = ac->log_prob(cpy_act);

        auto ratio = (new_log_prob - cpy_log).exp();
        auto surr1 = ratio*cpy_adv;
        auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param)*cpy_adv;

        auto val = std::get<1>(av);
        auto actor_loss = -torch::min(surr1, surr2).mean();
        auto critic_loss = (cpy_ret-val).pow(2).mean();
        auto loss = 0.5*critic_loss+actor_loss-beta*entropy;

        opt->zero_grad();
        loss.backward();
        opt->step();
    }

    states.clear();
    actions.clear();
    rewards.clear();
    dones.clear();
    log_probs.clear();
    return_vals.clear();
    values.clear();
}

void PPO::rewardCallback(double throughput, double drops, double oversize_penalty) {
    rewards.push_back(torch::tensor({throughput - 0.5 * (drops + oversize_penalty)}, torch::kF64));
}

torch::Tensor PPO::buildState(int curr_batch, int arrival, int target_latency, int pre_queue_size, int inf_queue_size, int mem) {
    return torch::tensor({{curr_batch, arrival, target_latency, pre_queue_size, inf_queue_size, mem}}, torch::kF64);
}

double PPO::runStep(torch::Tensor& state) {
    states.push_back(state);

    auto av = ac->forward(states[counter]);
    actions.push_back(std::get<0>(av));
    values.push_back(std::get<1>(av));
    log_probs.push_back(ac->log_prob(actions[counter]));

    double act = actions[counter][0][0].item<double>();

    dones.push_back(torch::zeros({1, 1}, torch::kF64));
    avg_reward += rewards[counter][0][0].item<double>()/update_steps;
    counter++;

    if (counter%update_steps == 0) {
        printf("Updating the network.\n");
        std::thread t(&PPO::update, this, 1e-3, 0.2);
        t.detach();
        out << avg_reward << std::endl;
        counter = 0;
        avg_reward = 0.0;
    }

    return act;
}
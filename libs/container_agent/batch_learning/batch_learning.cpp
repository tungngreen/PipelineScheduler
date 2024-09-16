#include "batch_learning.h"

PPO::PPO(std::string& cont_name, ActorCritic ac, uint max_batch, uint update_steps, uint mini_batch_size, uint epochs,
         double lambda, double gamma) : ac(ac), max_batch(max_batch), update_steps(update_steps),
                                        mini_batch_size(mini_batch_size), epochs(epochs), lambda(lambda), gamma(gamma) {
    path = "../models/batch_learning/" + cont_name;
    std::filesystem::create_directories(std::filesystem::path(path));
    out.open(path + "/latest_log.csv");

    std::random_device rd;
    re = std::mt19937(rd());
    opt = new torch::optim::Adam(ac->parameters(), 1e-3);
    avg_reward = 0.0;
}

VT PPO::returns() {
    torch::Tensor gae = torch::zeros({1}, torch::kFloat64);
    VT returns(rewards.size(), torch::zeros({1}, torch::kFloat64));

    for (uint i=rewards.size();i-- >0;) { // inverse for loops over unsigned: https://stackoverflow.com/questions/665745/whats-the-best-way-to-do-a-reverse-for-loop-with-an-unsigned-index/665773
        auto delta = rewards[i] + gamma*values[i+1] - values[i];
        gae = delta + gamma*lambda*gae;
        returns[i] = gae + values[i];
    }
    return returns;
}

void PPO::update(double beta, double clip_param) {
    spdlog::info("Updating reinforcement learning batch size model!");
    Stopwatch sw;
    sw.start();
    values.push_back(std::get<1>(ac->forward(states[counter-1])));
    torch::Tensor t_log_probs = torch::cat(log_probabilities).detach().slice(0, 0, update_steps);
    torch::Tensor return_values = torch::cat(PPO::returns()).detach().slice(0, 0, update_steps);
    torch::Tensor t_states = torch::cat(states).slice(0, 0, update_steps);
    torch::Tensor t_actions = torch::cat(actions).slice(0, 0, update_steps);
    torch::Tensor advantages = return_values - torch::cat(values).detach().slice(0, 0, update_steps);

    for (uint e=0;e<epochs;e++) {
        // Generate random indices.
        torch::Tensor cpy_sta = torch::zeros({mini_batch_size, t_states.size(1)}, t_states.options());
        torch::Tensor cpy_act = torch::zeros({mini_batch_size, t_actions.size(1)}, t_actions.options());
        torch::Tensor cpy_log = torch::zeros({mini_batch_size, t_log_probs.size(1)}, t_log_probs.options());
        torch::Tensor cpy_ret = torch::zeros({mini_batch_size, return_values.size(1)}, return_values.options());
        torch::Tensor cpy_adv = torch::zeros({mini_batch_size, advantages.size(1)}, advantages.options());

        for (uint b=0;b<mini_batch_size;b++) {
            uint idx = std::uniform_int_distribution<uint>(0, update_steps-1)(re);
            cpy_sta[b] = t_states[idx];
            cpy_act[b] = t_actions[idx];
            cpy_log[b] = t_log_probs[idx];
            cpy_ret[b] = return_values[idx];
            cpy_adv[b] = advantages[idx];
        }

        std::cout << "cpy_sta: " << cpy_sta << std::endl;
        std::tuple<torch::Tensor, torch::Tensor> av = ac->forward(cpy_sta); // action value pairs
        torch::Tensor action = std::get<0>(av);
        torch::Tensor entropy = ac->entropy().mean();
        torch::Tensor new_log_prob = ac->log_prob(cpy_act);

        torch::Tensor ratio = (new_log_prob - cpy_log).exp();
        torch::Tensor surr1 = ratio*cpy_adv;
        torch::Tensor surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param)*cpy_adv;

        torch::Tensor val = std::get<1>(av);
        torch::Tensor actor_loss = -torch::min(surr1, surr2).mean();
        torch::Tensor critic_loss = (cpy_ret-val).pow(2).mean();
        torch::Tensor loss = 0.5*critic_loss+actor_loss-beta*entropy;

        opt->zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(ac->parameters(), 1.0); // Add gradient clipping
        opt->step();
    }
    sw.stop();
    std::cout << "Training: " << sw.elapsed_microseconds() << "," << counter << "," << avg_reward << std::endl;


    counter = 0;
    avg_reward = 0.0;
    states.clear();
    actions.clear();
    rewards.clear();
    log_probabilities.clear();
    values.clear();
}

void PPO::rewardCallback(double throughput, double drops, double oversize_penalty) {
    rewards.push_back(torch::tensor({throughput - drops + (1 - oversize_penalty) + 1e-8}, torch::kF64)); // Add a small epsilon to avoid zero rewards
}

void PPO::setState(double curr_batch, double arrival, double pre_queue_size, double inf_queue_size) {
    states.push_back(torch::tensor({{curr_batch / max_batch, arrival, pre_queue_size, inf_queue_size}}, torch::kF64));
}

int PPO::runStep() {
    Stopwatch sw;
    sw.start();
    auto av = ac->forward(states[counter]);
    actions.push_back(std::get<0>(av));
    values.push_back(std::get<1>(av));
    log_probabilities.push_back(ac->log_prob(actions[counter]));
    avg_reward += rewards[counter].item<double>()/update_steps;

    unsigned int action = std::max(1, (int)std::floor(actions[counter][0][0].item<double>() * max_batch));
    counter++;
    sw.stop();
    out << sw.elapsed_microseconds() << "," << counter << "," << avg_reward << "," << action << std::endl;

    if (counter%update_steps == 0) {
        std::thread t(&PPO::update, this, 1e-3, 0.2);
        t.detach();
    }

    return std::min(action, max_batch);
}
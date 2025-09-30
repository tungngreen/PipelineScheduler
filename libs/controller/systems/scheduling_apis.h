#include "controller.h"
#include "PAHC.h"

struct LocalCRLAgentHandle {
    int id;
    std::string name;
    std::vector<float> weights;

    void updateUtilityWeights(std::vector<float> updates) {
        for (size_t i = 0; i < weights.size() && i < updates.size(); i++) {
            weights[i] += updates[i];
            if (weights[i] < 0) weights[i] = 0;
        }
    }
};

std::map<std::string, LocalCRLAgentHandle> localCRLAgents;

PAHC *ApisRL;

#include "cardamage.h"

int main(int argc, char **argv) {

    json configs = loadRunArgs(argc, argv);

    ContainerAgent *agent;

    json pipeConfigs = configs["container"]["cont_pipeline"];

    agent = new CarDamageAgent(configs);

    agent->runService(pipeConfigs, configs);

    delete agent;
    return 0;
}
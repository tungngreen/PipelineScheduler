#include "bcedge.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, algorithm, "bcedge", "model you want to train");
ABSL_FLAG(uint16_t, epochs, 500, "number of epochs to train");
ABSL_FLAG(uint16_t, steps, 512, "number of steps per epoch");

int main(int argc, char **argv)
{
    absl::ParseCommandLine(argc, argv);

    std::string algorithm = absl::GetFlag(FLAGS_algorithm);
    uint16_t epochs = absl::GetFlag(FLAGS_epochs);
    uint16_t steps = absl::GetFlag(FLAGS_steps);

    if (algorithm == "bcedge")
    {
        BCEdgeAgent *bcedge = new BCEdgeAgent(algorithm, torch::kF32, steps);
        for (int i = 0; i < epochs; i++)
        {
            for (int j = 0; j < steps; j++)
            {
                bcedge->setState(Yolov5n, {1,1,1}, 1);
                bcedge->runStep();
                bcedge->rewardCallback(1.0, 0.5, 1, 0.5);
            }
        }
    }
    else
    {
        std::cerr << "Invalid algorithm: " << algorithm << std::endl;
        return 1;
    }

    return 0;
}
#include "controller.h"

struct Partitioner
{
    // NodeHandle& edge;
    // NodeHandle& server;
    // need server here
    float BaseParPoint;
    float FineGrainedOffset;
};

namespace Dis {
    double calculateTotalprocessedRate(Devices &nodes, Tasks &pipelines, bool is_edge);
    int calculateTotalQueue(Devices &nodes, Tasks &pipelines, bool is_edge);
    void scheduleBaseParPointLoop(Partitioner *partitioner, Devices &nodes, Tasks &pipelines);
    void scheduleFineGrainedParPointLoop(Partitioner *partitioner, Devices &nodes, Tasks &pipelines);
    void DecideAndMoveContainer(Devices &nodes, Tasks &pipelines, Partitioner *partitioner, int cuda_device);
}
#include "controller.h"
#include "controlmessages.pb.h"
#include <cstdint>
#include <memory>
#include <vector>

ABSL_FLAG(std::string, ctrl_configPath, "../jsons/experiments/base-experiment.json",
          "Path to the configuration file for this experiment.");
ABSL_FLAG(uint16_t, ctrl_verbose, 0, "Verbosity level of the controller.");
ABSL_FLAG(uint16_t, ctrl_loggingMode, 0, "Logging mode of the controller. 0:stdout, 1:file, 2:both");
ABSL_FLAG(std::string, ctrl_logPath, "../logs", "Path to the log dir for the controller.");


// =========================================================GPU Lanes/Portions Control===========================================================
// ==============================================================================================================================================
// ==============================================================================================================================================
// ==============================================================================================================================================

void Controller::initiateGPULanes(NodeHandle &node) {
    // Currently only support powerful GPUs capable of running multiple models in parallel
    if (node.name == "sink") {
        return;
    }
    auto deviceList = devices.getMap();

    if (deviceList.find(node.name) == deviceList.end()) {
        spdlog::get("container_agent")->error("Device {0:s} is not found in the device list", node.name);
        return;
    }

    // TODO: Check if we can remove this if
    if (node.type == Server || node.type == Virtual) {
        node.numGPULanes = NUM_LANES_PER_GPU * NUM_GPUS;
    } else {
        node.numGPULanes = 1;
    }

    node.gpuLanes.clear();
    node.freeGPUPortions.list.clear();
    node.freeGPUPortions.head = nullptr;

    for (unsigned short i = 0; i < node.numGPULanes; i++) {
        GPUHandle *targetGPU = node.gpuHandles[i / NUM_LANES_PER_GPU].get();

        // Create the NUM_LANES_PER_GPU-th lane for the GPU
        auto gpuLane = std::make_unique<GPULane>(targetGPU, &node, i);
        GPULane *gpuLanePtr = gpuLane.get();

        // This is currently the only portion in a lane, later when it is divided
        // we need to keep track of the portions in the lane to be able to recover the free portions
        // when the container is removed.
        auto portion = std::make_unique<GPUPortion>(gpuLanePtr);
        GPUPortion *portionPtr = portion.get();

        gpuLanePtr->portionList.head = portionPtr;
        // The Lane strictly owns the portion via unique_ptr
        gpuLanePtr->portionList.list.push_back(std::move(portion));

        portionPtr->nextInLane = nullptr;
        portionPtr->prevInLane = nullptr;

        // TODO: HANDLE FREE PORTIONS WITHIN THE GPU
        // gpuLane->gpuHandle->freeGPUPortions.push_back(portion);

        // Each lane now has a bigass portion that covers the whole GPU time, 
        // We chain all the portions in the same GPU together in a linked list, so that when we traverse the free portions for scheduling, 
        // we can easily find the next portion in the same GPU when we need to split a portion for a container.
        if (i == 0) {
            node.freeGPUPortions.head = portionPtr;
            portionPtr->prev = nullptr;
        } else {
            GPUPortion* prevGlobal = node.freeGPUPortions.list.back();
            prevGlobal->next = portionPtr;
            portionPtr->prev = prevGlobal;
        }
        portionPtr->next = nullptr;

        // Add to global free portion list of the node
        node.freeGPUPortions.list.push_back(portionPtr);

        // Move the ownership of the lane to the node
        node.gpuLanes.push_back(std::move(gpuLane));
    }
}

bool GPUPortion::assignContainer(std::shared_ptr<ContainerHandle> cont) {
    // 1. Check if this portion is already occupied
    // .lock() returns a shared_ptr if valid, or nullptr if empty/expired
    if (auto existing = this->container.lock()) {
        spdlog::get("console")->error("Portion already assigned to container {}", existing->name);
        return false;
    }

    if (!cont) {
        spdlog::get("console")->error("Cannot assign null container to portion");
        return false;
    }

    // Container points to Portion, Portion observes Container
    cont->executionPortion = this;
    this->container = cont; // Automatic conversion from shared_ptr to weak_ptr
   
    start = cont->startTime;
    end = cont->endTime;

    spdlog::get("container_agent")->info("Portion assigned to container {0:s}", cont->name);
    return true;
}

/**
 * @brief insert the newly created free portion into the sorted list of free portions
 * Since the list is sorted, we can insert the new portion by traversing the list from the head
 * Complexity: O(n)
 *
 * @param head
 * @param freePortion
 */
void Controller::insertFreeGPUPortion(GPUPortionList &portionList, GPUPortion *freePortion) {
    if (freePortion == nullptr) {
        spdlog::get("container_agent")->error("Cannot insert null portion into the list of free portions");
        return;
    }

    // Initialize the new free portion's next and prev pointers to null before inserting it into the list
    freePortion->next = nullptr;
    freePortion->prev = nullptr;

    portionList.list.push_back(freePortion);

    auto &head = portionList.head;

    // Case 1: If the list is empty, the new portion becomes the head
    if (head == nullptr) {
        head = freePortion;
        return;
    }

    // Traverse the list to find the correct position to insert the new portion
    uint64_t freeSize = freePortion->getLength();
    GPUPortion *curr = head;
    GPUPortion *prev = nullptr;

    // Case 2: Traverse to find the insertion point
    // The list is sorted by the length of the free portions in ascending order
    // so we keep traversing until we find a portion that is smaller than the new portion
    // We want to insert the new portion before the first portion that is smaller than it, to keep the list sorted
    while (curr != nullptr && curr->getLength() < freeSize) {
        prev = curr;
        curr = curr->next;
    }

    // Case 3: We reached the end and the last element is still < the new portion, 
    // so we insert the new portion at the end of the list using the prev pointer we tracked
    if (curr == nullptr) {
        prev->next = freePortion;
        freePortion->prev = prev;
        return;
    }

    // Case 4: Insert before curr, which is the first portion smaller than the new portion
    if (curr == head) {
        // If curr is the head, we need to update the head pointer to point to the new portion
        freePortion->next = curr;
        // The current head's previous pointer should point to the new portion
        curr->prev = freePortion;
        // Update the head pointer to point to the new portion
        head = freePortion;
    } else {
        // Insert the new portion between curr->prev and curr
        freePortion->next = curr;
        freePortion->prev = curr->prev;
        curr->prev->next = freePortion;
        curr->prev = freePortion;
    }
}

/**
 * @brief Find a free portion that can fit the container's resource demand and time window
 * 
 * @param portionList 
 * @param container 
 * @return GPUPortion* 
 */
GPUPortion* Controller::findFreePortionForInsertion(GPUPortionList &portionList, ContainerHandle *container) {
    if (container == nullptr) {
        spdlog::get("container_agent")->error("Container is null");
        return nullptr;
    }

    // Safely lock the weak_ptr to access the PipelineModel
    auto pm = container->pipelineModel.lock();
    if (!pm) {
        spdlog::get("container_agent")->error("Container's pipeline model is null or expired");
        return nullptr;
    }

    GPUPortion *curr = portionList.head;
    while (curr != nullptr) {
        std::uint64_t laneDutyCycle = curr->lane->dutyCycle;

        // TODO: For now we still apply the scheduling logic of CORAL but with assigned gpu index based on pipeline name to avoid
        // inconsistency.
        uint8_t desiredGPUIndex;
        if (container->name.find("traffic") != std::string::npos) {
            desiredGPUIndex = 0;
        } else if (container->name.find("people") != std::string::npos) {
            desiredGPUIndex = 1;
        } else if (container->name.find("factory") != std::string::npos) {
            desiredGPUIndex = 2;
        } else if (container->name.find("campus") != std::string::npos) {
            desiredGPUIndex = 3;
        } else {
            spdlog::get("container_agent")->error("Cannot determine GPU index for container {}", container->name);
            return nullptr;
        }

        std::uint8_t laneGPUIndex = curr->lane->laneNum / NUM_LANES_PER_GPU;
        if (laneGPUIndex != desiredGPUIndex) {
            curr = curr->next;
            continue;
        }
        if (curr->start <= container->startTime &&
            curr->end >= container->endTime &&
            pm->localDutyCycle >= laneDutyCycle) {
            
            spdlog::get("container_agent")->debug("Found portion on Lane {} for container {}", 
                                                  curr->lane->laneNum, container->name);
            return curr;
        }
        curr = curr->next;
    }
    spdlog::get("container_agent")->debug("No suitable portion found for container {}", container->name);
    return nullptr;
}

/**
 * @brief Once a portion is found for a container and the container doesn't use up the whole portion, we need to divide the portion into three parts: 
 * the used portion in the middle, and the free portions on the left and right.
 *
 * @param node
 * @param scheduledPortion
 * @param toBeDividedFreePortion
 */
std::pair<GPUPortion *, GPUPortion *> Controller::insertUsedGPUPortion(
    GPUPortionList &portionList, 
    std::shared_ptr<ContainerHandle> container,
    GPUPortion *toBeDividedFreePortion
) {
    auto gpuLane = toBeDividedFreePortion->lane;

    // Safely lock the PipelineModel and Task to access their properties later
    auto pm = container->pipelineModel.lock();
    if (!pm) throw std::runtime_error("Pipeline model expired before portion assignment");
    
    auto task = pm->task.lock();
    if (!task) throw std::runtime_error("Task expired before portion assignment");

    // Create the new portion for the container in the lane and assign the container to it
    auto usedPortionUPtr = std::make_unique<GPUPortion>(gpuLane);
    GPUPortion *usedPortion = usedPortionUPtr.get();

    // Convert container raw pointer to shared pointer
    usedPortion->assignContainer(container);
    
    // Set the physical boundaries of the used portion
    usedPortion->start = container->startTime;
    usedPortion->end = container->endTime;

    // Capture original neighbors to ensure stitching logic remains consistent
    GPUPortion* originalPrev = toBeDividedFreePortion->prevInLane;
    GPUPortion* originalNext = toBeDividedFreePortion->nextInLane;

    // Handle the left portion
    auto &head = portionList.head;
    // new portion on the left
    uint64_t newStart = toBeDividedFreePortion->start;
    uint64_t newEnd = container->startTime;

    GPUPortion* leftPortion = nullptr;
    bool goodLeft = false;

    if (newEnd - newStart > 0) {
        auto leftPortionUPtr = std::make_unique<GPUPortion>(gpuLane);
        leftPortion = leftPortionUPtr.get();
        leftPortion->start = newStart;
        leftPortion->end = newEnd;

        // Link [Prev] <-> [Left] <-> [Used]
        leftPortion->prevInLane = originalPrev;
        leftPortion->nextInLane = usedPortion;
        
        if (originalPrev != nullptr) {
            originalPrev->nextInLane = leftPortion;
        }
        usedPortion->prevInLane = leftPortion;

        if (toBeDividedFreePortion == gpuLane->portionList.head) {
            gpuLane->portionList.head = leftPortion;
        }

        gpuLane->portionList.list.push_back(std::move(leftPortionUPtr));

        if (newEnd - newStart >= MINIMUM_PORTION_SIZE) {
            goodLeft = true;
        }
    } else {
        // No left portion: Link [Prev] <-> [Used]
        usedPortion->prevInLane = originalPrev;
        if (originalPrev != nullptr) {
            originalPrev->nextInLane = usedPortion;
        }
        if (toBeDividedFreePortion == gpuLane->portionList.head) {
            gpuLane->portionList.head = usedPortion;
        }
    }

    // Handle the right portion
    newStart = container->endTime;
    auto laneDutyCycle = gpuLane->dutyCycle;
    
    // If the lane duty cycle is 0, it means the lane is currently not being used by any container
    if (laneDutyCycle == 0) {
        if (pm->localDutyCycle == 0) {
            throw std::runtime_error("Duty cycle of the container 0");
        }
        // Time remained in the pipeline after this contianer, also the only one in the lane right now.
        int64_t slack = static_cast<int64_t>(task->tk_slo) - static_cast<int64_t>(pm->localDutyCycle * 2);
        if (slack < 0) {
            spdlog::get("container_agent")->error("Slack is negative. Duty cycle is larger than the SLO");
        }
        laneDutyCycle = pm->localDutyCycle;
        newEnd = pm->localDutyCycle;
    } else {
        newEnd = toBeDividedFreePortion->end;
    }

    GPUPortion* rightPortion = nullptr;
    bool goodRight = false;
    
    if (newEnd - newStart > 0) {
        auto rightPortionUPtr = std::make_unique<GPUPortion>(gpuLane);
        rightPortion = rightPortionUPtr.get();
        rightPortion->start = newStart;
        rightPortion->end = newEnd;

        // Link [Used] <-> [Right] <-> [Next]
        rightPortion->nextInLane = originalNext;
        rightPortion->prevInLane = usedPortion;
        
        if (originalNext != nullptr) {
            // Link the next portion in the lane to the right portion [Right] <-> [Next]
            originalNext->prevInLane = rightPortion;
        }
        // Link the used portion to the right portion [Used] <-> [Right]]
        usedPortion->nextInLane = rightPortion;

        // Hand over the ownership of the right portion to the lane's portion list
        gpuLane->portionList.list.push_back(std::move(rightPortionUPtr));

        if (newEnd - newStart >= MINIMUM_PORTION_SIZE) {
            goodRight = true;
        }
    } else {
        // No right portion: Link [Used] <-> [Next]
        usedPortion->nextInLane = originalNext;
        if (originalNext != nullptr) {
            originalNext->prevInLane = usedPortion;
        }
    }

    gpuLane->dutyCycle = laneDutyCycle;

    // Unstitch the original free portion from the global free portion list as it is now occupied by the container
    if (toBeDividedFreePortion->prev != nullptr) {
        toBeDividedFreePortion->prev->next = toBeDividedFreePortion->next;
    } else {
        head = toBeDividedFreePortion->next;
    }
    if (toBeDividedFreePortion->next != nullptr) {
        toBeDividedFreePortion->next->prev = toBeDividedFreePortion->prev;
    }

    // Remove from ownership list (O(N) search for now)
    portionList.list.remove(toBeDividedFreePortion);
    gpuLane->portionList.list.remove_if([&](const auto& ptr) {
        return ptr.get() == toBeDividedFreePortion;
    });

    // This must happen before returning to prevent usedPortionUPtr from deleting usedPortion
    gpuLane->portionList.list.push_back(std::move(usedPortionUPtr));

    if (goodLeft) {
        insertFreeGPUPortion(portionList, leftPortion);
    }
    if (goodRight) {
        insertFreeGPUPortion(portionList, rightPortion);
    }

    return {leftPortion, rightPortion};
}

/**
 * @brief Remove a free GPU portion from the list of free portions
 * This happens when a container is removed from the system and its portion is reclaimed
 * and merged with the free portions on the left and right.
 * These left and right portions are to be removed from the list of free portions.
 *
 * @param portionList
 * @param toBeRemovedPortion
 * @return true
 * @return false
 */
bool Controller::removeFreeGPUPortion(GPUPortionList &portionList, GPUPortion *toBeRemovedPortion) {
    if (toBeRemovedPortion == nullptr) {
        spdlog::get("container_agent")->error("Portion to be removed doesn't exist");
        return false;
    }

    // Safely check if the portion is currently occupied
    auto container = toBeRemovedPortion->container.lock();
    // Check if container is a weak_ptr that can be locked to a shared_ptr, 
    // which means the portion is still being used by a container and cannot be removed from the free portion list
    if (container) {
        spdlog::get("container_agent")->error("Portion to be removed is being used by container {0:s}", container->name);
        return false;
    }

    auto &head = portionList.head;

    // Safely unstitch the intrusive linked list pointers
    portionList.list.remove(toBeRemovedPortion);

    if (toBeRemovedPortion->prev != nullptr) {
        toBeRemovedPortion->prev->next = toBeRemovedPortion->next;
    } else {
        if (toBeRemovedPortion != head) {
            throw std::runtime_error("Portion is not the head of the list but its previous is null");
        }
        head = toBeRemovedPortion->next;
    }
    
    if (toBeRemovedPortion->next != nullptr) {
        toBeRemovedPortion->next->prev = toBeRemovedPortion->prev;
    }

    spdlog::get("container_agent")->info("Portion from {0:d} to {1:d} removed from the list of free portions of lane {2:d}",
                                         toBeRemovedPortion->start,
                                         toBeRemovedPortion->end,
                                         toBeRemovedPortion->lane->laneNum);
    return true;
}

/**
 * @brief After a container is removed from the system, we need to reclaim its portion and merge it with the free portions on the left and right if they are free, 
 * to avoid fragmentation of the GPU time.
 *
 * @param toBeReclaimedPortion
 * @return true
 * @return false
 */
bool Controller::reclaimGPUPortion(GPUPortion *toBeReclaimedPortion) {
    if (toBeReclaimedPortion == nullptr) {
        throw std::runtime_error("Portion to be reclaimed is null");
    }

    GPULane *gpuLane = toBeReclaimedPortion->lane;
    NodeHandle *node = gpuLane->node;

    spdlog::get("container_agent")->info("Reclaiming portion from {0:d} to {1:d} in lane {2:d}",
                                         toBeReclaimedPortion->start,
                                         toBeReclaimedPortion->end,
                                         gpuLane->laneNum);

    auto container = toBeReclaimedPortion->container.lock();
    if (container) {
        spdlog::get("container_agent")->warn("Portion is being used by container {0:s}", container->name);
    }

    // Mark the portion as free by resetting its container pointer to null
    toBeReclaimedPortion->container.reset();

    /**
     * @brief Organizing the lsit of portions in the lane the container is currently using
     *
     */

    GPUPortion *leftInLanePortion = toBeReclaimedPortion->prevInLane;
    GPUPortion *rightInLanePortion = toBeReclaimedPortion->nextInLane;

    // Merge left
    // If the left portion is not null, then the current portion is not the head of the list, 
    // and we can try to merge it with the left portion if it is free
    if (leftInLanePortion != nullptr) {
        // If the left portion is not occupied by a container, signalled by an empty weak_ptr, 
        // then we can merge the current portion with the left portion by resetting the start of the current portion to the start of the left portion, 
        // and removing the left portion from the list of portions in the lane
        if (leftInLanePortion->container.expired()) {
            spdlog::get("container_agent")->trace("Left portion is free and can be merged with the portion to be reclaimed.");

            // Update boundaries of the portion to be reclaimed by merging it with the left portion
            // The start of the merged portion will be the start of the left portion
            toBeReclaimedPortion->start = leftInLanePortion->start;
            // The previous portion of the merged portion will be the previous portion of the left portion, 
            // as the left portion will be removed from the lane's portion list
            toBeReclaimedPortion->prevInLane = leftInLanePortion->prevInLane;
            if (leftInLanePortion->prevInLane != nullptr) {
                // If the left portion is not the head of the lane's portion list, we need to link the previous portion of the left portion to the portion to be reclaimed
                leftInLanePortion->prevInLane->nextInLane = toBeReclaimedPortion;
            }

            // If the left portion is the head of the lane's portion list, we need to update the head pointer to point to the portion to be reclaimed
            if (leftInLanePortion == gpuLane->portionList.head) {
                gpuLane->portionList.head = toBeReclaimedPortion;
            }

            // Remove the left portion from the lane's portion list as it is now merged with the portion to be reclaimed
            removeFreeGPUPortion(node->freeGPUPortions, leftInLanePortion);

            gpuLane->removePortion(leftInLanePortion);
        }
    }

    if (rightInLanePortion != nullptr) {
        // If the right portion is not occupied by a container, signalled by an empty weak_ptr, 
        // then we can merge the current portion with the right portion by resetting the end of the current portion to the end of the right portion, 
        // and removing the right portion from the list of portions in the lane
        if (rightInLanePortion->container.expired()) {
            spdlog::get("container_agent")->trace("Right portion is free and can be merged with the portion to be reclaimed.");

            // Update boundaries of the portion to be reclaimed by merging it with the right portion
            // The end of the merged portion will be the end of the right portion
            toBeReclaimedPortion->end = rightInLanePortion->end;
            // The next portion of the merged portion will be the next portion of the right portion, 
            // as the right portion will be removed from the lane's portion list
            toBeReclaimedPortion->nextInLane = rightInLanePortion->nextInLane;
            if (rightInLanePortion->nextInLane != nullptr) {
                // If the right portion is not the head of the lane's portion list, we need to link the next portion of the right portion to the portion to be reclaimed
                rightInLanePortion->nextInLane->prevInLane = toBeReclaimedPortion;
            }

            // Remove the right portion from the lane's portion list as it is now merged with the portion to be reclaimed
            removeFreeGPUPortion(node->freeGPUPortions, rightInLanePortion);

            gpuLane->removePortion(rightInLanePortion);
        }
    }
        
    if (toBeReclaimedPortion->prevInLane == nullptr) {
        toBeReclaimedPortion->start = 0;
    }

    // If the lane is now completely free after merging with the left and right portions, we can reset its duty cycle to 0 to indicate that it is not being used by any container,
    // and reset the end of the portion to the maximum portion size to indicate that it covers the whole lane again
    if (toBeReclaimedPortion->nextInLane == nullptr && toBeReclaimedPortion->start == 0) {
        toBeReclaimedPortion->end = MAX_PORTION_SIZE;
        gpuLane->dutyCycle = 0;
    }

    // Re insert the merged portion into the list of free portions as it is now free after merging with the left and right portions
    insertFreeGPUPortion(node->freeGPUPortions, toBeReclaimedPortion);

    return true;
}

/**
 * @brief Remove a portion from the lane's portion list when the container using it is removed from the system, 
 * and update the next and prev pointers of the neighboring portions in the lane to recover the lane's original structure
 * before the portion was divided for the container.
 * 
 * @param portion 
 * @return true 
 * @return false 
 */
bool GPULane::removePortion(GPUPortion *portion) {
    if (portion == nullptr) {
        spdlog::get("container_agent")->error("Cannot remove null portion from lane {}", laneNum);
        return false;
    }

    if (portion->lane != this) {
        std::string containerName = "unassigned";
        if (auto cont = portion->container.lock()) {
            containerName = cont->name;
        }
        throw std::runtime_error(absl::StrFormat("Lane %d cannot remove portion %s, which does not belong to it.", laneNum, containerName));
    }
    if (portion->prevInLane != nullptr) {
        portion->prevInLane->nextInLane = portion->nextInLane;
        
    }
    if (portion->nextInLane != nullptr) {
        portion->nextInLane->prevInLane = portion->prevInLane;
    }

    if (portion == portionList.head) {
        portionList.head = portion->nextInLane;
    }
    portion->prevInLane = nullptr;
    portion->nextInLane = nullptr;

    auto it = std::find_if(portionList.list.begin(), portionList.list.end(),
                           [&](const std::unique_ptr<GPUPortion>& ptr) {
                               return ptr.get() == portion;
                           });

    if (it != portionList.list.end()) {
        // This erase() is what finally deletes the GPUPortion object
        portionList.list.erase(it);
        return true;
    }

    return false;
}

bool GPUHandle::addContainer(std::shared_ptr<ContainerHandle> container) {
    if (container == nullptr) {
        spdlog::get("container_agent")->error("Cannot add null container to GPU {} of {}", number, hostName);
        return false;
    }

    // For datasource and sink containers, we allow them to be added to the GPU without checking the memory limit,
    // as they essentially don't consume GPU memory for the purpose of simplifying the scheduling algorithm.
    if (container->name.find("datasource") != std::string::npos ||
        container->name.find("sink") != std::string::npos) {

        containers.insert({container->name, container});
        container->gpuHandle = this;
        spdlog::get("container_agent")->info("Container {} successfully added to GPU {} of {}", container->name, number, hostName);
        return true;
    }


    MemUsageType potentialMemUsage;
    potentialMemUsage = currentMemUsage + container->getExpectedTotalMemUsage();
    
    if (potentialMemUsage > memLimit) {
        spdlog::get("container_agent")->error("Container {} cannot be assigned to GPU {} of {}"
                                            "due to memory limit", container->name, number, hostName);
        return false;
    }
    containers.insert({container->name, container});
    container->gpuHandle = this;
    currentMemUsage = potentialMemUsage;
    spdlog::get("container_agent")->info("Container {} successfully added to GPU {} of {}", container->name, number, hostName);
    return true;
}

bool GPUHandle::removeContainer(std::shared_ptr<ContainerHandle> container) {
    if (container == nullptr) {
        spdlog::get("container_agent")->error("Cannot remove null container from GPU {} of {}", number, hostName);
        return false;
    }

    if (containers.find(container->name) == containers.end()) {
        spdlog::get("container_agent")->error("Container {} not found in GPU {} of {}", container->name, number, hostName);
        return false;
    }
    containers.erase(container->name);
    container->gpuHandle = nullptr;

    if (container->name.find("datasource") == std::string::npos &&
        container->name.find("sink") == std::string::npos) {
        
        MemUsageType memUsageToFree = container->getExpectedTotalMemUsage();
        if (memUsageToFree > currentMemUsage) {
            spdlog::get("container_agent")->error("Memory usage to free {} is larger than current memory usage {} on GPU {} of {}",
                                                        memUsageToFree, currentMemUsage, number, hostName);
            currentMemUsage = 0;
        } else {
            currentMemUsage -= memUsageToFree;
        }
    }

    spdlog::get("container_agent")->info("Container {} successfully removed from GPU {} of {}", container->name, number, hostName);
    return true;

}

// ============================================================ Configurations ============================================================ //
// ======================================================================================================================================== //
// ======================================================================================================================================== //
// ======================================================================================================================================== //

void Controller::readConfigFile(const std::string &path) {
    std::ifstream file(path);
    json j = json::parse(file);

    ctrl_experimentName = j["expName"];
    ctrl_systemName = j["systemName"];
    ctrl_runtime = j["runtime"];
    ctrl_clusterCount = j["cluster_count"];
    ctrl_port_offset = j["port_offset"];
    ctrl_systemFPS = j["system_fps"];
    ctrl_initialBatchSizes["yolov5"] = j["yolov5_batch_size"];
    ctrl_initialBatchSizes["edge"] = j["edge_batch_size"];
    ctrl_initialBatchSizes["server"] = j["server_batch_size"];
    ctrl_controlTimings.schedulingIntervalSec = j["scheduling_interval_sec"];
    ctrl_controlTimings.rescalingIntervalSec = j["rescaling_interval_sec"];
    ctrl_controlTimings.scaleUpIntervalThresholdSec = j["scale_up_interval_threshold_sec"];
    ctrl_controlTimings.scaleDownIntervalThresholdSec = j["scale_down_interval_threshold_sec"];
    initialTasks = j["initial_pipelines"];
    AddDevice("sink");
    AddDevice("server");
    if (j.contains("initial_devices")) {
        auto initialDevices = j["initial_devices"];
        for (const auto &device: initialDevices)
            AddDevice(device);
    } else {
        for (const auto &task : initialTasks) {
            AddDevice(task.srcDevice);
            AddDevice(task.edgeNode);
        }
    }

    if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") ctrl_fcpo_config = j["fcpo_parameters"];
}

void TaskDescription::from_json(const nlohmann::json &j, TaskDescription::TaskStruct &val) {
    j.at("pipeline_name").get_to(val.name);
    j.at("pipeline_target_slo").get_to(val.slo);
    j.at("pipeline_type").get_to(val.type);
    j.at("video_source").get_to(val.stream);
    j.at("pipeline_source_device").get_to(val.srcDevice);
    if (j.contains("pipeline_edge_node"))
        j.at("pipeline_edge_node").get_to(val.edgeNode);
    else
        val.edgeNode = "server";
    val.fullName = val.name + "_" + val.srcDevice;
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //


// ============================================================= Con/Destructors ============================================================== //
// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //


Controller::Controller(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    std::ifstream file("../jsons/experiments/cluster_info.json");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open cluster info file");
    }
    ctrl_clusterInfo = json::parse(file);
    readConfigFile(absl::GetFlag(FLAGS_ctrl_configPath));
    readInitialObjectCount("../jsons/experiments/object_count.json");

    ctrl_logPath = absl::GetFlag(FLAGS_ctrl_logPath);
    ctrl_logPath += "/" + ctrl_experimentName;
    std::filesystem::create_directories(
            std::filesystem::path(ctrl_logPath)
    );
    ctrl_logPath += "/" + ctrl_systemName;
    std::filesystem::create_directories(
            std::filesystem::path(ctrl_logPath)
    );
    ctrl_verbose = absl::GetFlag(FLAGS_ctrl_verbose);
    ctrl_loggingMode = absl::GetFlag(FLAGS_ctrl_loggingMode);

    setupLogger(
            ctrl_logPath,
            "controller",
            ctrl_loggingMode,
            ctrl_verbose,
            ctrl_loggerSinks,
            ctrl_logger
    );

    ctrl_containerLib = getContainerLib("all");

    std::ifstream metricsFile("../jsons/experiments/metricsserver.json");
    if (!metricsFile.is_open()) {
        throw std::runtime_error("Failed to open ../jsons/experiments/metricsserver.json");
    }
    json metricsCfgs = json::parse(metricsFile);
    ctrl_metricsServerConfigs.from_json(metricsCfgs);
    ctrl_metricsServerConfigs.schema = abbreviate(ctrl_experimentName + "_" + ctrl_systemName);
    ctrl_metricsServerConfigs.user = "controller";
    ctrl_metricsServerConfigs.password = "agent";
    ctrl_metricsServerConn = connectToMetricsServer(ctrl_metricsServerConfigs, "controller");

    // Check if schema exists
    std::string sql = "SELECT schema_name FROM information_schema.schemata WHERE schema_name = '" + ctrl_metricsServerConfigs.schema + "';";
    pqxx::result res = pullSQL(*ctrl_metricsServerConn, sql);
    if (res.empty()) {
        sql = "CREATE SCHEMA IF NOT EXISTS " + ctrl_metricsServerConfigs.schema + ";";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "ALTER DEFAULT PRIVILEGES IN SCHEMA " + ctrl_metricsServerConfigs.schema + 
              " GRANT ALL PRIVILEGES ON TABLES TO controller;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "ALTER DEFAULT PRIVILEGES IN SCHEMA " + ctrl_metricsServerConfigs.schema + " GRANT SELECT, INSERT ON TABLES TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
        sql = "GRANT USAGE, CREATE ON SCHEMA " + ctrl_metricsServerConfigs.schema + " TO device_agent, container_agent;";
        pushSQL(*ctrl_metricsServerConn, sql);
    }


    if (ctrl_systemName != "fcpo" && ctrl_systemName != "bce") {
        std::thread networkCheckThread(&Controller::checkNetworkConditions, this);
        networkCheckThread.detach();
    }

    running = true;
    ctrl_clusterID = 0;

    std::string server_address = absl::StrFormat("tcp://*:%d", CONTROLLER_API_PORT + ctrl_port_offset);
    api_ctx = context_t(1);
    api_socket = socket_t(api_ctx, ZMQ_REP);
    api_socket.bind(server_address);
    api_socket.set(zmq::sockopt::rcvtimeo, 1000);

    server_address = absl::StrFormat("tcp://*:%d", CONTROLLER_RECEIVE_PORT + ctrl_port_offset);
    system_ctx = context_t(2);
    server_socket = socket_t(system_ctx, ZMQ_REP);
    server_socket.bind(server_address);
    server_socket.set(zmq::sockopt::rcvtimeo, 1000);
    system_handlers = {
        {MSG_TYPE[DEVICE_ADVERTISEMENT], std::bind(&Controller::handleDeviseAdvertisement, this, std::placeholders::_1)},
        {MSG_TYPE[DUMMY_DATA], std::bind(&Controller::handleDummyDataRequest, this, std::placeholders::_1)},
        {MSG_TYPE[START_FL], std::bind(&Controller::handleForwardFLRequest, this, std::placeholders::_1)},
        {MSG_TYPE[SINK_METRICS], std::bind(&Controller::handleSinkMetrics, this, std::placeholders::_1)}
    };

    server_address = absl::StrFormat("tcp://*:%d", CONTROLLER_MESSAGE_QUEUE_PORT + ctrl_port_offset);
    message_queue = socket_t(system_ctx, ZMQ_PUB);
    message_queue.bind(server_address);
    message_queue.set(zmq::sockopt::sndtimeo, 100);

    if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") {
        // FIX: Use make_unique for the unique_ptr allocation
        ctrl_fcpo_server = std::make_unique<FCPOServer>(ctrl_systemName + "_" + ctrl_experimentName, ctrl_fcpo_config, ctrl_clusterCount, &message_queue);
    }

    ctrl_nextSchedulingTime = std::chrono::system_clock::now();
}

Controller::~Controller() {
    running = false;
    if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") ctrl_fcpo_server->stop();
    for (auto msvc: containers.getList()) {
        // FIX: Lock the weak_ptr to safely access the device agent before stopping the container
        auto deviceAgent = msvc->device_agent.lock();
        StopContainer(msvc, deviceAgent.get(), true);
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
    for (auto &device: devices.getList()) {
        if (device->name == "sink") { continue; }
        sendMessageToDevice(device->name, MSG_TYPE[DEVICE_SHUTDOWN], "");
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
}

// =========================================================== Executor/Maintainers ============================================================ //
// ============================================================================================================================================= //
// ============================================================================================================================================= //
// ============================================================================================================================================= //

/**
 * @brief Calculate the expected total memory usage of a container based on the process profiles in the pipeline model, which is used for scheduling decisions.
 * 
 * @return MemUsageType 
 */
MemUsageType ContainerHandle::getExpectedTotalMemUsage() const {
    auto agent = device_agent.lock();
    auto pModel = pipelineModel.lock();

    if (!agent || !pModel) {
        spdlog::get("container_agent")->error("Cannot calculate mem usage: device agent or pipeline model is null");
        return 0; 
    }

    std::string deviceTypeName = getDeviceTypeName(agent->type);
    
    try {
        if (agent->type == SystemDeviceType::Virtual || agent->type == SystemDeviceType::Server
                    || agent->type == SystemDeviceType::OnPremise) {
            return pModel->processProfiles.at(deviceTypeName).batchInfer.at(pModel->batchSize).gpuMemUsage;
        }
        return (pModel->processProfiles.at(deviceTypeName).batchInfer.at(pModel->batchSize).gpuMemUsage +
                pModel->processProfiles.at(deviceTypeName).batchInfer.at(pModel->batchSize).rssMemUsage) / 1000;
    } catch (const std::out_of_range& e) {
        spdlog::get("container_agent")->error("Missing memory profile for device type {} or batch size {}", deviceTypeName, pModel->batchSize);
        return 0;
    }
}

void Controller::AddDevice(const std::string name) {
    // check if devices already contains the device
    if (devices.hasDevice(name)) return;
    std::string ip = ctrl_clusterInfo[name]["ip"];
    int port = ctrl_clusterInfo[name]["port"];
    SystemInfo request;
    request.set_name(ctrl_systemName);
    request.set_experiment(ctrl_experimentName);
    std::string message = absl::StrFormat("%s %s", MSG_TYPE[CONNECT_DEVICE], request.SerializeAsString());
    message_t zmq_msg(message.size()), reply;
    memcpy(zmq_msg.data(), message.data(), message.size());
    context_t ctx(1);
    socket_t socket(ctx, ZMQ_REQ);
    socket.set(zmq::sockopt::sndtimeo, 100);
    socket.set(zmq::sockopt::rcvtimeo, 100);
    socket.set(zmq::sockopt::linger, 0);
    std::string address = absl::StrFormat("tcp://%s:%d", ip, port);
    socket.connect(address);
    if (!socket.send(zmq_msg)) {
        spdlog::error("Failed to send connection request to device {}.", ip);
        return;
    }
    if (!socket.recv(reply)) {
        spdlog::error("Failed to receive reply from device {}.", ip);
        return;
    }
    DeviceInfo info;
    if (!info.ParseFromString(reply.to_string())) {
        spdlog::error("Failed to parse reply from device {}.", ip);
        return;
    }
    std::string deviceName = info.name();
    
    auto node = std::make_shared<NodeHandle>(deviceName, ip, static_cast<SystemDeviceType>(info.type()),
                                      DATA_BASE_PORT + port - DEVICE_RECEIVE_PORT, std::map<std::string, std::shared_ptr<ContainerHandle>>{});
                                      
    // Use .get() to pass the raw pointer temporarily for initialization
    initialiseGPU(node.get(), info.processors(), std::vector<int>(info.memory().begin(), info.memory().end()));

    devices.addDevice(deviceName, node);
    spdlog::info("Device {} is connected to the system", deviceName);

    queryInDeviceNetworkEntries(devices.getDevice(deviceName));

    if (node->type != SystemDeviceType::Server && node->type != SystemDeviceType::SinkDevice) {
        std::thread networkCheck(&Controller::initNetworkCheck, this, std::ref(*(devices.getDevice(deviceName))), 1000, 300000, 30);
        networkCheck.detach();
    } else {
        node->initialNetworkCheck = true;
    }
}

/**
 * @brief Add a new task (pipeline) to the system by creating a TaskHandle for it and retrieving the corresponding pipeline models based on the task description,
 * 
 * @param t 
 * @return true 
 * @return false 
 */
bool Controller::AddTask(const TaskDescription::TaskStruct &t) {
    std::cout << "Adding task: " << t.name << std::endl;
    
    auto task = std::make_shared<TaskHandle>(t.name, t.type, t.stream, t.srcDevice, t.slo, ClockType{});

    std::map<std::string, std::shared_ptr<NodeHandle>> deviceList = devices.getMap();

    if (deviceList.find(t.srcDevice) == deviceList.end()) {
        spdlog::error("Device {0:s} is not connected", t.srcDevice);
        return false;
    }

    // FIX: Eliminate the Data Race by securely locking the mutex before checking the boolean
    while (true) {
        bool networkChecked = false;
        {
            std::lock_guard<std::mutex> lock(deviceList.at(t.srcDevice)->networkCheckMutex);
            networkChecked = deviceList.at(t.srcDevice)->initialNetworkCheck;
        }
        
        if (networkChecked) break;
        
        spdlog::get("container_agent")->info("Waiting for device {0:s} to finish network check", t.srcDevice);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    task->tk_src_device = t.srcDevice;

    task->tk_pipelineModels = getModelsByPipelineType(t.type, t.srcDevice, t.name, t.stream, t.edgeNode);
    for (auto &model: task->tk_pipelineModels) {
        model->datasourceName = {t.stream};
        model->task = task; // Safely assigns the shared_ptr to the weak_ptr
    }

    ctrl_savedUnscheduledPipelines.addTask(task->tk_name, task);
    return true;
}

/**
 * @brief Initialising the GPU abstraction for a newly joined device based on its type and the memory limits of its GPUs, which is used for scheduling decisions.
 * 
 * @param node 
 * @param numGPUs 
 * @param memLimits 
 */
void Controller::initialiseGPU(NodeHandle *node, int numGPUs, std::vector<int> memLimits) {
    // Sanity check to prevent out-of-bounds access if memLimits is empty
    if (memLimits.empty()) {
        spdlog::get("container_agent")->error("Failed to initialize GPU: memLimits array is empty.");
        return;
    }

    if (node->type == SystemDeviceType::Virtual) {
        node->gpuHandles.push_back(std::make_unique<GPUHandle>("3090", node->name, 0, memLimits[0] - 2000, NUM_LANES_PER_GPU, node));
        // TODO: differentiate between virtualEdge and virtualServer
        // node->gpuHandles.push_back(std::make_unique<GPUHandle>(node->name, node->name, 0, memLimits[0] / 5, 1, node));
    } else if (node->type == SystemDeviceType::Server) {
        for (uint8_t gpuIndex = 0; gpuIndex < numGPUs - 2; gpuIndex++) {
            // Added boundary check to prevent out-of-bounds access on memLimits array
            if (gpuIndex >= memLimits.size()) break;
            std::string gpuName = "gpu" + std::to_string(gpuIndex);
            node->gpuHandles.push_back(std::make_unique<GPUHandle>("3090", "server", gpuIndex, memLimits[gpuIndex] - 2000, NUM_LANES_PER_GPU, node));
        }
    } else {
        node->gpuHandles.push_back(std::make_unique<GPUHandle>(node->name, node->name, 0, memLimits[0] - 1500, 1, node));
    }
}

/**
 * @brief Basic GPU scheduling algorithm for the baselines that iterates through the new containers to be scheduled and assigns them to the GPU with 
 * the biggest memory gap after accommodating the container, while ensuring that the memory limit of the GPU is not exceeded.
 * 
 * @param new_containers 
 */
void Controller::basicGPUScheduling(std::vector<std::shared_ptr<ContainerHandle>> new_containers) {
    //
    std::map<std::string, std::vector<std::shared_ptr<ContainerHandle>>> scheduledContainers;
    
    for (auto device: devices.getMap()) {
        for (auto &container: new_containers) {
            auto agent = container->device_agent.lock();
            if (!agent || agent->name != device.first) {
                continue;
            }
            if (container->name.find("datasource") != std::string::npos ||
                container->name.find("sink") != std::string::npos) {
                continue;
            }
            scheduledContainers[device.first].push_back(container);
        }
        
        std::sort(scheduledContainers[device.first].begin(), scheduledContainers[device.first].end(),
                [](const std::shared_ptr<ContainerHandle>& a, const std::shared_ptr<ContainerHandle>& b) {
                    auto aMemUsage = a->getExpectedTotalMemUsage();
                    auto bMemUsage = b->getExpectedTotalMemUsage();
                    return aMemUsage > bMemUsage;
                });
    }
    
    for (auto device: devices.getMap()) {
        auto &gpus = device.second->gpuHandles;
        
        for (auto &container: scheduledContainers[device.first]) {
            MemUsageType containerMemUsage = container->getExpectedTotalMemUsage();
            MemUsageType biggestGap  = 0; // Standardized to 0 to align with memory comparisons
            
            // Save a pointer to the winning GPU instead of an index to prevent out-of-bounds access
            GPUHandle* targetGpu = nullptr; 
            
            for (auto &gpu: gpus) {
                // Use addition to completely prevent unsigned integer underflow
                MemUsageType requiredMem = gpu->currentMemUsage + containerMemUsage + 500;
                
                if (gpu->memLimit < requiredMem) {
                    continue; // Not enough memory
                }
                
                MemUsageType gap = gpu->memLimit - (gpu->currentMemUsage + containerMemUsage);
                
                if (gap > biggestGap) {
                    biggestGap = gap;
                    targetGpu = gpu.get(); // Save pointer to the specific GPU object
                }
            }
            
            if (targetGpu == nullptr) {
                spdlog::get("container_agent")->error("No GPU available for container {}", container->name);
                continue;
            }
            
            targetGpu->addContainer(container);
        }
    }
}

/**
 * @brief Main scheduling function that applies the scheduling decisionS.

 * Essentially, it compares the desired state of the system after scheduling, which is represented by the scheduled pipelines and their corresponding models and containers, 
 * with the current state of the system, which is represented by the currently running containers on the devices, and makes necessary adjustments to align the current state 
 * with the desired state.
 *
 */
void Controller::ApplyScheduling() {
    std::vector<std::shared_ptr<ContainerHandle>> new_containers;

    std::map<std::string, std::shared_ptr<TaskHandle>> pastScheduledPipelines = ctrl_pastScheduledPipelines.getMap();
    std::map<std::string, std::shared_ptr<TaskHandle>> newScheduledPipelines = ctrl_scheduledPipelines.getMap();

    // designate all current models no longer valid to run
    for (auto &[pipeName, pastTask]: pastScheduledPipelines) {
        for (auto &pastModel: pastTask->tk_pipelineModels) {
            pastModel->toBeRun = false;
        }
    }

    /**
    * @brief For models that have their upstreams as datasources and they are on the same node, we modify their names to use containers with datasources combined 
    * to reduce the cost of communication between the datasource and the model, which is a common optimization for edge devices with limited resources.
    * */
    for (auto &[pipeName, pipe]: newScheduledPipelines) {
        for (auto &model: pipe->tk_pipelineModels) {
            bool upstreamIsDatasource = false;
//            bool upstreamIsDatasource = (std::find_if(model->upstreams.begin(), model->upstreams.end(),
//                                                      [](const PipelineEdge &upstream) {
//                                                          if (auto up = upstream.targetNode.lock()) {
//                                                              return up->canBeCombined && (up->name.find("datasource") != std::string::npos);
//                                                          }
//                                                          return false;
//                                                      }) != model->upstreams.end());
            if (upstreamIsDatasource) {
                if (model->name.find("yolov5n") != std::string::npos && model->device != "server") {
                    if (model->name.find("yolov5ndsrc") == std::string::npos) {
                        model->name = replaceSubstring(model->name, "yolov5n", "yolov5ndsrc");
                }

                } else if (model->name.find("retinamtface") != std::string::npos && model->device != "server") {
                    if (model->name.find("retinamtfacedsrc") == std::string::npos) {
                        model->name = replaceSubstring(model->name, "retinamtface", "retinamtfacedsrc");
                    }
                } else if (model->name.find("firedetect") != std::string::npos && model->device != "server") {
                    if (model->name.find("firedetectdsrc") == std::string::npos) {
                        model->name = replaceSubstring(model->name, "firedetect", "firedetectdsrc");
                    }
                } else if (model->name.find("equipmentdetect") != std::string::npos && model->device != "server") {
                    if (model->name.find("equipmentdetectdsrc") == std::string::npos) {
                        model->name = replaceSubstring(model->name, "equipmentdetect", "equipmentdetectdsrc");
                    }
                }
            }
        }
    }
    /****************************************************************************************************************************************/
    
    std::vector<std::shared_ptr<ContainerHandle>> toBeStoppedContainers = {};

    /**
     * @brief Remove PipelineModel that are no longer valid to run based on the new scheduling decision by comparing pastScheduledPipelines with
     * the new scheduled pipelines, and stop and remove the containers of those PipelineModels from the system.
     * */

    for (auto &[pipeName, pastTask]: pastScheduledPipelines) {
        // If the pipeline is still in the new scheduled pipelines, check its models one by one.
        if (newScheduledPipelines.find(pipeName) != newScheduledPipelines.end()) {
            auto newTask = newScheduledPipelines[pipeName];
            
            for (auto &pastModel: pastTask->tk_pipelineModels) {
                for (auto &newModel: newTask->tk_pipelineModels) {
                    if (pastModel->name == newModel->name) {
                        // Mark pastModel so we don't kill its containers right now.
                        pastModel->toBeRun = true; 
                        // Mark newModel to signal to downstream logic that it is inherited.
                        newModel->toBeRun = true; 
                        break;
                    }
                }
            }
        }
    }

    for (auto &[pipeName, pastTask]: pastScheduledPipelines) {
        for (auto &pastModel: pastTask->tk_pipelineModels) {
            // If the model survived the graph reduction, skip the kill sequence
            if (pastModel->toBeRun) {
                continue; 
            }

            // The model is an orphan. Safely lock and kill its physical manifestations.
            if (auto pTask = pastModel->task.lock()) {
                for (auto &contWeak: pTask->tk_subTasks[pastModel->name]) {
                    if (auto container = contWeak.lock()) {
                        container->isRunning = false; // Mark the container as not running to prevent new scheduling decisions from trying to reuse it
                        toBeStoppedContainers.push_back(container);
                    }
                }
                // We dont have to worry about the deleting the pastModel object because it will be automatically deleted when the pastScheduledPipelines is 
                // updated with the new scheduled pipelines at the end of the ApplyScheduling function, as the pastModel is only referenced by the TaskHandle
                // in pastScheduledPipelines, and there will be no reference to it in the new scheduled pipelines if it is not valid to run anymore.

                // We also don't need to worry about the subTasks as they will be cleaned up when the pastScheduledPipelines is updated, and the containers 
                // in the subTasks have already been stopped and removed from the system, so there should be no dangling pointers or memory leaks.
            }
        }
    }

    /****************************************************************************************************************************************/

    /**
     * @brief // Turn schedule tasks/pipelines into containers
     * Containers that are already running may be kept running if they are still valid
     */
    for (auto &[pipeName, pipe]: ctrl_scheduledPipelines.getMap()) {
        for (auto &model: pipe->tk_pipelineModels) {
            if (ctrl_systemName == "tuti" && model->name.find("datasource") == std::string::npos && model->name.find("sink") == std::string::npos) {
                model->numReplicas = 3;
            } else if (ctrl_systemName == "rim" || ctrl_systemName == "dis") {
                // The GPU assignment here is dummy value and will be changed later with GPU scheduling
                model->cudaDevices.emplace_back(0);
                model->numReplicas = 1;
            }

            /**
             * @brief Reusing the containers of the same model in the past scheduled pipelines if they are still valid to run, which can reduce the
             * cost of starting new containers and make the scheduling more efficient.
             *
             * IMPORTANT: this block updates the task and PipelineModel link of the containers to the new scheduled pipeline, so it must be executed before 
             * any attempt to use container->task or container->pipelineModel
             * */
            std::unique_lock lock_model(model->pipelineModelMutex);
            // look for the model full name 
            std::string modelFullName = model->name;

            // check if the pipeline already been scheduled once before
            std::shared_ptr<PipelineModel> pastModel = nullptr;
            std::map<std::string, std::shared_ptr<TaskHandle>> pastScheduledPipelines = ctrl_pastScheduledPipelines.getMap();
            if (pastScheduledPipelines.find(pipeName) != pastScheduledPipelines.end()) {
                // look for the model in the past scheduled pipeline with the same name
                auto it = std::find_if(pastScheduledPipelines[pipeName]->tk_pipelineModels.begin(),
                                       pastScheduledPipelines[pipeName]->tk_pipelineModels.end(),
                                              [&modelFullName](const std::shared_ptr<PipelineModel> &m) {
                                                  return m->name == modelFullName;
                                              });
                // if the model is found in the past scheduled pipelines, its containers will be reused
                if (it != pastScheduledPipelines[pipeName]->tk_pipelineModels.end()) {
                    pastModel = *it;
                    // FIX: Safely lock task and container weak_ptrs
                    if (auto pTask = pastModel->task.lock()) {
                        if (auto currTask = model->task.lock()) {
                            std::vector<std::weak_ptr<ContainerHandle>> pastModelContainers = pTask->tk_subTasks[model->name];
                            for (auto contWeak: pastModelContainers) {
                                if (auto container = contWeak.lock()) {
                                    auto agent = container->device_agent.lock();
                                    if (agent && agent->name == model->device) {
                                        auto &subs = currTask->tk_subTasks[model->name];
                                        
                                        // Check if container is already in the new task's subtasks
                                        bool exists = false;
                                        for (auto& s : subs) {
                                            if (s.lock() == container) { exists = true; break; }
                                        }
                                        // If not, add it to the new task's subtasks and update the model's manifestations
                                        // and the live container's weak_ptrs to observe the new blueprint
                                        if (!exists) {
                                            subs.push_back(container);
                                            // Add the container to the list of manifestations for the model,
                                            // which represents the containers that are running the model in the current scheduled pipelines
                                            model->manifestations.push_back(container);
                                            // FIX: Update the live container's weak_ptrs to observe the logical resources (pipeline and task)
                                            // of the new scheduled pipeline
                                            container->pipelineModel = model;
                                            container->task = currTask;
                                        }
                                    }
                                }
                            }
                            pastModel->toBeRun = true;
                        }
                    }
                }
            }
            /****************************************************************************************************************************************/

            /**
             * @brief Scaling up and down the number of containers for the model if the number of candidates (containers of the same model in the past 
             * scheduled pipelines that can be reused) is smaller or larger than the desired number of replicas for the model.
             * */
            auto currTask = model->task.lock();
            if (!currTask) continue; 
            
            // List of candidate containers that can be reused for the current model
            std::vector<std::shared_ptr<ContainerHandle>> candidates;
            for (auto& cwk : currTask->tk_subTasks[model->name]) {
                if (auto cand = cwk.lock()) candidates.push_back(cand);
            }
            int candidate_size = candidates.size();
            
            // If the number of candidate containers is smaller than the desired number of replicas for the model, we need additional 
            // containers to meet the desired number of replicas
            if (candidate_size < model->numReplicas) {
                // start additional containers
                for (unsigned int i = candidate_size; i < model->numReplicas; i++) {
                    std::shared_ptr<ContainerHandle> container = TranslateToContainer(model, 
                                                                                      devices.getDevice(model->device), i);
                    if (container == nullptr) {
                        continue;
                    }
                    container->isRunning = true;
                    new_containers.push_back(container);
                    new_containers.back()->pipelineModel = model; // Safely assigns shared_ptr to weak_ptr
                    
                    currTask->tk_subTasks[model->name].push_back(container);
                    // Populate the model manifestations with the newly created container
                    model->manifestations.push_back(container);
                }
            // If the number of candidate containers is larger than the desired number of replicas for the model, we need to remove some of the
            // extra containers
            } else if (candidate_size > model->numReplicas) {
                // remove the extra containers
                for (int i = model->numReplicas; i < candidate_size; i++) {
                    // Mark the container as not running and add it to the list of containers to be stopped
                    candidates[i]->isRunning = false; 
                    // Add the container to the list of containers to be stopped, and the actual stopping and cleanup of the container will be handled later 
                    // at the end of this function
                    toBeStoppedContainers.push_back(candidates[i]);
                    
                    // auto agent = candidates[i]->device_agent.lock();
                    // StopContainer(candidates[i], agent ? agent.get() : nullptr);
                    
                    // Safely erase the scaled-down container from the current task
                    auto &subs = currTask->tk_subTasks[model->name];
                    subs.erase(
                        std::remove_if(subs.begin(), subs.end(), 
                            [&](const std::weak_ptr<ContainerHandle>& wk) {
                                return wk.lock() == candidates[i];
                            }), 
                        subs.end());
                        
                    // Safely erase the scaled-down container from current manifestations
                    auto &manifs = model->manifestations;
                    manifs.erase(
                        std::remove_if(manifs.begin(), manifs.end(), 
                            [&](const std::weak_ptr<ContainerHandle>& wk) {
                                return wk.lock() == candidates[i];
                            }), 
                        manifs.end());
                }
            }

            /****************************************************************************************************************************************/
        }
    }
    /****************************************************************************************************************************************/
    
    /**
     * @brief Rearranging the upstream and downstream connections between the containers based on the upstream and downstream relationships 
     * between the models in the scheduled pipelines,
     * */
    for (auto pipe: ctrl_scheduledPipelines.getList()) {
        
        // List of adjustments to be made to the downstream connection for each container
        // ContainerHandle, vector of pairs of (upstream container, adjust mode)
        ContainersDnstreamAdjustmentMap containerDnstreamAdjustMap = {};

        /**
         * @brief Find all the containers that are supposed to be stopped from the old scheduled pipelines so we can adjust the downstream connections 
         * of their upstream containers and
         * 
         */
        // upstream connections of their downstream containers accordingly
        auto pastPipe = pastScheduledPipelines.find(pipe->tk_name);
        if (pastPipe != pastScheduledPipelines.end()) {
            auto oldModelList = pastPipe->second->tk_pipelineModels;
            for (auto &model: oldModelList) {
                for (auto &contWeak: model->manifestations) {
                    if (auto cont = contWeak.lock()) {
                        bool isModelRunning = model->toBeRun;
                        auto deviceAgent = cont->device_agent.lock();
                        std::string containerIP = deviceAgent ? deviceAgent->ip + ":" + std::to_string(cont->recv_port) : "unknown";
                        if (!cont->isRunning) {
                            for (auto &[upName, upEdge]: cont->upstreams) {
                                if (auto upCont = upEdge.targetContainer.lock()) {
                                    // If the model of the container is running, but the container is to be stopped, this case is scaling down, we need to 
                                    // remove the container from its upstream connections
                                    if (isModelRunning) {
                                        if (std::find(toBeStoppedContainers.begin(), toBeStoppedContainers.end(), upCont) == toBeStoppedContainers.end()) {
                                            containerDnstreamAdjustMap[upCont].push_back({cont, AdjustMode::Remove, containerIP});
                                        }
                                    // If the model of the container is not running, then all the containers of the model will be stopped, and we need to
                                    // stop and shutdown the upstream container's connection (sender).
                                    } else {
                                        if (std::find(toBeStoppedContainers.begin(), toBeStoppedContainers.end(), upCont) == toBeStoppedContainers.end()) {
                                            containerDnstreamAdjustMap[upCont].push_back({cont, AdjustMode::Stop, ""});
                                        }
                                    }
                                    upCont->downstreams.erase(cont->name);
                                }
                            }
                            for (auto &[downName, downEdge]: cont->downstreams) {
                                if (auto downCont = downEdge.targetContainer.lock()) {
                                    downCont->upstreams.erase(cont->name);
                                }
                            }
                        }
                    }
                }
            }
        }
        /****************************************************************************************************************************************/

        for (auto &model: pipe->tk_pipelineModels) {
            // If it's a datasource, we don't have to do it now
            // datasource doesn't have upstreams and the downstreams will be set later
            if (model->name.find("datasource") != std::string::npos) continue;

            auto currTask = model->task.lock();
            if (!currTask) continue;

            for (auto &contWeak: currTask->tk_subTasks[model->name]) {
                auto container = contWeak.lock();
                if (!container) continue;

                /**
                 * @brief If the container is still running, we need to check if its upstreams are still the same as before based on the model's upstream 
                 * relationships in the logical model, and if the upstreams are changed, we need to update the upstream and downstream connections of the
                 * container accordingly,
                 * 
                 */
                std::map<std::shared_ptr<ContainerHandle>, PipelineEdge> desiredUpstreams;
                
                for (auto &edge: model->upstreams) {
                    if (auto upstream = edge.targetNode.lock()) {
                        if (auto upTask = upstream->task.lock()) {
                            for (auto &upContainerWk: upTask->tk_subTasks[upstream->name]) {
                                if (auto upContainer = upContainerWk.lock()) {
                                    desiredUpstreams[upContainer] = edge;
                                }
                            }
                        }
                    }
                }

                // The upstream edges of the container that are to be kept after rearrangement
                ContainerEdgeMap toBeKeptUpstreams;
                // We iterate through the old upstreams of the container, 
                // if the old upstream is still in the desired upstreams, we keep the connection but update the routing rules if necessary,
                for (auto &[oldUpName, oldEdge]: container->upstreams) {
                    if (auto oldUp = oldEdge.targetContainer.lock()) {
                        
                        auto it = desiredUpstreams.find(oldUp);
                        // If the old upstream is found in the desired upstreams, we keep the connection but check if the routing rules need to be updated,
                        // and if the routing rules changed, we add the container to the adjustment map with start mode to update its downstream connection later
                        if (it != desiredUpstreams.end() ) {
                            bool routingChanged = false;
                            
                            // Check if routing streams or classes changed
                            if (oldEdge.classOfInterest != it->second.classOfInterest ||
                                oldEdge.streamNames != it->second.streamNames) {
                                routingChanged = true;
                            }

                            // We add the new upstream edge to the container's to-be-kept upstreams, which will be swapped with the old upstreams at the end outside of the loop
                            toBeKeptUpstreams[oldUpName] = ContainerEdge{oldUp, it->second.classOfInterest, it->second.streamNames};

                            // We also update the downstream edge of the old upstream container to reflect the new routing rules if they changed
                            auto &downs = oldUp->downstreams;
                            downs[container->name] = ContainerEdge{container, it->second.classOfInterest, it->second.streamNames};

                            // If the routing rules changed, we MUST tell the upstream container to update its downstream connection to the current container with the new routing 
                            // rules
                            // But we only need to do this if the upstream container is neither a brand new container nor a container to be stopped, 
                            // because if it's a brand new container, it will set up the downstream connection with the correct routing rules when it starts, 
                            // and if it's a container to be stopped, we don't care about its downstream connection because it will be removed from the system soon
                            if (routingChanged && 
                                std::find(new_containers.begin(), new_containers.end(), oldUp) == new_containers.end()
                                && std::find(toBeStoppedContainers.begin(), toBeStoppedContainers.end(), oldUp) == toBeStoppedContainers.end()) {
                                    containerDnstreamAdjustMap[oldUp].push_back({container, AdjustMode::Routing, ""});
                            }

                        // If the old upstream is NOT found in the desired upstreams
                        } else {
                            // We need to remove the downstream edge from the old upstream container to the current container, 
                            // and add the old upstream edge to the adjustment map with stop mode to adjust its downstream connection later
                            auto &downs = oldUp->downstreams;
                            downs.erase(container->name);

                            if (std::find(toBeStoppedContainers.begin(), toBeStoppedContainers.end(), oldUp) == toBeStoppedContainers.end() &&
                                std::find(new_containers.begin(), new_containers.end(), oldUp) == new_containers.end()) {

                                    // We check if the old upstream container still has connection to any of the manifestation containers of the current model, 
                                    // if not, we need to stop the old upstream container's sender to the current container,
                                    // if yes, we just need to remove the current container from the round robin table of the old upstream container's sender 
                                    // without stopping it.
                                    for (auto &manifWeak: container->pipelineModel.lock()->manifestations) {
                                        if (auto manif = manifWeak.lock()) {
                                            if (oldUp->downstreams.find(manif->name) == oldUp->downstreams.end()) {
                                                // If the old upstream container has no downstream connection to any of the manifestation containers of the current model, 
                                                // we need to stop the old upstream container's sender to the current container
                                                containerDnstreamAdjustMap[oldUp].push_back({container, AdjustMode::Stop, ""});
                                                break; // No need to check further because we only need to stop the sender once for all manifestations of the same model
                                            }
                                        }
                                        containerDnstreamAdjustMap[oldUp].push_back({container, AdjustMode::Remove, ""});
                                    }
                            }
                        }
                    }
                }
                container->upstreams.swap(toBeKeptUpstreams);

                // We compare the desired upstreams with current upstreams of the container, 
                // if there is any desired upstream that is not in the current upstreams, we need to add the upstream edge to the container,
                for (auto &[desiredUpstrCont, contEdge] : desiredUpstreams) {
                    auto &upstrConts = container->upstreams;
                    
                    // If the desired upstream container is not in the current upstreams, 
                    // we need to add the upstream edge to the container, and also add the downstream edge from the desired upstream container to the current container,
                    if (upstrConts.find(desiredUpstrCont->name) == upstrConts.end()) {
                        // Add two-way edges between the desired upstream container and the current container with the routing rules specified in the model
                        upstrConts[desiredUpstrCont->name] = ContainerEdge{desiredUpstrCont, contEdge.classOfInterest, contEdge.streamNames};
                        auto &downs = desiredUpstrCont->downstreams;
                        downs[container->name] = ContainerEdge{container, contEdge.classOfInterest, contEdge.streamNames};

                        auto contModel = container->pipelineModel.lock();
                        
                        // If the upstream container is neither a brand new container nor a container to be stopped, we need to adjust its downstream connection to the current 
                        if (std::find(new_containers.begin(), new_containers.end(), desiredUpstrCont) == new_containers.end() && 
                            std::find(toBeStoppedContainers.begin(), toBeStoppedContainers.end(), desiredUpstrCont) == toBeStoppedContainers.end()) {
                                // We need to check if the upstream container has established the downstream connection to any of the manifestation containers 
                                // of the current model
                                // If NOT ESTABLISHED, we need the start mode to start a new sender in the upstream container to send data to the current container and 
                                //      all of the other manifestation containers of the current model
                                // If ESTABLISHED, we use the add mode to just add this container's IP to round robin table of upstream container's sender.
                                for (auto &manifWeak: contModel->manifestations) {
                                    if (auto manif = manifWeak.lock()) {
                                        if (desiredUpstrCont->downstreams.find(manif->name) == desiredUpstrCont->downstreams.end()) {
                                            // If the downstream connection is not established, we need to start a new sender in the upstream container to send data to the current container and all of the other manifestation containers of the current model
                                            containerDnstreamAdjustMap[desiredUpstrCont].push_back({container, AdjustMode::Start});
                                            break; // No need to check further because we only need one sender for all manifestations of the same model
                                        }
                                    }
                                }
                                containerDnstreamAdjustMap[desiredUpstrCont].push_back({container, AdjustMode::Add, ""});
                        }
                    }
                }

                /****************************************************************************************************************************************/
            }
        }
        for (auto &[container, adjustList]: containerDnstreamAdjustMap) {
            AdjustContainerDownstreamsInBatch(container, adjustList);
        }
    }
    /****************************************************************************************************************************************/

    /**
     * @brief Remove all the containers that are marked as not running from the system by stopping them and cleaning up their resources, 
     * 
     */

    for (auto &container: toBeStoppedContainers) {
        auto agent = container->device_agent.lock();
        auto model = container->pipelineModel.lock();
        auto task = container->task.lock();
        if (agent) {
            StopContainer(container, agent.get());
        }
        if (model) {
            model->manifestations.erase(
                std::remove_if(model->manifestations.begin(), model->manifestations.end(), 
                    [&](const std::weak_ptr<ContainerHandle>& wk) {
                        return wk.lock() == container;
                    }), 
                model->manifestations.end());
        }
        if (task) {
            task->tk_subTasks[model->name].erase(
                std::remove_if(task->tk_subTasks[model->name].begin(), task->tk_subTasks[model->name].end(), 
                    [&](const std::weak_ptr<ContainerHandle>& wk) {
                        return wk.lock() == container;
                    }), 
                task->tk_subTasks[model->name].end());
            if (task->tk_subTasks[model->name].empty()) {
                task->tk_subTasks.erase(model->name);
            }
        }
    }


    /**
     * @brief GPU scheduling. Baselines use the basic GPU scheduling algorithm that assigns containers to GPUs based on the biggest memory gap, 
     * while our proposed system uses a more advanced colocation temporal scheduling algorithm.
     * */
    if (ctrl_systemName != "ppp" && ctrl_systemName != "fcpo" && ctrl_systemName != "bce") {
        basicGPUScheduling(new_containers);
    } else {
        colocationTemporalScheduling();
    }

    /****************************************************************************************************************************************/

    /**
     * @brief Ensuring that the valid containers after scheduling are running on the devices with the correct batch size and the communication
     * connections between them are properly set up, while the invalid containers are stopped and removed from the system.
     * */

    for (auto pipe: ctrl_scheduledPipelines.getList()) {
        for (auto &model: pipe->tk_pipelineModels) {
            auto currTask = model->task.lock();
            if (!currTask) continue;
            
            for (auto &candWk: currTask->tk_subTasks[model->name]) {
                auto candidate = candWk.lock();
                if (!candidate) continue;
                
                if (std::find(new_containers.begin(), new_containers.end(), candidate) != new_containers.end() || candidate->model == Sink)
                    continue;
                
                auto agent = candidate->device_agent.lock();
                if (agent && agent->name != model->device) {
                    candidate->batch_size = model->batchSize;
                    MoveContainer(candidate, devices.getDevice(model->device).get());
                    continue;
                }
                if (candidate->batch_size != model->batchSize)
                    AdjustBatchSize(candidate, model->batchSize);
                AdjustTiming(candidate);
            }
        }
    }
    /****************************************************************************************************************************************/

    /**
     * @brief Starting the new containers for things to go brrr
     * 
     * @param new_containers 
     */
    for (auto container: new_containers) {
        StartContainer(container);
        containers.addContainer(container->name, container);
    }

    /****************************************************************************************************************************************/

    ctrl_pastScheduledPipelines.createSnapshotFrom(ctrl_scheduledPipelines);
    spdlog::get("container_agent")->info("SCHEDULING DONE! SEE YOU NEXT TIME!");
}

bool CheckMergable(const std::string &m) {
    return false;
//    return m.find("datasource") != std::string::npos || m.find("yolov5n") != std::string::npos || m.find("retinamtface") != std::string::npos ||
//           m.find("yolov5ndsrc") != std::string::npos || m.find("retinamtfacedsrc") != std::string::npos || \
//           m.find("firedetect") != std::string::npos || m.find("firedetectdsrc") != std::string::npos || m.find("equipmentdetect") != std::string::npos || \
//           m.find("equipmentdetectdsrc") != std::string::npos;
}

/**
 * @brief Translate a pipeline model into a container handle by assigning it a unique name, determining its class of interest based on its
 * upstreams, looking up the corresponding container type in the container library, and setting its dimensions and timing parameters based
 * on the model and the current scheduling time.
 * 
 * @param model 
 * @param device 
 * @param i 
 * @return ContainerHandle* 
 */
/**
 * @brief Translate a pipeline model into a container handle by assigning it a unique name, determining its class of interest based on its
 * upstreams, looking up the corresponding container type in the container library, and setting its dimensions and timing parameters based
 * on the model and the current scheduling time.
 * * @param model 
 * @param device 
 * @param i 
 * @return ContainerHandle* */
std::shared_ptr<ContainerHandle> Controller::TranslateToContainer(std::shared_ptr<PipelineModel> model, std::shared_ptr<NodeHandle> device, unsigned int i) {
    if (!model || !device) return nullptr;

    /**
     * @brief For models that can be combined with datasources, we will use the same container for the model and its downstreams if
     * the downstreams are on the same node and they can be combined
     * */
    if (model->name.find("datasource") != std::string::npos && model->canBeCombined) {
        // FIX: Updated to iterate over the PipelineEdge struct instead of a std::pair
        for (const auto &edge : model->downstreams) {
            if (auto downstream = edge.targetNode.lock()) {
                if (CheckMergable(downstream->name) && downstream->device != "server") {
                    return nullptr;
                }
            }
        }
    }
    /****************************************************************************************************************************************/

    /**
     * @brief
     * */
    std::string modelName = splitString(model->name, "_").back();

    // All models have a (COCO) class of interest from their upstreams
    // -1 means no specific class of interest, which means it takes everything from upstream
    int class_of_interest = -1;
    if (!model->upstreams.empty() && model->name.find("datasource") == std::string::npos &&
        model->name.find("dsrc") == std::string::npos) {
        class_of_interest = model->upstreams[0].classOfInterest;
    }

    auto task = model->task.lock();
    if (!task) {
        spdlog::get("container_agent")->error("Task is null for model {}", model->name);
        return nullptr;
    }

    std::string subTaskName = model->name;
    std::string containerName = ctrl_experimentName + "_" + ctrl_systemName + "_" + device->name + "_" + task->tk_name + "_" +
            modelName + "_" + std::to_string(i);
    // the name of the container type to look it up in the container library
    std::string containerTypeName = modelName + "_" + getDeviceTypeName(device->type);
    if (getDeviceTypeName(device->type) == "virtual")
        containerTypeName = modelName + "_server";

    if (ctrl_systemName == "ppp" || ctrl_systemName == "fcpo" || ctrl_systemName== "apis" || ctrl_systemName == "bce") {
        if (model->batchSize < model->datasourceName.size()) model->batchSize = model->datasourceName.size();
    } // ensure minimum global batch size setting for these configurations for a good comparison

    // Verify container type exists in the library to prevent silent insertions and null pointer errors later
    if (ctrl_containerLib.find(containerTypeName) == ctrl_containerLib.end()) {
        spdlog::get("container_agent")->error("Container type {} not found in library for model {}", containerTypeName, model->name);
        return nullptr;
    }
    std::string targetModelPath = ctrl_containerLib[containerTypeName].modelPath;
    
    auto typeIt = ModelTypeReverseList.find(modelName);
    if (typeIt == ModelTypeReverseList.end()) {
        spdlog::get("container_agent")->error("ModelType {} not found in Reverse List for model {}", modelName, model->name);
        return nullptr;
    }
    ModelType targetModelType = typeIt->second;

    // FIX: Safely assign the next free port using the device's mutex to prevent ZMQ binding collisions
    int assigned_recv_port = 0;
    {
        std::lock_guard<std::mutex> lock(device->nodeHandleMutex);
        assigned_recv_port = device->next_free_port++;
    }

    // Create the container handle for the model with the appropriate parameters
    auto container = std::make_shared<ContainerHandle>(
        containerName, 
        model->position_in_pipeline, 
        i,
        class_of_interest,
        targetModelType,
        CheckMergable(modelName) && model->canBeCombined,
        std::vector<int>{0},
        static_cast<uint64_t>(task->tk_slo),
        0.0f,
        model->batchSize,
        assigned_recv_port, // Pass safely allocated port
        targetModelPath,    // Pass safely extracted path
        device, // Shared pointer implicitly converts to weak_ptr inside the constructor
        task,   // Shared pointer implicitly converts to weak_ptr
        model   // Shared pointer implicitly converts to weak_ptr
    );
    
    try {
        if (model->name.find("datasource") != std::string::npos) {
            container->dimensions = ctrl_containerLib[containerTypeName].templateConfig.at("container").at("cont_pipeline").at(0).at("msvc_dataShape").at(0).get<std::vector<int>>();
        } else if (model->name.find("320") != std::string::npos) {
            container->dimensions = {3, 320, 320};
        } else if (model->name.find("512") != std::string::npos) {
            container->dimensions = {3, 512, 512};
        } else if (model->name.find("sink") == std::string::npos) {
            container->dimensions = ctrl_containerLib[containerTypeName].templateConfig.at("container").at("cont_pipeline").at(1).at("msvc_dnstreamMicroservices").at(0).at("nb_expectedShape").at(0).get<std::vector<int>>();
        }
    } catch (const std::exception& e) {
        spdlog::get("container_agent")->error("Failed to parse dimension data for container type {}. Check JSON structure. Error: {}", containerTypeName, e.what());
        // Dimensions safely remain at {0} initialized in the constructor
    }

    // container->timeBudgetLeft for lazy dropping
    container->timeBudgetLeft = model->timeBudgetLeft;
    // container start time
    container->startTime = model->startTime;
    // container end time
    container->endTime = model->endTime;
    // container SLO
    container->localDutyCycle = model->localDutyCycle;
    // 
    container->cycleStartTime = ctrl_currSchedulingTime;

    // FIX: REMOVED task->tk_subTasks[subTaskName].push_back(container);
    // FIX: REMOVED model->manifestations.push_back(container);
    // Reason: ApplyScheduling() already explicitly handles registering this container into the task and model vectors.
    // Pushing them here creates duplicate memory registrations that corrupt the orchestrator loops.

    // for (auto &upstream: model->upstreams) {
    //     std::string upstreamSubTaskName = upstream.first->name;
    //     for (auto &upstreamContainer: upstream.first->task->tk_subTasks[upstreamSubTaskName]) {
    //         container->upstreams.push_back(upstreamContainer);
    //         upstreamContainer->downstreams.push_back(container);
    //     }
    // }

    // for (auto &downstream: model->downstreams) {
    //     std::string downstreamSubTaskName = downstream.first->name;
    //     for (auto &downstreamContainer: downstream.first->task->tk_subTasks[downstreamSubTaskName]) {
    //         container->downstreams.push_back(downstreamContainer);
    //         downstreamContainer->upstreams.push_back(container);
    //     }
    // }
    
    return container;
}

/**
 * @brief Adjust the timing parameters of the container based on the timing parameters of its pipeline model and the current scheduling 
 * time, which is used for scheduling decisions and also sent to the device agent to update the time keeping of the container.
 * 
 * @param container 
 */
void Controller::AdjustTiming(std::shared_ptr<ContainerHandle> container) {
    if (!container) return;

    auto pm = container->pipelineModel.lock();
    if (!pm) {
        spdlog::get("container_agent")->error("PipelineModel is null for container {}", container->name);
        return;
    }

    auto agent = container->device_agent.lock();
    if (!agent) {
        spdlog::get("container_agent")->error("Device agent is null for container {}", container->name);
        return;
    }

    // container->timeBudgetLeft for lazy dropping
    container->timeBudgetLeft = pm->timeBudgetLeft;
    // container->start_time
    container->startTime = pm->startTime;
    // container->end_time
    container->endTime = pm->endTime;
    // duty cycle of the lane where the container is assigned
    container->localDutyCycle = pm->localDutyCycle;
    // `container->task->tk_slo` for the total SLO of the pipeline
    container->cycleStartTime = ctrl_currSchedulingTime;

    
    TimeKeeping message;
    message.set_slo(container->pipelineSLO);
    message.set_time_budget(container->timeBudgetLeft);
    message.set_start_time(container->startTime);
    message.set_end_time(container->endTime);
    message.set_local_duty_cycle(container->localDutyCycle);
    message.set_cycle_start_time(std::chrono::duration_cast<TimePrecisionType>(container->cycleStartTime.time_since_epoch()).count());



    PackagedMsg request;
    request.set_target_name(container->name);
    request.set_target_type(MSG_TYPE[TIME_KEEPING_UPDATE]);
    request.set_payload(message.SerializeAsString());

    sendMessageToDevice(agent->name, MSG_TYPE[TO_CONTAINER], request.SerializeAsString());
    spdlog::get("container_agent")->info("Requested container {0:s} to update time keeping!", container->name);
}

void Controller::StartContainer(std::shared_ptr<ContainerHandle> container, bool easy_allocation) {
    if (!container) return;

    // Safely lock all core observer pointers immediately
    auto agent = container->device_agent.lock();
    auto task = container->task.lock();
    auto pModel = container->pipelineModel.lock();

    if (!agent || !task || !pModel) {
        spdlog::get("container_agent")->error("Cannot start container {}: missing core observers", container->name);
        return;
    }

    spdlog::get("container_agent")->info("Starting container: {0:s}", container->name);
    ContainerConfig request;
    json start_config;
    unsigned int control_port;

    std::string pipelineName = task->tk_name;
    ModelType model = static_cast<ModelType>(container->model);
    std::string modelName = getContainerName(agent->type, model);
    spdlog::get("container_agent")->info("Creating container: {0:s} of model {1:s} on device {2:s}", container->name, modelName, agent->name);

    // If its a sink, we only need to send some very basic info to the device agent to start the container
    if (model == ModelType::Sink) {
        start_config["experimentName"] = ctrl_experimentName;
        start_config["systemName"] = ctrl_systemName;
        start_config["pipelineName"] = pipelineName;
        start_config["controllerIP"] = "<IP>";
        control_port = container->recv_port;
    } else {
        start_config = ctrl_containerLib[modelName].templateConfig;

        /**
         * @brief Injecting global orchestration parameters
         * 
         */
        // adjust container configs
        start_config["container"]["cont_experimentName"] = ctrl_experimentName;
        start_config["container"]["cont_systemName"] = ctrl_systemName;
        start_config["container"]["cont_pipeName"] = pipelineName;
        start_config["container"]["cont_hostDevice"] = agent->name;
        start_config["container"]["cont_hostDeviceType"] = SystemDeviceTypeList[agent->type];
        start_config["container"]["cont_name"] = container->name;
        start_config["container"]["cont_allocationMode"] = easy_allocation ? 1 : 0;
        if (ctrl_systemName == "bce") {
            start_config["container"]["cont_batchMode"] = 0;
        } else if (ctrl_systemName == "ppp") {
            //TODO: set back to 2 after OURs working again with batcher
            start_config["container"]["cont_batchMode"] = 1;
        } else if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") {
            start_config["container"]["cont_batchMode"] = 1;
        } 
        
        if (ctrl_systemName == "ppp" || ctrl_systemName == "jlf") {
            start_config["container"]["cont_dropMode"] = 1;
        }
        start_config["container"]["cont_pipelineSLO"] = task->tk_slo;
        start_config["container"]["cont_timeBudgetLeft"] = container->timeBudgetLeft;
        start_config["container"]["cont_startTime"] = container->startTime;
        start_config["container"]["cont_endTime"] = container->endTime;
        start_config["container"]["cont_localDutyCycle"] = container->localDutyCycle;
        start_config["container"]["cont_cycleStartTime"] = std::chrono::duration_cast<TimePrecisionType>(container->cycleStartTime.time_since_epoch()).count();

        /****************************************************************************************************************************************/

        /**
         * @brief Injecting model profiles for the model in the container
         * 
         */

        if (container->model != DataSource) {
            std::vector<uint32_t> modelProfile;
            for (auto &[batchSize, profile]: pModel->processProfiles.at(SystemDeviceTypeList[agent->type]).batchInfer) {
                modelProfile.push_back(batchSize);
                modelProfile.push_back(profile.p95prepLat);
                modelProfile.push_back(profile.p95inferLat);
                modelProfile.push_back(profile.p95postLat);
            }

            if (modelProfile.empty()) {
                spdlog::get("container_agent")->warn("Model profile not found for container: {0:s}", container->name);
            }
            start_config["container"]["cont_modelProfile"] = modelProfile;
            if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") {
                ctrl_fcpo_server->incrementClientCounter();
            }
        }

        json base_config = start_config["container"]["cont_pipeline"];

        // adjust pipeline configs
        for (auto &j: base_config) {
            j["msvc_idealBatchSize"] = container->batch_size;
            j["msvc_pipelineSLO"] = container->pipelineSLO;
        }
        if (model == ModelType::DataSource) {
            base_config[0]["msvc_dataShape"] = {container->dimensions};
            if (pModel->datasourceName[0].find("spot") != std::string::npos) {
                base_config[0]["msvc_idealBatchSize"] = 7; // spot data only available in 7 fps
            } else {
                base_config[0]["msvc_idealBatchSize"] = ctrl_systemFPS;
            }
        } else {
            if (model == ModelType::Yolov5nDsrc || model == ModelType::RetinaMtfaceDsrc || \
                model == ModelType::FireDetectDsrc || model == ModelType::EquipDetectDsrc) {
                    base_config[0]["msvc_dataShape"] = {container->dimensions};
                    base_config[0]["msvc_type"] = 500;
                    base_config[0]["msvc_idealBatchSize"] = ctrl_systemFPS;
            }
            base_config[1]["msvc_dnstreamMicroservices"][0]["nb_expectedShape"] = {container->dimensions};
            base_config[3]["path"] = container->model_file;
        }
        /****************************************************************************************************************************************/


        /**
         * @brief Binding the upstream communication connections
         * 
         */
        // adjust receiver upstreams
        base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"] = {};
        if (container->model == DataSource || container->model == Yolov5nDsrc || container->model == RetinaMtfaceDsrc || \
            container->model == FireDetectDsrc || container->model == EquipDetectDsrc) {
                base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = "video_source";
                for (auto &source: pModel->datasourceName) {
                    base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"].push_back(source);
                }
        } else {
            // FIX: Safely lock the weak_ptr inside the upstream pair
            if (!pModel->upstreams.empty()) {
                if (auto upModel = pModel->upstreams[0].targetNode.lock()) {
                    base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = upModel->name;
                } else {
                    base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = "empty";
                }
            } else {
                base_config[0]["msvc_upstreamMicroservices"][0]["nb_name"] = "empty";
            }
            base_config[0]["msvc_upstreamMicroservices"][0]["nb_link"].push_back(absl::StrFormat("0.0.0.0:%d", container->recv_port));
        }
        /****************************************************************************************************************************************/

        // if ((container->device_agent == container->upstreams[0]->device_agent) && (container->gpuHandle == container->upstreams[0]->gpuHandle)) {
        //     base_config[0]["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::localGPU;
        // } else {
        //     base_config[0]["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;
        // }
        //TODO: REMOVE THIS IF WE EVER DECIDE TO USE GPU COMM AGAIN

        /**
         * @brief Downstream communication
         * 
         */

        base_config[0]["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;

        // adjust sender downstreams
        json sender = base_config.back();
        uint16_t postprocessorIndex = base_config.size() - 2;
        json post_down = base_config[base_config.size() - 2]["msvc_dnstreamMicroservices"][0];
        base_config[base_config.size() - 2]["msvc_dnstreamMicroservices"] = json::array();
        base_config.erase(base_config.size() - 1);
        int i = 1;
        
        for (auto &edge: pModel->downstreams) {
            auto dwnstr = edge.targetNode.lock();
            int coi = edge.classOfInterest;
            if (!dwnstr) continue;

            json *postprocessor = &base_config[postprocessorIndex];
            sender["msvc_name"] = "sender" + std::to_string(i++);
            sender["msvc_dnstreamMicroservices"][0]["nb_name"] = dwnstr->name;
            sender["msvc_dnstreamMicroservices"][0]["nb_link"] = {};
            
            for (auto &[repName, repEdge]: container->downstreams) {
                auto replica = repEdge.targetContainer.lock();
                if (!replica) continue;
                
                auto repModel = replica->pipelineModel.lock();
                auto repAgent = replica->device_agent.lock();
                if (repModel && repAgent && repModel->name == dwnstr->name) {
                    sender["msvc_dnstreamMicroservices"][0]["nb_link"].push_back(
                            absl::StrFormat("%s:%d", repAgent->ip, replica->recv_port));
                }
            }
            post_down["nb_name"] = sender["msvc_name"];
            
            // Lock downstream device agent to compare routing bounds
            auto dwnstrAgent = dwnstr->deviceAgent.lock();
            if (agent != dwnstrAgent) {
                post_down["nb_commMethod"] = CommMethod::encodedCPU;
                sender["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;
            } else {
                // // TODO: REMOVE AND FIX THIS IF WE EVER DECIDE TO USE GPU COMM AGAIN
                // if ((container->gpuHandle == dwnstr->gpuHandle)) {
                //     post_down["nb_commMethod"] = CommMethod::localGPU;
                //     sender["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::localGPU;
                // } else {
                post_down["nb_commMethod"] = CommMethod::localCPU;
                sender["msvc_dnstreamMicroservices"][0]["nb_commMethod"] = CommMethod::serialized;
            }
            post_down["nb_classOfInterest"] = coi;

            // Inject the allowed streamNames array into the deployed container's configuration JSON
            post_down["nb_allowedProducerList"] = json::array();
            for (const auto& streamName : edge.streamNames) {
                post_down["nb_allowedProducerList"].push_back(streamName);
            }

            postprocessor->at("msvc_dnstreamMicroservices").push_back(post_down);
            base_config.push_back(sender);
            
            if (ctrl_systemName == "fcpo" || ctrl_systemName== "apis") {
                start_config["fcpo"] = ctrl_fcpo_server->getConfig();
                std::string deviceTypeName = getDeviceTypeName(agent->type);
                start_config["fcpo"]["timeout_size"] = (deviceTypeName == "server") ? 3 : 2;
                
                if (pModel->processProfiles.count(deviceTypeName)) {
                    start_config["fcpo"]["batch_size"] = pModel->processProfiles.at(deviceTypeName).maxBatchSize;
                } else {
                    start_config["fcpo"]["batch_size"] = 1; // Safe fallback
                }
                start_config["fcpo"]["threads_size"] = (deviceTypeName == "server") ? 4 : 2;
            }
        }

        start_config["container"]["cont_pipeline"] = base_config;
        control_port = container->recv_port - 5000;

        /****************************************************************************************************************************************/
    }
    container->fcpo_conf = start_config["fcpo"];

    /**
     * @brief Protobuf message construction and sending to the device agent to start the container with the specified configuration
     * */

    request.set_name(container->name);
    std::string docker_tag = "lucasliebe/pipeplusplus:";
    
    auto dev_type = agent->type;
    
    if (dev_type == Virtual || dev_type == Server || dev_type == OnPremise || dev_type == SinkDevice)
        docker_tag += "amd64-torch";
    else if (dev_type == NanoXavier || dev_type == NXXavier || dev_type == AGXXavier)
        docker_tag += "jp512-torch";
    else if (dev_type == OrinNano || dev_type == OrinNX || dev_type == OrinAGX)
        docker_tag += "jp61-torch";
    request.set_docker_tag(docker_tag);
    request.set_json_config(start_config.dump());
    std::cout << start_config.dump() << std::endl;
    request.set_executable(ctrl_containerLib[modelName].runCommand);
    
    if (container->model == DataSource || container->model == Sink) {
        request.set_device(-1);
    } else if (agent->name == "server") {
        // FIX: Added safe array bounds checking to prevent segfaults if the server has fewer than 4 GPUs
        if (container->gpuHandle == nullptr) {
            if (agent->gpuHandles.size() > 3) {
                container->gpuHandle = agent->gpuHandles[3].get();
            } else if (!agent->gpuHandles.empty()) {
                container->gpuHandle = agent->gpuHandles[0].get(); // Fallback to first available GPU
            }
        }
        
        // Final sanity check before dereferencing
        if (container->gpuHandle) {
            request.set_device(container->gpuHandle->number);
        } else {
            request.set_device(0);
        }
    } else {
        request.set_device(0);
    }
    
    request.set_control_port(control_port);
    request.set_model_type(container->model);
    for (auto &dim: container->dimensions) {
        request.add_input_shape(dim);
    }

    sendMessageToDevice(agent->name, MSG_TYPE[CONTAINER_START], request.SerializeAsString());
    spdlog::get("container_agent")->info("Requested container {0:s} to start!", container->name);
    /****************************************************************************************************************************************/
}

/**
 * @brief Move a container to a new device by stopping it on the old device, updating its device agent and communication connections, 
 * and starting it on the new device, while ensuring that the upstream and downstream containers are properly adjusted to maintain the 
 * correct data flow and meet the SLOs.
 * 
 * @param container 
 * @param device 
 */
void Controller::MoveContainer(std::shared_ptr<ContainerHandle> container, 
                               NodeHandle *device) {
    if (!container) return;

    // Lock the old device safely before manipulating it
    auto old_device_shp = container->device_agent.lock();
    if (!old_device_shp) {
        spdlog::get("container_agent")->error("Cannot move container, old device agent is null");
        return;
    }

    bool start_dsrc = false, merge_dsrc = false;
    // Move container from server -> device
    if (device->name != "server") {
        if (container->mergable) {
            merge_dsrc = true;
            if (container->model == Yolov5n) {
                container->model = Yolov5nDsrc;
            } else if (container->model == RetinaMtface) {
                container->model = RetinaMtfaceDsrc;
            } else if (container->model == FireDetect) {
                container->model = FireDetectDsrc;
            } else if (container->model == EquipDetect) {
                container->model = EquipDetectDsrc;
            }
        }
    // Move container from device -> server
    } else {
        if (container->mergable) {
            start_dsrc = true;
            if (container->model == Yolov5nDsrc) {
                container->model = Yolov5n;
            } else if (container->model == RetinaMtfaceDsrc) {
                container->model = RetinaMtface;
            } else if (container->model == FireDetectDsrc) {
                container->model = FireDetect;
            } else if (container->model == EquipDetectDsrc) {
                container->model = EquipDetect;
            }
        }
    }
    
    // The old address that the container is listening to for upstream communication,
    // which will be used by the upstream containers to send data to the container after it's moved
    std::string old_link = absl::StrFormat("%s:%d", old_device_shp->ip, container->recv_port);
    
    // FIX: We need a shared_ptr to assign to the container's weak_ptr device_agent
    auto new_device_shp = devices.getDevice(device->name);
    container->device_agent = new_device_shp;
    
    {
        std::lock_guard<std::mutex> lock(device->nodeHandleMutex);
        container->recv_port = device->next_free_port++;
    }
    
    new_device_shp->containers.insert({container->name, container});
    container->gpuHandle = container->gpuHandle; // Redundant, but preserved as requested
    
    StartContainer(container, !(start_dsrc || merge_dsrc));
    
    // Collect upstreams safely into a separate vector to prevent iterator invalidation
    std::vector<std::shared_ptr<ContainerHandle>> upstreams_to_process;
    
    // Lock weak_ptrs when traversing the upstream map
    for (auto &[upName, edge] : container->upstreams) {
        if (auto upstr = edge.targetContainer.lock()) {
            upstreams_to_process.push_back(upstr);
        }
    }
    
    // Now safe to manipulate and stop containers without crashing the map iterator
    for (auto &upstr : upstreams_to_process) {
        if (start_dsrc) {
            StartContainer(upstr, false);
            SyncDatasource(container, upstr);
        } else if (merge_dsrc) {
            SyncDatasource(upstr, container);
            StopContainer(upstr, old_device_shp.get());
        } else {
            AdjustContainerDownstreams(container, upstr, device, AdjustMode::Overwrite, old_link);
        }
    }
    
    StopContainer(container, old_device_shp.get());
    spdlog::get("container_agent")->info("Container {0:s} moved to device {1:s}", container->name, device->name);
    old_device_shp->containers.erase(container->name);
}

/**
 * @brief Adjust the downstream communication connections of a container by sending a message to the container's device agent, which will be forwarded
 * to the container.
 * Each message contain a list of adjustments, each of which is for a specific downstream container.
 * 
 * @param container 
 * @param dnstreamAdjustmentList 
 */
void Controller::AdjustContainerDownstreamsInBatch(const std::shared_ptr<ContainerHandle> container, 
                                            const SingleContainerDownstreamAdjustmentList &dnstreamAdjustmentList) {
    if (!container) return;

    auto senderAgent = container->device_agent.lock();
    if (!senderAgent) {
        spdlog::get("container_agent")->error("Cannot adjust downstreams: device agent is null for sender container {}", container->name);
        return;
    }

    BatchedConnections message;
    
    message.set_batch_size(dnstreamAdjustmentList.size());

    for (auto &[dwnstr, mode, old_link]: dnstreamAdjustmentList) {
        auto dwnstrAgent = dwnstr->device_agent.lock();
        if (!dwnstrAgent) {
            spdlog::get("container_agent")->error("Cannot adjust downstream, device agent is null for target container {}", dwnstr->name);
            continue;
        }

        Connection conn;
        conn.set_mode(mode);
        conn.set_name(dwnstr->name);
        conn.set_ip(dwnstrAgent->ip);
        conn.set_port(dwnstr->recv_port);
        conn.set_data_portion(1.0);
        conn.set_old_link(old_link);
        conn.set_offloading_duration(0);
        conn.set_class_of_interest(dwnstr->class_of_interest);
        
        auto edgeIt = container->downstreams.find(dwnstr->name);
        if (edgeIt != container->downstreams.end()) {
            for (const auto& stream : edgeIt->second.streamNames) {
                conn.add_allowed_producers(stream);
            }
        } else if (mode != AdjustMode::Stop) {
            // If the edge is missing and we AREN'T stopping, something is wrong with our state machine
            spdlog::get("container_agent")->error("An edge not found between {0:s} and {1:s} during downstream adjustment in batch", container->name, dwnstr->name);
        }

        message.add_connections()->CopyFrom(conn);
    }

    PackagedMsg request;
    request.set_target_name(container->name);
    request.set_target_type(MSG_TYPE[UPDATE_SENDER_IN_BATCH]);
    request.set_payload(message.SerializeAsString());

    sendMessageToDevice(senderAgent->name, MSG_TYPE[TO_CONTAINER], request.SerializeAsString());

    spdlog::get("container_agent")->info("Downstreams of {0:s} adjusted", container->name);
}

/**
 * @brief Adjust the downstream communication connection of a container by sending a message to the upstream container's device agent to update the
 * 
 * @param dnstreamCont 
 * @param container 
 * @param new_device 
 * @param mode 
 * @param old_link 
 */
void Controller::AdjustContainerDownstreams(std::shared_ptr<ContainerHandle> dnstreamCont, std::shared_ptr<ContainerHandle> container, NodeHandle *new_device,
                                AdjustMode mode, const std::string &old_link) {
    if (!dnstreamCont || !container) return;

    auto contModel = dnstreamCont->pipelineModel.lock();
    auto upstrAgent = container->device_agent.lock();

    if (!contModel || !upstrAgent) {
        spdlog::get("container_agent")->error("Cannot adjust upstream: missing pipeline model or device agent");
        return;
    }

    BatchedConnections message;

    Connection conn;
    conn.set_mode(mode);
    conn.set_name(contModel->name);
    conn.set_ip(new_device->ip);
    conn.set_port(dnstreamCont->recv_port);
    conn.set_data_portion(1.0);
    conn.set_old_link(old_link);
    conn.set_offloading_duration(0);
    conn.set_class_of_interest(dnstreamCont->class_of_interest);
    
    // FIX: Safely lookup the streamNames without recreating deleted edges
    auto edgeIt = container->downstreams.find(dnstreamCont->name);
    if (edgeIt != container->downstreams.end()) {
        for (const auto& stream : edgeIt->second.streamNames) {
            conn.add_allowed_producers(stream);
        }
    } else if (mode != AdjustMode::Stop) {
        // If the edge is missing and we AREN'T stopping, something is wrong with our state machine
        spdlog::get("container_agent")->error("An edge not found between {0:s} and {1:s} during downstream adjustment", container->name, dnstreamCont->name);
    }

    message.set_batch_size(1);
    message.add_connections()->CopyFrom(conn);

    PackagedMsg request;
    request.set_target_name(container->name);
    request.set_target_type(MSG_TYPE[UPDATE_SENDER_IN_BATCH]);
    request.set_payload(message.SerializeAsString());

    sendMessageToDevice(upstrAgent->name, MSG_TYPE[TO_CONTAINER], request.SerializeAsString());
    
    spdlog::get("container_agent")->info("Upstream of {0:s} adjusted to container {1:s}", container->name, contModel->name);
}

void Controller::SyncDatasource(std::shared_ptr<ContainerHandle> prev, std::shared_ptr<ContainerHandle> curr) {
    if (!prev || !curr) {
        spdlog::get("container_agent")->error("Cannot sync datasource: missing container handles");
        return;
    }
    
    auto currAgent = curr->device_agent.lock();
    if (!currAgent) {
        spdlog::get("container_agent")->error("Cannot sync datasource: current container's device agent is null");
        return;
    }

    Connection request;
    request.set_name(prev->name);
    request.set_old_link(curr->name);

    sendMessageToDevice(currAgent->name, MSG_TYPE[SYNC_DATASOURCES], request.SerializeAsString());
    spdlog::get("container_agent")->info("Datasource {0:s} synced with {1:s}", prev->name, curr->name);
}

void Controller::AdjustBatchSize(std::shared_ptr<ContainerHandle> msvc, int new_bs) {
    if (!msvc) {
        spdlog::get("container_agent")->error("Cannot adjust batch size: container handle is null");
        return;
    }

    auto agent = msvc->device_agent.lock();
    if (!agent) {
        spdlog::get("container_agent")->error("Cannot adjust batch size: device agent is null");
        return;
    }

    msvc->batch_size = new_bs;
    Int32 bs;
    bs.set_value(new_bs);

    PackagedMsg request;
    request.set_target_name(msvc->name);
    request.set_target_type(MSG_TYPE[BATCH_SIZE_UPDATE]);
    request.set_payload(bs.SerializeAsString());

    sendMessageToDevice(agent->name, MSG_TYPE[TO_CONTAINER], request.SerializeAsString());
    spdlog::get("container_agent")->info("Batch size of {0:s} adjusted to {1:d}", msvc->name, new_bs);
}

void Controller::AdjustCudaDevice(std::shared_ptr<ContainerHandle> msvc, GPUHandle *new_device) {
    if (!msvc) {
        spdlog::get("container_agent")->error("Cannot adjust CUDA device: container handle is null");
        return;
    }
    msvc->gpuHandle = new_device;
    // TODO: also adjust actual running container
}

void Controller::AdjustResolution(std::shared_ptr<ContainerHandle> msvc, std::vector<int> new_resolution) {
    if (!msvc) {
        spdlog::get("container_agent")->error("Cannot adjust resolution: container handle is null");
        return;
    }
    
    auto agent = msvc->device_agent.lock();
    if (!agent) {
        spdlog::get("container_agent")->error("Cannot adjust resolution: device agent is null");
        return;
    }

    msvc->dimensions = new_resolution;
    Dimensions dims;
    dims.set_channels(new_resolution[0]);
    dims.set_height(new_resolution[1]);
    dims.set_width(new_resolution[2]);

    PackagedMsg request;
    request.set_target_name(msvc->name);
    request.set_target_type(MSG_TYPE[RESOLUTION_UPDATE]);
    request.set_payload(dims.SerializeAsString());

    sendMessageToDevice(agent->name, MSG_TYPE[TO_CONTAINER], request.SerializeAsString());
    spdlog::get("container_agent")->info("Resolution of {0:s} adjusted to {1:d}x{2:d}x{3:d}",
                                      msvc->name, new_resolution[0], new_resolution[1], new_resolution[2]);
}

void Controller::StopContainer(std::shared_ptr<ContainerHandle> container, NodeHandle *device, bool forced) {
    if (!container || !device) return;

    spdlog::get("container_agent")->info("Stopping container: {0:s}", container->name);
    ContainerSignal request;
    request.set_name(container->name);
    request.set_forced(forced);
    sendMessageToDevice(device->name, MSG_TYPE[CONTAINER_STOP], request.SerializeAsString());

    if (container->gpuHandle != nullptr)
        container->gpuHandle->removeContainer(container);
        
    if ((ctrl_systemName == "fcpo" || ctrl_systemName== "apis") && container->model != DataSource && container->model != Sink) {
        ctrl_fcpo_server->decrementClientCounter();
    }
    
    if (!forced) { //not forced means the container is stopped during scheduling and should be removed
        containers.removeContainer(container->name);
        
        // Safely lock the weak_ptr to erase from the device agent's map
        if (auto agent = container->device_agent.lock()) {
            agent->containers.erase(container->name);
        }
    }
    
    // Safely remove this container from its upstreams' downstreams map using key erasure
    for (auto &[upName, edge]: container->upstreams) {
        if (auto upstr = edge.targetContainer.lock()) {
            upstr->downstreams.erase(container->name);
        }
    }
    
    // Safely remove this container from its downstreams' upstreams map using key erasure
    for (auto &[dwnName, edge]: container->downstreams) {
        if (auto dwnstr = edge.targetContainer.lock()) {
            dwnstr->upstreams.erase(container->name);
        }
    }
    spdlog::get("container_agent")->info("Container {0:s} stopped", container->name);
}

/**
 * @brief 
 * 
 * @param node 
 */
void Controller::queryInDeviceNetworkEntries(std::shared_ptr<NodeHandle> node) {
    if (!node) return;

    std::string deviceTypeName = SystemDeviceTypeList[node->type];
    std::string deviceTypeNameAbbr = abbreviate(deviceTypeName);
    
    if (ctrl_inDeviceNetworkEntries.find(deviceTypeName) == ctrl_inDeviceNetworkEntries.end()) {
        std::string tableName = "prof_" + deviceTypeNameAbbr + "_netw";
        std::string sql = absl::StrFormat("SELECT p95_transfer_duration_us, p95_total_package_size_b "
                                    "FROM %s ", tableName);
        pqxx::result res = pullSQL(*ctrl_metricsServerConn, sql);
        
        if (res.empty()) {
            spdlog::get("container_agent")->error("No in-device network entries found for device type {}.", deviceTypeName);
            return;
        }
        
        for (pqxx::result::const_iterator row = res.begin(); row != res.end(); ++row) {
            std::pair<uint32_t, uint64_t> entry = {row["p95_total_package_size_b"].as<uint32_t>(), row["p95_transfer_duration_us"].as<uint64_t>()};
            ctrl_inDeviceNetworkEntries[deviceTypeName].emplace_back(entry);
        }
        spdlog::get("container_agent")->info("Finished querying in-device network entries for device type {}.", deviceTypeName);
    }
    
    std::unique_lock lock(node->nodeHandleMutex);
    node->latestNetworkEntries[deviceTypeName] = aggregateNetworkEntries(ctrl_inDeviceNetworkEntries[deviceTypeName]);
    std::cout << node->latestNetworkEntries[deviceTypeName].size() << std::endl;
}

/**
 * @brief 
 * 
 * @param container calculating queue sizes for the container before its official deployment.
 * @param modelType 
 */
void Controller::calculateQueueSizes(std::shared_ptr<ContainerHandle> container, const ModelType modelType) {
    if (!container) return;

    float preprocessRate = 1000000.f / container->expectedPreprocessLatency; // queries per second
    float postprocessRate = 1000000.f / container->expectedPostprocessLatency; // qps
    float inferRate = 1000000.f / (container->expectedInferLatency * container->batch_size); // batch per second

    QueueLengthType minimumQueueSize = 30;

    // Receiver to Preprocessor
    // Utilization of preprocessor
    float preprocess_rho = container->arrival_rate / preprocessRate;
    QueueLengthType preprocess_inQueueSize = std::max((QueueLengthType) std::ceil(preprocess_rho * preprocess_rho / (2 * (1 - preprocess_rho))), minimumQueueSize);
    float preprocess_thrpt = std::min(preprocessRate, container->arrival_rate);

    // Preprocessor to Inference
    // Utilization of inference
    float infer_rho = preprocess_thrpt / container->batch_size / inferRate;
    QueueLengthType infer_inQueueSize = std::max((QueueLengthType) std::ceil(infer_rho * infer_rho / (2 * (1 - infer_rho))), minimumQueueSize);
    float infer_thrpt = std::min(inferRate, preprocess_thrpt / container->batch_size); // batch per second

    float postprocess_rho = (infer_thrpt * container->batch_size) / postprocessRate;
    QueueLengthType postprocess_inQueueSize = std::max((QueueLengthType) std::ceil(postprocess_rho * postprocess_rho / (2 * (1 - postprocess_rho))), minimumQueueSize);
    float postprocess_thrpt = std::min(postprocessRate, infer_thrpt * container->batch_size);

    QueueLengthType sender_inQueueSize = postprocess_inQueueSize * container->batch_size;

    container->queueSizes = {preprocess_inQueueSize, infer_inQueueSize, postprocess_inQueueSize, sender_inQueueSize};

    container->expectedThroughput = postprocess_thrpt;
}

// ============================================================ Communication Handlers ============================================================ //
// ================================================================================================================================================ //
// ================================================================================================================================================ //
// ================================================================================================================================================ //

void Controller::HandleControlMessages() {
    while (running) {
        message_t message;
        if (server_socket.recv(message, recv_flags::none)) {
            std::string raw = message.to_string();
            std::istringstream iss(raw);
            std::string topic;
            iss >> topic;
            iss.get(); // skip the space after the topic
            std::string payload((std::istreambuf_iterator<char>(iss)),
                                std::istreambuf_iterator<char>());
            if (system_handlers.count(topic)) {
                system_handlers[topic](payload);
            } else {
                spdlog::get("container_agent")->error("Received unknown topic: {}", topic);
            }
//        } else {
//            spdlog::get("container_agent")->trace("Communication Receive Timeout");
        }
    }
}

void Controller::handleDeviseAdvertisement(const std::string& msg) {
    DeviceInfo request;
    SystemInfo reply;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed to connect device with msg: {}", msg);
        return;
    }
    std::string deviceName = request.name();

    // Allocate the node safely using make_shared
    auto node = std::make_shared<NodeHandle>(
        deviceName, 
        request.ip_address(), 
        static_cast<SystemDeviceType>(request.type()),
        DATA_BASE_PORT + ctrl_port_offset + request.agent_port_offset(), 
        std::map<std::string, std::shared_ptr<ContainerHandle>>{}
    );

    reply.set_name(ctrl_systemName);
    reply.set_experiment(ctrl_experimentName);
    server_socket.send(message_t(reply.SerializeAsString()), send_flags::dontwait);
    
    // Pass the raw pointer to initialiseGPU, as it's safe within this scope
    initialiseGPU(node.get(), request.processors(), std::vector<int>(request.memory().begin(), request.memory().end()));
    
    // addDevice now securely takes ownership of the shared_ptr
    devices.addDevice(deviceName, node);
    spdlog::get("container_agent")->info("Device {} is connected to the system", request.name());
    
    queryInDeviceNetworkEntries(devices.getDevice(deviceName));

    if (node->type != SystemDeviceType::Server) {
        // The thread needs to capture the shared_ptr to keep the node alive during the detached thread execution
        auto targetNode = devices.getDevice(deviceName);
        std::thread networkCheck([this, targetNode]() {
            this->initNetworkCheck(*targetNode, 1000, 300000, 30);
        });
        networkCheck.detach();
    } else {
        node->initialNetworkCheck = true;
    }
}

void Controller::handleDummyDataRequest(const std::string& msg) {
    DummyMessage request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed handle dummy data: {}", msg);
        return;
    }
    ClockType now = std::chrono::system_clock::now();
    unsigned long diff = std::chrono::duration_cast<TimePrecisionType>(
            now - std::chrono::time_point<std::chrono::system_clock>(TimePrecisionType(request.gen_time()))).count();
    unsigned int size = request.data().size();
    network_check_buffer[request.origin_name()].push_back({size, diff});
    server_socket.send(message_t("success"), send_flags::dontwait);
}

void Controller::handleForwardFLRequest(const std::string& msg) {
    FlData request;
    if (!request.ParseFromString(msg)){
        spdlog::get("container_agent")->error("Failed adding FCPO client with msg: {}", msg);
        return;
    }
    for (auto &dev: devices.getMap()) {
        if (dev.first == request.device_name()) {
            if (ctrl_fcpo_server->addClient(request)) {
                spdlog::get("container_agent")->info("Successfully added client {} to FCPO Aggregation.", request.device_name());
                server_socket.send(message_t("success"), send_flags::dontwait);
            } else {
                spdlog::get("container_agent")->error("Failed adding client {} to FCPO Aggregation.", request.device_name());
                server_socket.send(message_t("error"), send_flags::dontwait);
            }
            break;
        }
    }
}

void Controller::handleSinkMetrics(const std::string& msg) {
    SinkMetrics request;
    if (!request.ParseFromString(msg)) {
        spdlog::get("container_agent")->error("Failed to parse sink metrics with msg: {}", msg);
        server_socket.send(message_t("error"), send_flags::dontwait);
        return;
    }
    if (!ctrl_scheduledPipelines.hasTask(request.name()))  {
        server_socket.send(message_t("notfound"), send_flags::dontwait);
        return;
    }
    
    std::shared_ptr<TaskHandle> task = ctrl_scheduledPipelines.getTask(request.name());
    if (task) {
        std::lock_guard<std::mutex> lock(task->tk_mutex);
        task->tk_lastLatency = request.avg_latency();
        task->tk_lastThroughput = request.throughput();
        server_socket.send(message_t("success"), send_flags::dontwait);
    }
}

void Controller::sendMessageToDevice(const std::string &topik, const std::string &type, const std::string &content) {
    std::string msg = absl::StrFormat("%s| %s %s", topik, type, content);
    message_t zmq_msg(msg.size());
    memcpy(zmq_msg.data(), msg.data(), msg.size());
    message_queue.send(zmq_msg, send_flags::none);
}

/**
 * @brief '
 * 
 * @param node 
 * @param minPacketSize bytes
 * @param maxPacketSize bytes
 * @param numLoops 
 * @return NetworkEntryType 
 */
NetworkEntryType Controller::initNetworkCheck(NodeHandle &node, uint32_t minPacketSize, uint32_t maxPacketSize, uint32_t numLoops) {
    if (!node.networkCheckMutex.try_lock()) {
        return {};
    }
    LoopRange request;
    request.set_min(minPacketSize);
    request.set_max(maxPacketSize);
    request.set_repetitions(numLoops);
    try {
        sendMessageToDevice(node.name, MSG_TYPE[NETWORK_CHECK], request.SerializeAsString());
        spdlog::get("container_agent")->info("Successfully started network check for device {}.", node.name);
    } catch (const std::exception &e) {
        spdlog::get("container_agent")->error("Error while starting network check for device {}.", node.name);
    }

    while (network_check_buffer[node.name].size() < numLoops) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    NetworkEntryType entries = network_check_buffer[node.name];
    entries = aggregateNetworkEntries(entries);
    network_check_buffer[node.name].clear();
    spdlog::get("container_agent")->info("Finished network check for device {}.", node.name);
    
    // find the closest latency between min and max packet size
    float latency = 0.0f;
    for (auto &entry: entries) {
        if (entry.first >= (minPacketSize + maxPacketSize) / 2) {
            latency = entry.second;
            break;
        }
    }
    
    // Scoped lock for node manipulation
    std::lock_guard lock(node.nodeHandleMutex);
    node.initialNetworkCheck = true;
    if (entries.empty()) entries = {std::pair<uint32_t, uint64_t>{1, 1}};
    node.latestNetworkEntries["server"] = entries;
    node.lastNetworkCheckTime = std::chrono::system_clock::now();
    if (ctrl_systemName == "fcpo") {
        if (node.transmissionLatencyHistory.size() > ctrl_bandwidth_predictor.getWindowSize()) {
            node.transmissionLatencyHistory.erase(node.transmissionLatencyHistory.begin());
        }
        node.transmissionLatencyHistory.push_back(latency / 1000.0f); // convert to ms
        node.transmissionLatencyPrediction = ctrl_bandwidth_predictor.predict(node.transmissionLatencyHistory);
    }
    node.networkCheckMutex.unlock();
    return entries;
};

/**
 * @brief Query the latest network entries for each device to determine the network conditions.
 * If no such entries exists, send to each device a request for network testing.
 * 
 */
void Controller::checkNetworkConditions() {
    std::this_thread::sleep_for(TimePrecisionType(5 * 1000000));
    while (running) {
        Stopwatch stopwatch;
        stopwatch.start();
        std::map<std::string, NetworkEntryType> networkEntries = {};

        for (const auto& [deviceName, nodeHandle] : devices.getMap()) {
            if (!nodeHandle) continue;
            std::unique_lock<std::mutex> lock(nodeHandle->nodeHandleMutex);
            bool initialNetworkCheck = nodeHandle->initialNetworkCheck;
            uint64_t timeSinceLastCheck = std::chrono::duration_cast<TimePrecisionType>(
                    std::chrono::system_clock::now() - nodeHandle->lastNetworkCheckTime).count() / 1000000;
            lock.unlock();
            
            if (nodeHandle->type == SystemDeviceType::Server || (initialNetworkCheck && timeSinceLastCheck < 60)) {
                spdlog::get("container_agent")->info("Skipping network check for device {}.", deviceName);
                continue;
            }
            initNetworkCheck(*nodeHandle, 1000, 300000, 30);
        }

        stopwatch.stop();
        auto elapsed_us = stopwatch.elapsed_microseconds();
        
        // Prevent unsigned underflow if network checks took longer than 60 seconds
        if (elapsed_us < 60ULL * 1000000ULL) {
            uint64_t sleepTimeUs = 60ULL * 1000000ULL - elapsed_us;
            std::this_thread::sleep_for(std::chrono::microseconds(sleepTimeUs));
        }
    }
}

// ============================================================================================================================================ //
// ============================================================================================================================================ //
// ============================================================================================================================================ //
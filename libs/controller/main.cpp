#include "controller.h"

int main(int argc, char **argv) {
    auto controller = new Controller(argc, argv);
    std::thread receiver_thread(&Controller::HandleRecvRpcs, controller);
    receiver_thread.detach();
    std::thread scheduling_thread(&Controller::Scheduling, controller);
    scheduling_thread.detach();
    std::string command;

    while (controller->isRunning()) {
        while (true) {
            // Get input from user
            std::cout << "You need to connect the devices before adding task. Ready? (yes/no): " << std::endl;
            std::cin >> command;
            if (command == "yes") {
                break;
            } else if (command == "no") {
                std::this_thread::sleep_for(std::chrono::seconds(5));
            } else {
                std::cout << "Invalid command" << std::endl;
            }
        }
        TaskDescription::TaskStruct task;
        std::cout << "Enter command {init, init_remain, traffic, video_call, people, exit): ";
        std::cin >> command;
        if (command == "exit") {
            controller->Stop();
            break;
        } else if (command == "init") {
            controller->Init();
            continue;
        } else if (command == "init_remain") {
            controller->InitRemain();
            continue;
        }  else if (command == "traffic") {
            task.type = PipelineType::Traffic;
        } else if (command == "video_call") {
            task.type = PipelineType::Indoor;
        } else if (command == "people") {
            task.type = PipelineType::Building_Security;
        } else {
            std::cout << "Invalid command" << std::endl;
            continue;
        }
        std::cout << "Enter name of task: ";
        std::cin >> task.name;
        std::cout << "Enter SLO in ns: ";
        std::cin >> task.slo;
        std::cout << "Enter total path to source file: ";
        std::cin >> task.source;
        std::cout << "Enter name of source device: ";
        std::cin >> task.device;
        bool added = controller->AddTask(task);
        if (!added) {
            std::cout << "Failed to add task" << std::endl;
            controller->addRemainTask(task);
        }
    }
    delete controller;
    return 0;
}

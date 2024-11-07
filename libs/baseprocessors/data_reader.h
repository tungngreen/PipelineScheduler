#ifndef PIPEPLUSPLUS_DATA_READER_H
#define PIPEPLUSPLUS_DATA_READER_H

#include <opencv2/opencv.hpp>
#include <thread>
#include "microservice.h"

class DataReader : public Microservice {
    friend class DataSourceAgent;
public:
    //msvc_idealBatchSize is used for the wait time, where 33 equals ~30.3 fps
    //the link to the upstream microservice is used to specify the file location
    DataReader(const json &jsonConfigs);

    ~DataReader() override {
        waitStop();
        source.release();
        spdlog::get("container_agent")->info("{0:s} has stopped", msvc_name);
    };

    void dispatchThread() override {
        std::thread handler(&DataReader::Process, this);
        READY = true;
        handler.detach();
    }

    PerSecondArrivalRecord getPerSecondArrivalRecord() override;

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing) override;

private:
    void Process();

    std::string link;
    cv::VideoCapture source;
    int target_fps;
    int wait_time_ms;
    float skipRatio;
};


#endif //PIPEPLUSPLUS_DATA_READER_H

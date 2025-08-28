#include "controller.h"

std::vector<std::string> Controller::getPipelineNames() {
    return {"traffic", "people", "indoor", "surveillancerobot", "campusdrone", "factoryrobot", "factorycctv"};
}

PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice, const std::string &pipelineName, const std::string &streamName) {
    std::string sourceName = streamName;
    if (ctrl_initialRequestRates.find(sourceName) == ctrl_initialRequestRates.end()) {
        for (auto [key, rates]: ctrl_initialRequestRates) {
            if (key.find(pipelineName) != std::string::npos) {
                sourceName = key;
                break;
            }
        }
    }
    switch (type) {
        case PipelineType::Traffic: {
            auto *datasource = new PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {"server"};
            }
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *yolov5n320 = nullptr;
            PipelineModel *yolov5n512 = nullptr;
            PipelineModel *yolov5s = nullptr;
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {"server"};

                yolov5n320 = new PipelineModel{
                        "server",
                        "yolov5n320",
                        ModelType::Yolov5n320,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n320->possibleDevices = {"server"};


                yolov5n512 = new PipelineModel{
                        "server",
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n512->possibleDevices = {"server"};

                yolov5s= new PipelineModel{
                        "server",
                        "yolov5s",
                        ModelType::Yolov5s,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5s->possibleDevices = {"server"};
            }

            auto *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    ModelType::Retinaface,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            retina1face->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({retina1face, 0});

            if (ctrl_systemName == "jlf") {
                retina1face->possibleDevices = {"server"};
                retina1face->upstreams = {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}};
                yolov5n320->downstreams.push_back({retina1face, 0});
                yolov5n512->downstreams.push_back({retina1face, 0});
                yolov5s->downstreams.push_back({retina1face, 0});
            }

            auto *arcface = new PipelineModel{
                    "server",
                    "arcface",
                    ModelType::Arcface,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            arcface->possibleDevices = {"server"};
            retina1face->downstreams.push_back({arcface, -1});

            auto *carbrand = new PipelineModel{
                    "server",
                    "carbrand",
                    ModelType::CarBrand,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}}
            };
            carbrand->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({carbrand, 2});

            if (ctrl_systemName == "jlf") {
                carbrand->possibleDevices = {"server"};
                carbrand->upstreams = {{yolov5n, 2}, {yolov5n320, 2}, {yolov5n512, 2}, {yolov5s, 2}};
                yolov5n320->downstreams.push_back({carbrand, 2});
                yolov5n512->downstreams.push_back({carbrand, 2});
                yolov5s->downstreams.push_back({carbrand, 2});
            }

            auto *platedet = new PipelineModel{
                    "server",
                    "platedet",
                    ModelType::PlateDet,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}}
            };
            platedet->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({platedet, 2});

            if (ctrl_systemName == "jlf") {
                platedet->possibleDevices = {"server"};
                platedet->upstreams = {{yolov5n, 2}, {yolov5n320, 2}, {yolov5n512, 2}, {yolov5s, 2}};
                yolov5n320->downstreams.push_back({platedet, 2});
                yolov5n512->downstreams.push_back({platedet, 2});
                yolov5s->downstreams.push_back({platedet, 2});
            }

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{arcface, -1}, {carbrand, -1}, {platedet, -1}}
            };
            sink->possibleDevices = {"sink"};
            arcface->downstreams.push_back({sink, -1});
            carbrand->downstreams.push_back({sink, -1});
            platedet->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
                arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][arcface->name];
                carbrand->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carbrand->name];
                platedet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][platedet->name];
            }

            if (ctrl_systemName == "jlf") {
                arcface->possibleDevices = {"server"};
                return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, retina1face, arcface, carbrand, platedet, sink};
            }
            return {datasource, yolov5n, retina1face, arcface, carbrand, platedet, sink};
        }
        case PipelineType::Indoor: {
            auto *datasource = new PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};
            auto *retinamtface = new PipelineModel{
                    startDevice,
                    "retinamtface",
                    ModelType::RetinaMtface,
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            retinamtface->possibleDevices = {startDevice};
            datasource->downstreams.push_back({retinamtface, -1});

            auto *emotionnet = new PipelineModel{
                    "server",
                    "emotionnet",
                    ModelType::Emotionnet,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retinamtface, -1}}
            };
            emotionnet->possibleDevices = {"server"};
            retinamtface->downstreams.push_back({emotionnet, -1});

            auto *age = new PipelineModel{
                    "server",
                    "age",
                    ModelType::Age,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retinamtface, -1}}
            };
            age->possibleDevices = {"server"};
            retinamtface->downstreams.push_back({age, -1});

            auto *gender = new PipelineModel{
                    "server",
                    "gender",
                    ModelType::Gender,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retinamtface, -1}}
            };
            gender->possibleDevices = {"server"};
            retinamtface->downstreams.push_back({gender, -1});

            auto *arcface = new PipelineModel{
                    "server",
                    "arcface",
                    ModelType::Arcface,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retinamtface, -1}}
            };
            arcface->possibleDevices = {"server"};
            retinamtface->downstreams.push_back({arcface, -1});

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{emotionnet, -1}, {age, -1}, {gender, -1}, {arcface, -1}}
            };
            sink->possibleDevices = {"sink"};
            emotionnet->downstreams.push_back({sink, -1});
            age->downstreams.push_back({sink, -1});
            gender->downstreams.push_back({sink, -1});
            arcface->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                retinamtface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retinamtface->name];
                emotionnet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][emotionnet->name];
                age->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][age->name];
                gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][gender->name];
                arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][arcface->name];
            }

            return {datasource, retinamtface, emotionnet, age, gender, arcface, sink};
        }
        case PipelineType::Building_Security: {
            auto *datasource = new PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {"server"};
            }
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *yolov5n320 = nullptr;
            PipelineModel *yolov5n512 = nullptr;
            PipelineModel *yolov5s = nullptr;
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {"server"};

                yolov5n320 = new PipelineModel{
                        "server",
                        "yolov5n320",
                        ModelType::Yolov5n320,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n320->possibleDevices = {"server"};


                yolov5n512 = new PipelineModel{
                        "server",
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n512->possibleDevices = {"server"};

                yolov5s = new PipelineModel{
                        "server",
                        "yolov5s",
                        ModelType::Yolov5s,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5s->possibleDevices = {"server"};
            }

            auto *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    ModelType::Retinaface,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            retina1face->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({retina1face, 0});

            if (ctrl_systemName == "jlf") {
                retina1face->possibleDevices = {"server"};
                retina1face->upstreams = {{yolov5n,    0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s,    0}};
                yolov5n320->downstreams.push_back({retina1face, 0});
                yolov5n512->downstreams.push_back({retina1face, 0});
                yolov5s->downstreams.push_back({retina1face, 0});
            }

            auto *movenet = new PipelineModel{
                    "server",
                    "movenet",
                    ModelType::Movenet,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            movenet->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({movenet, 0});

            if (ctrl_systemName == "jlf") {
                movenet->possibleDevices = {"server"};
                movenet->upstreams = {{yolov5n,    0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s,    0}};
                yolov5n320->downstreams.push_back({movenet, 0});
                yolov5n512->downstreams.push_back({movenet, 0});
                yolov5s->downstreams.push_back({movenet, 0});
            }

            auto *gender = new PipelineModel{
                    "server",
                    "gender",
                    ModelType::Gender,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            gender->possibleDevices = {startDevice, "server"};
            retina1face->downstreams.push_back({gender, -1});

            auto *age = new PipelineModel{
                    "server",
                    "age",
                    ModelType::Age,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            age->possibleDevices = {startDevice, "server"};
            retina1face->downstreams.push_back({age, -1});

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{gender, -1}, {age, -1}, {movenet, -1}}
            };
            sink->possibleDevices = {"sink"};
            gender->downstreams.push_back({sink, -1});
            age->downstreams.push_back({sink, -1});
            movenet->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
                movenet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][movenet->name];
                gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][movenet->name];
                age->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][age->name];
            }

            if (ctrl_systemName == "jlf") {
                gender->possibleDevices = {"server"};
                age->possibleDevices = {"server"};
                return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, retina1face, movenet, gender, age, sink};
            }
            return {datasource, yolov5n, retina1face, movenet, gender, age, sink};
        }
        case PipelineType::Surveillance_Robot: {
            auto *datasource = new PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {"server"};
            }
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *yolov5n320 = nullptr;
            PipelineModel *yolov5n512 = nullptr;
            PipelineModel *yolov5s = nullptr;
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {"server"};

                yolov5n320 = new PipelineModel{
                        "server",
                        "yolov5n320",
                        ModelType::Yolov5n320,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n320->possibleDevices = {"server"};


                yolov5n512 = new PipelineModel{
                        "server",
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n512->possibleDevices = {"server"};

                yolov5s= new PipelineModel{
                        "server",
                        "yolov5s",
                        ModelType::Yolov5s,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5s->possibleDevices = {"server"};
            }

            auto *arcface = new PipelineModel{
                    "server",
                    "arcface",
                    ModelType::Arcface,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            arcface->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({arcface, -1});
            if (ctrl_systemName == "jlf") {
                arcface->upstreams = {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}};
                yolov5n320->downstreams.push_back({arcface, 0});
                yolov5n512->downstreams.push_back({arcface, 0});
                yolov5s->downstreams.push_back({arcface, 0});
            }

            auto *age = new PipelineModel{
                    "server",
                    "age",
                    ModelType::Age,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            age->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({age, -1});
            if (ctrl_systemName == "jlf") {
                age->upstreams = {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}};
                yolov5n320->downstreams.push_back({age, 0});
                yolov5n512->downstreams.push_back({age, 0});
                yolov5s->downstreams.push_back({age, 0});
            }

            auto *gender = new PipelineModel{
                    "server",
                    "gender",
                    ModelType::Gender,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            gender->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({gender, -1});
            if (ctrl_systemName == "jlf") {
                gender->upstreams = {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}};
                yolov5n320->downstreams.push_back({gender, 0});
                yolov5n512->downstreams.push_back({gender, 0});
                yolov5s->downstreams.push_back({gender, 0});
            }

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{arcface, -1}, {age, -1}, {gender, -1}}
            };
            sink->possibleDevices = {"sink"};
            arcface->downstreams.push_back({sink, -1});
            age->downstreams.push_back({sink, -1});
            gender->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][arcface->name];
                age->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][age->name];
                gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][gender->name];
            }

            if (ctrl_systemName == "jlf") {
                arcface->possibleDevices = {"server"};
                return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, arcface, gender, age, sink};
            }
            return {datasource, yolov5n, arcface, gender, age, sink};
        }
        case PipelineType::Surveillance_Campus: {
            auto *datasource = new PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {"server"};
            }
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *yolov5n320 = nullptr;
            PipelineModel *yolov5n512 = nullptr;
            PipelineModel *yolov5s = nullptr;
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {"server"};

                yolov5n320 = new PipelineModel{
                        "server",
                        "yolov5n320",
                        ModelType::Yolov5n320,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n320->possibleDevices = {"server"};


                yolov5n512 = new PipelineModel{
                        "server",
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n512->possibleDevices = {"server"};

                yolov5s= new PipelineModel{
                        "server",
                        "yolov5s",
                        ModelType::Yolov5s,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5s->possibleDevices = {"server"};
            }

            auto *platedet = new PipelineModel{
                    "server",
                    "platedet",
                    ModelType::PlateDet,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}}
            };
            platedet->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({platedet, 2});
            if (ctrl_systemName == "jlf") {
                platedet->possibleDevices = {"server"};
                platedet->upstreams = {{yolov5n, 2},{yolov5n320, 2},{yolov5n512, 2},{yolov5s, 2}};
                yolov5n320->downstreams.push_back({platedet, 2});
                yolov5n512->downstreams.push_back({platedet, 2});
                yolov5s->downstreams.push_back({platedet, 2});
            }

            auto *movenet = new PipelineModel{
                    "server",
                    "movenet",
                    ModelType::Movenet,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            movenet->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({movenet, 0});

            if (ctrl_systemName == "jlf") {
                movenet->possibleDevices = {"server"};
                movenet->upstreams = {{yolov5n,    0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s,    0}};
                yolov5n320->downstreams.push_back({movenet, 0});
                yolov5n512->downstreams.push_back({movenet, 0});
                yolov5s->downstreams.push_back({movenet, 0});
            }

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{platedet, -1}, {movenet, -1}}
            };
            sink->possibleDevices = {"sink"};
            platedet->downstreams.push_back({sink, -1});
            movenet->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                platedet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][platedet->name];
                movenet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][movenet->name];
            }

            if (ctrl_systemName == "jlf") {
                return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, platedet, movenet, sink};
            }
            return {datasource, yolov5n, platedet, movenet, sink};
        }
        case PipelineType::Factory_Robot: {
            auto *datasource = new PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {"server"};
            }
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *yolov5n320 = nullptr;
            PipelineModel *yolov5n512 = nullptr;
            PipelineModel *yolov5s = nullptr;
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {"server"};

                yolov5n320 = new PipelineModel{
                        "server",
                        "yolov5n320",
                        ModelType::Yolov5n320,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n320->possibleDevices = {"server"};


                yolov5n512 = new PipelineModel{
                        "server",
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n512->possibleDevices = {"server"};

                yolov5s= new PipelineModel{
                        "server",
                        "yolov5s",
                        ModelType::Yolov5s,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5s->possibleDevices = {"server"};
            }

            auto *labledet = new PipelineModel{
                    "server",
                    "platedet",
                    ModelType::PlateDet,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 2}}
            };
            labledet->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({labledet, 2});
            if (ctrl_systemName == "jlf") {
                labledet->possibleDevices = {"server"};
                labledet->upstreams = {{yolov5n, 2},{yolov5n320, 2},{yolov5n512, 2},{yolov5s, 2}};
                yolov5n320->downstreams.push_back({labledet, 2});
                yolov5n512->downstreams.push_back({labledet, 2});
                yolov5s->downstreams.push_back({labledet, 2});
            }

            auto *obstacle = new PipelineModel{
                    "server",
                    "carbrand",
                    ModelType::CarBrand,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, -1}}
            };
            obstacle->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({obstacle, -1});
            if (ctrl_systemName == "jlf") {
                obstacle->upstreams = {{yolov5n, -1}, {yolov5n320, -1}, {yolov5n512, -1}, {yolov5s, -1}};
                yolov5n320->downstreams.push_back({obstacle, -1});
                yolov5n512->downstreams.push_back({obstacle, -1});
                yolov5s->downstreams.push_back({obstacle, -1});
            }

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{labledet, -1}, {obstacle, -1}}
            };
            sink->possibleDevices = {"sink"};
            labledet->downstreams.push_back({sink, -1});
            obstacle->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                labledet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][labledet->name];
                obstacle->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][obstacle->name];
            }

            if (ctrl_systemName == "jlf") {
                return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, labledet, obstacle, sink};
            }
            return {datasource, yolov5n, labledet, obstacle, sink};
        }
        case PipelineType::Factory_CCTV: {
            auto *datasource = new PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *yolov5n = new PipelineModel{
                    "server",
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    true,
                    {},
                    {},
                    {},
                    {{datasource, -1}}
            };
            yolov5n->possibleDevices = {startDevice, "server"};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {"server"};
            }
            datasource->downstreams.push_back({yolov5n, -1});

            PipelineModel *yolov5n320 = nullptr;
            PipelineModel *yolov5n512 = nullptr;
            PipelineModel *yolov5s = nullptr;
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {"server"};

                yolov5n320 = new PipelineModel{
                        "server",
                        "yolov5n320",
                        ModelType::Yolov5n320,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n320->possibleDevices = {"server"};


                yolov5n512 = new PipelineModel{
                        "server",
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5n512->possibleDevices = {"server"};

                yolov5s= new PipelineModel{
                        "server",
                        "yolov5s",
                        ModelType::Yolov5s,
                        {},
                        true,
                        {},
                        {},
                        {},
                        {{datasource, -1}}
                };
                yolov5s->possibleDevices = {"server"};
            }

            auto *retina1face = new PipelineModel{
                    "server",
                    "retina1face",
                    ModelType::Retinaface,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            retina1face->possibleDevices = {"server"};
            yolov5n->downstreams.push_back({retina1face, 0});

            if (ctrl_systemName == "jlf") {
                retina1face->possibleDevices = {"server"};
                retina1face->upstreams = {{yolov5n, 0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s, 0}};
                yolov5n320->downstreams.push_back({retina1face, 0});
                yolov5n512->downstreams.push_back({retina1face, 0});
                yolov5s->downstreams.push_back({retina1face, 0});
            }

            auto *arcface = new PipelineModel{
                    "server",
                    "arcface",
                    ModelType::Arcface,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{retina1face, -1}}
            };
            arcface->possibleDevices = {"server"};
            retina1face->downstreams.push_back({arcface, -1});

            auto *activity = new PipelineModel{
                    "server",
                    "movenet",
                    ModelType::Movenet,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{yolov5n, 0}}
            };
            activity->possibleDevices = {startDevice, "server"};
            yolov5n->downstreams.push_back({activity, 0});

            if (ctrl_systemName == "jlf") {
                activity->possibleDevices = {"server"};
                activity->upstreams = {{yolov5n,    0}, {yolov5n320, 0}, {yolov5n512, 0}, {yolov5s,    0}};
                yolov5n320->downstreams.push_back({activity, 0});
                yolov5n512->downstreams.push_back({activity, 0});
                yolov5s->downstreams.push_back({activity, 0});
            }

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{arcface, -1}, {activity, -1}}
            };
            sink->possibleDevices = {"sink"};
            arcface->downstreams.push_back({sink, -1});
            activity->downstreams.push_back({sink, -1});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][arcface->name];
                activity->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][activity->name];
            }

            if (ctrl_systemName == "jlf") {
                arcface->possibleDevices = {"server"};
                return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, retina1face, arcface, activity, sink};
            }
            return {datasource, yolov5n, retina1face, arcface, activity, sink};
        }
        case PipelineType::Smart_Glasses: {
            auto *datasource = new PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, true, {}, {}};
            datasource->possibleDevices = {startDevice};

            auto *vlm = new PipelineModel{};
            vlm->possibleDevices = {"server"};

            auto *sink = new PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    false,
                    {},
                    {},
                    {},
                    {{vlm, -1}}
            };
            sink->possibleDevices = {"sink"};
            vlm->upstreams = {{sink, -1}};

            return {datasource, vlm, sink};
        }
        default:
            return {};
    }
}

void Controller::readInitialObjectCount(const std::string &path) {
    std::ifstream file(path);
    json j = json::parse(file);
    std::map<std::string, std::map<std::string, std::map<int, float>>> initialPerSecondRate;
    for (auto &item: j.items()) {
        std::string streamName = item.key();
        initialPerSecondRate[streamName] = {};
        for (auto &object: item.value().items()) {
            std::string objectName = object.key();
            initialPerSecondRate[streamName][objectName] = {};
            std::vector<int> perFrameObjCount = object.value().get<std::vector<int>>();
            int numFrames = perFrameObjCount.size();
            int totalNumObjs = 0;
            for (auto i = 0; i < numFrames; i++) {
                totalNumObjs += perFrameObjCount[i];
                if ((i + 1) % 30 != 0) {
                    continue;
                }
                int seconds = (i + 1) / 30;
                initialPerSecondRate[streamName][objectName][seconds] = totalNumObjs * 1.f / seconds;
            }
        }
        float skipRate = ctrl_systemFPS / 30.f;
        std::map<std::string, float> *stream = &(ctrl_initialRequestRates[streamName]);
        float maxPersonRate = 1.2 * std::max_element(
                initialPerSecondRate[streamName]["person"].begin(),
                initialPerSecondRate[streamName]["person"].end()
        )->second * skipRate;
        maxPersonRate = std::max(maxPersonRate, ctrl_systemFPS * 1.f);
        float maxCarRate = 1.2 * std::max_element(
                initialPerSecondRate[streamName]["car"].begin(),
                initialPerSecondRate[streamName]["car"].end()
        )->second * skipRate;
        maxCarRate = std::max(maxCarRate, ctrl_systemFPS * 1.f);
        if (streamName.find("traffic") != std::string::npos) {
            stream->insert({"yolov5n", ctrl_systemFPS});
            stream->insert({"retina1face", std::ceil(maxPersonRate)});
            stream->insert({"movenet", std::ceil(maxPersonRate)});
            stream->insert({"arcface", std::ceil(maxPersonRate * 0.6)});
            stream->insert({"carbrand", std::ceil(maxCarRate)});
            stream->insert({"platedet", std::ceil(maxCarRate)});
        } else if (streamName.find("campus") != std::string::npos) {
            stream->insert({"yolov5n", ctrl_systemFPS});
            stream->insert({"retina1face", std::ceil(maxPersonRate)});
            stream->insert({"movenet", std::ceil(maxPersonRate)});
            stream->insert({"arcface", std::ceil(maxPersonRate * 0.6)});
            stream->insert({"carbrand", std::ceil(maxCarRate)});
            stream->insert({"platedet", std::ceil(maxCarRate)});
        } else if (streamName.find("people") != std::string::npos) {
            stream->insert({"yolov5n", ctrl_systemFPS});
            stream->insert({"retina1face", std::ceil(maxPersonRate)});
            stream->insert({"age", std::ceil(maxPersonRate) * 0.6});
            stream->insert({"gender", std::ceil(maxPersonRate) * 0.6});
            stream->insert({"movenet", std::ceil(maxPersonRate)});
            stream->insert({"platedet", std::ceil(maxCarRate)});
        } else if (streamName.find("indoor") != std::string::npos) {
            stream->insert({"retinamtface", ctrl_systemFPS});
            stream->insert({"arcface", std::ceil(maxPersonRate)});
            stream->insert({"age", std::ceil(maxPersonRate)});
            stream->insert({"gender", std::ceil(maxPersonRate)});
            stream->insert({"emotionnet", std::ceil(maxPersonRate)});
        } else if (streamName.find("surveillance_robot") != std::string::npos) {
            stream->insert({"yolov5n", ctrl_systemFPS});
            stream->insert({"arcface", std::ceil(maxPersonRate)});
            stream->insert({"age", std::ceil(maxPersonRate)});
            stream->insert({"gender", std::ceil(maxPersonRate)});
        } else if (streamName.find("factory_robot") != std::string::npos) {
            stream->insert({"yolov5n", ctrl_systemFPS});
            stream->insert({"platedet", std::ceil(maxCarRate)});
            stream->insert({"carbrand", std::ceil(maxCarRate + maxPersonRate)});
        } else if (streamName.find("factory") != std::string::npos) {
            stream->insert({"yolov5n", ctrl_systemFPS});
            stream->insert({"retina1face", std::ceil(maxPersonRate)});
            stream->insert({"movenet", std::ceil(maxPersonRate)});
            stream->insert({"arcface", std::ceil(maxPersonRate * 0.6)});
        }
    }
}

PipelineModelListType deepCopyPipelineModelList(const PipelineModelListType& original) {
    PipelineModelListType newList;
    newList.reserve(original.size());
    for (const auto* model : original) {
        newList.push_back(new PipelineModel(*model));
    }
    return newList;
}

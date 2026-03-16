#include "controller.h"

std::vector<std::string> Controller::getPipelineNames() {
    return {"traffic", "people", "indoor", "surveillancerobot", "campusdrone", "factoryrobot", "factorycctv"};
}

PipelineModelListType Controller::getModelsByPipelineType(PipelineType type, const std::string &startDevice, 
                                                          const std::string &pipelineName, const std::string &streamName, 
                                                          const std::string &edgeNode) {
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
            // FIX: Use std::make_shared and PipelineModel constructor directly
            auto datasource = std::make_shared<PipelineModel>(PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, 0, true});
            datasource->possibleDevices = {startDevice};

            auto yolov5n = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    1,
                    true,
                    false,
                    {},
                    {},
                    {},
                    // FIX: Replaced std::pair syntax with PipelineEdge struct, injecting streamName for routing
                    {PipelineEdge{datasource, -1, {streamName}}}
            });
            yolov5n->possibleDevices = {startDevice, edgeNode};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {edgeNode};
            }
            // FIX: Replaced std::pair syntax with PipelineEdge
            datasource->downstreams.push_back(PipelineEdge{yolov5n, -1, {streamName}});

            // std::shared_ptr<PipelineModel> yolov5n320 = nullptr;
            std::shared_ptr<PipelineModel> yolov5n512 = nullptr;
            // std::shared_ptr<PipelineModel> yolov5s = nullptr;
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {edgeNode};

                // yolov5n320 = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5n320",
                //         ModelType::Yolov5n320,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5n320->possibleDevices = {edgeNode};
                
                yolov5n512 = std::make_shared<PipelineModel>(PipelineModel{
                        edgeNode,
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        1,
                        true,
                        false,
                        {},
                        {},
                        {},
                        {PipelineEdge{datasource, -1, {streamName}}}
                });
                yolov5n512->possibleDevices = {edgeNode};

                // yolov5s = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5s",
                //         ModelType::Yolov5s,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5s->possibleDevices = {edgeNode};

                // // Add downstream connections for the additional YOLO variants in the JLF pipeline
                // datasource->downstreams.push_back(PipelineEdge{yolov5n320, -1, {streamName}});
                datasource->downstreams.push_back(PipelineEdge{yolov5n512, -1, {streamName}});
                // datasource->downstreams.push_back(PipelineEdge{yolov5s, -1, {streamName}});
            }

            auto fashioncolor = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "fashioncolor",
                    ModelType::FashionColor,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, 0, {streamName}}}
            });
            fashioncolor->possibleDevices = { edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});

            if (ctrl_systemName == "jlf") {
                fashioncolor->possibleDevices = {edgeNode};
                fashioncolor->upstreams = {
                    PipelineEdge{yolov5n, 0, {streamName}}, 
                    // PipelineEdge{yolov5n320, 0, {streamName}}, 
                    PipelineEdge{yolov5n512, 0, {streamName}}, 
                    // PipelineEdge{yolov5s, 0, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
            }

            auto carcolor = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "carcolor",
                    ModelType::CarColor,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, -1, {streamName}}}
            });
            carcolor->possibleDevices = {edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{carcolor, -1, {streamName}});

            if (ctrl_systemName == "jlf") {
                carcolor->possibleDevices = {edgeNode};
                carcolor->upstreams = {
                    PipelineEdge{yolov5n, 2, {streamName}}, 
                    // PipelineEdge{yolov5n320, 2, {streamName}}, 
                    PipelineEdge{yolov5n512, 2, {streamName}}, 
                    // PipelineEdge{yolov5s, 2, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{carcolor, 2, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{carcolor, 2, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{carcolor, 2, {streamName}});
            }

            auto carbrand = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "carbrand",
                    ModelType::CarBrand,
                    {},
                    3,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{carcolor, -1, {streamName}}}
            });
            carbrand->possibleDevices = {edgeNode};
            carcolor->downstreams.push_back(PipelineEdge{carbrand, -1, {streamName}});

            auto sink = std::make_shared<PipelineModel>(PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    0,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{fashioncolor, -1, {streamName}}, PipelineEdge{carbrand, -1, {streamName}}}
            });
            sink->possibleDevices = {"sink"};
            fashioncolor->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            carbrand->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});

            // if (sourceName.find("traffic1") != std::string::npos) {
            //     auto retina1face = std::make_shared<PipelineModel>(PipelineModel{
            //         edgeNode,
            //         "retina1face",
            //         ModelType::Retinaface,
            //         {},
            //         2,
            //         false,
            //         false,
            //         {},
            //         {},
            //         {},
            //         {PipelineEdge{fashioncolor, -1, {streamName}}}
            //     });
            //     retina1face->possibleDevices = {edgeNode};
            //     fashioncolor->downstreams.push_back(PipelineEdge{retina1face, -1, {streamName}});
            //     retina1face->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            //     sink->upstreams.push_back(PipelineEdge{retina1face, -1, {streamName}});

            //     if (!sourceName.empty()) {
            //         yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
            //         fashioncolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][fashioncolor->name];
            //         carbrand->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carbrand->name];
            //         carcolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carcolor->name];
            //         retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
            //     }
            //     return {datasource, yolov5n, fashioncolor, retina1face, carcolor, carbrand, sink};
            // }

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                fashioncolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][fashioncolor->name];
                carbrand->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carbrand->name];
                carcolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carcolor->name];
            }

            if (ctrl_systemName == "jlf") {
                fashioncolor->possibleDevices = {edgeNode};
                carcolor->possibleDevices = {edgeNode};
                carbrand->possibleDevices = {edgeNode};
                // return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, fashioncolor, carcolor, carbrand, sink};
                return {datasource, yolov5n, yolov5n512, fashioncolor, carcolor, carbrand, sink};
            }
            return {datasource, yolov5n, fashioncolor, carcolor, carbrand, sink};
        }
        case PipelineType::Indoor: {
            auto datasource = std::make_shared<PipelineModel>(PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, 0, true});
            datasource->possibleDevices = {startDevice};
            
            auto retinamtface = std::make_shared<PipelineModel>(PipelineModel{
                    startDevice,
                    "retinamtface",
                    ModelType::RetinaMtface,
                    {},
                    1,
                    true,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{datasource, -1, {streamName}}}
            });
            retinamtface->possibleDevices = {startDevice};
            datasource->downstreams.push_back(PipelineEdge{retinamtface, -1, {streamName}});

            auto emotionnet = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "emotionnet",
                    ModelType::Emotionnet,
                    {},
                    1,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{retinamtface, -1, {streamName}}}
            });
            emotionnet->possibleDevices = {edgeNode};
            retinamtface->downstreams.push_back(PipelineEdge{emotionnet, -1, {streamName}});

            auto age = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "age",
                    ModelType::Age,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{retinamtface, -1, {streamName}}}
            });
            age->possibleDevices = {edgeNode};
            retinamtface->downstreams.push_back(PipelineEdge{age, -1, {streamName}});

            auto gender = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "gender",
                    ModelType::Gender,
                    {},
                    3,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{retinamtface, -1, {streamName}}}
            });
            gender->possibleDevices = {edgeNode};
            retinamtface->downstreams.push_back(PipelineEdge{gender, -1, {streamName}});

            auto arcface = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "arcface",
                    ModelType::Arcface,
                    {},
                    4,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{retinamtface, -1, {streamName}}}
            });
            arcface->possibleDevices = {edgeNode};
            retinamtface->downstreams.push_back(PipelineEdge{arcface, -1, {streamName}});

            auto sink = std::make_shared<PipelineModel>(PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    0,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {
                        PipelineEdge{emotionnet, -1, {streamName}}, 
                        PipelineEdge{age, -1, {streamName}}, 
                        PipelineEdge{gender, -1, {streamName}}, 
                        PipelineEdge{arcface, -1, {streamName}}
                    }
            });
            sink->possibleDevices = {"sink"};
            emotionnet->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            age->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            gender->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            arcface->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});

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
            auto datasource = std::make_shared<PipelineModel>(PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, 0, true});
            datasource->possibleDevices = {startDevice};

            auto yolov5n = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    1,
                    true,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{datasource, -1, {streamName}}}
            });
            yolov5n->possibleDevices = {startDevice, edgeNode};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {edgeNode};
            }
            datasource->downstreams.push_back(PipelineEdge{yolov5n, -1, {streamName}});

            // std::shared_ptr<PipelineModel> yolov5n320 = nullptr;
            std::shared_ptr<PipelineModel> yolov5n512 = nullptr;
            // std::shared_ptr<PipelineModel> yolov5s = nullptr;
            
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {edgeNode};

                // yolov5n320 = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5n320",
                //         ModelType::Yolov5n320,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5n320->possibleDevices = {edgeNode};


                yolov5n512 = std::make_shared<PipelineModel>(PipelineModel{
                        edgeNode,
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        1,
                        true,
                        false,
                        {},
                        {},
                        {},
                        {PipelineEdge{datasource, -1, {streamName}}}
                });
                yolov5n512->possibleDevices = {edgeNode};

                // yolov5s = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5s",
                //         ModelType::Yolov5s,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5s->possibleDevices = {edgeNode};

                // Add downstream connections for the additional YOLO variants in the JLF pipeline
                // datasource->downstreams.push_back(PipelineEdge{yolov5n320, -1, {streamName}});
                datasource->downstreams.push_back(PipelineEdge{yolov5n512, -1, {streamName}});
                // datasource->downstreams.push_back(PipelineEdge{yolov5s, -1, {streamName}});
            }

            auto retina1face = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "retina1face",
                    ModelType::Retinaface,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, 0, {streamName}}}
            });
            retina1face->possibleDevices = {edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{retina1face, 0, {streamName}});

            if (ctrl_systemName == "jlf") {
                retina1face->possibleDevices = {edgeNode};
                retina1face->upstreams = {
                    PipelineEdge{yolov5n, 0, {streamName}}, 
                    // PipelineEdge{yolov5n320, 0, {streamName}}, 
                    PipelineEdge{yolov5n512, 0, {streamName}}, 
                    // PipelineEdge{yolov5s, 0, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{retina1face, 0, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{retina1face, 0, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{retina1face, 0, {streamName}});
            }

            auto gender = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "gender",
                    ModelType::Gender,
                    {},
                    3,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{retina1face, -1, {streamName}}}
            });
            gender->possibleDevices = {startDevice, edgeNode};
            retina1face->downstreams.push_back(PipelineEdge{gender, -1, {streamName}});

            auto emotionnet = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "emotionnet",
                    ModelType::Emotionnet,
                    {},
                    3,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{retina1face, -1, {streamName}}}
            });
            emotionnet->possibleDevices = {edgeNode};
            retina1face->downstreams.push_back(PipelineEdge{emotionnet, -1, {streamName}});

            auto arcface = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "arcface",
                    ModelType::Arcface,
                    {},
                    3,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{retina1face, -1, {streamName}}}
            });
            arcface->possibleDevices = {edgeNode};
            retina1face->downstreams.push_back(PipelineEdge{arcface, -1, {streamName}});

            auto fashioncolor = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "fashioncolor",
                    ModelType::FashionColor,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, 0, {streamName}}}
            });
            fashioncolor->possibleDevices = { edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});

            if (ctrl_systemName == "jlf") {
                fashioncolor->possibleDevices = {edgeNode};
                fashioncolor->upstreams = {
                    PipelineEdge{yolov5n, 0, {streamName}}, 
                    // PipelineEdge{yolov5n320, 0, {streamName}}, 
                    PipelineEdge{yolov5n512, 0, {streamName}}, 
                    // PipelineEdge{yolov5s, 0, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
            }

            auto sink = std::make_shared<PipelineModel>(PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    0,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {
                        PipelineEdge{gender, -1, {streamName}}, 
                        PipelineEdge{emotionnet, -1, {streamName}}, 
                        PipelineEdge{arcface, -1, {streamName}}, 
                        PipelineEdge{fashioncolor, -1, {streamName}}
                    }
            });
            sink->possibleDevices = {"sink"};
            gender->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            emotionnet->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            arcface->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            fashioncolor->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                retina1face->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][retina1face->name];
                gender->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][gender->name];
                emotionnet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][emotionnet->name];
                arcface->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][arcface->name];
                fashioncolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][fashioncolor->name];
            }

            if (ctrl_systemName == "jlf") {
                retina1face->possibleDevices = {edgeNode};
                gender->possibleDevices = {edgeNode};
                emotionnet->possibleDevices = {edgeNode};
                arcface->possibleDevices = {edgeNode};
                fashioncolor->possibleDevices = {edgeNode};
                // return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, retina1face, emotionnet, gender, arcface, fashioncolor, sink};
                return {datasource, yolov5n, yolov5n512, retina1face, emotionnet, gender, arcface, fashioncolor, sink};
            }
            return {datasource, yolov5n, retina1face, emotionnet, gender, arcface, fashioncolor, sink};
        }
        case PipelineType::Surveillance_Robot: {
            auto datasource = std::make_shared<PipelineModel>(PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, 0, true});
            datasource->possibleDevices = {startDevice};

            auto yolov5n = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    1,
                    true,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{datasource, -1, {streamName}}}
            });
            yolov5n->possibleDevices = {startDevice, edgeNode};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {edgeNode};
            }
            datasource->downstreams.push_back(PipelineEdge{yolov5n, -1, {streamName}});

            // std::shared_ptr<PipelineModel> yolov5n320 = nullptr;
            std::shared_ptr<PipelineModel> yolov5n512 = nullptr;
            // std::shared_ptr<PipelineModel> yolov5s = nullptr;
            
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {edgeNode};

                // yolov5n320 = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5n320",
                //         ModelType::Yolov5n320,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5n320->possibleDevices = {edgeNode};


                yolov5n512 = std::make_shared<PipelineModel>(PipelineModel{
                        edgeNode,
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        1,
                        true,
                        false,
                        {},
                        {},
                        {},
                        {PipelineEdge{datasource, -1, {streamName}}}
                });
                yolov5n512->possibleDevices = {edgeNode};

                // yolov5s = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5s",
                //         ModelType::Yolov5s,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5s->possibleDevices = {edgeNode};

                // Add downstream connections for the additional YOLO variants in the JLF pipeline
                // datasource->downstreams.push_back(PipelineEdge{yolov5n320, -1, {streamName}});
                datasource->downstreams.push_back(PipelineEdge{yolov5n512, -1, {streamName}});
                // datasource->downstreams.push_back(PipelineEdge{yolov5s, -1, {streamName}});
            }

            auto fashioncolor = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "fashioncolor",
                    ModelType::FashionColor,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, 0, {streamName}}}
            });
            fashioncolor->possibleDevices = { edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});

            if (ctrl_systemName == "jlf") {
                fashioncolor->possibleDevices = {edgeNode};
                fashioncolor->upstreams = {
                    PipelineEdge{yolov5n, 0, {streamName}}, 
                    // PipelineEdge{yolov5n320, 0, {streamName}}, 
                    PipelineEdge{yolov5n512, 0, {streamName}}, 
                    // PipelineEdge{yolov5s, 0, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
            }

            auto carcolor = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "carcolor",
                    ModelType::CarColor,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, -1, {streamName}}}
            });
            carcolor->possibleDevices = {edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{carcolor, -1, {streamName}});

            if (ctrl_systemName == "jlf") {
                carcolor->possibleDevices = {edgeNode};
                carcolor->upstreams = {
                    PipelineEdge{yolov5n, 2, {streamName}}, 
                    // PipelineEdge{yolov5n320, 2, {streamName}}, 
                    PipelineEdge{yolov5n512, 2, {streamName}}, 
                    // PipelineEdge{yolov5s, 2, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{carcolor, 2, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{carcolor, 2, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{carcolor, 2, {streamName}});
            }

            auto carbrand = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "carbrand",
                    ModelType::CarBrand,
                    {},
                    3,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{carcolor, -1, {streamName}}}
            });
            carbrand->possibleDevices = {edgeNode};
            carcolor->downstreams.push_back(PipelineEdge{carbrand, -1, {streamName}});

            auto sink = std::make_shared<PipelineModel>(PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    0,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{fashioncolor, -1, {streamName}}, PipelineEdge{carbrand, -1, {streamName}}}
            });
            sink->possibleDevices = {"sink"};
            fashioncolor->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            carbrand->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                fashioncolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][fashioncolor->name];
                carbrand->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carbrand->name];
                carcolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carcolor->name];
            }

            if (ctrl_systemName == "jlf") {
                fashioncolor->possibleDevices = {edgeNode};
                carcolor->possibleDevices = {edgeNode};
                carbrand->possibleDevices = {edgeNode};
                // return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, fashioncolor, carcolor, carbrand, sink};
                    return {datasource, yolov5n, yolov5n512, fashioncolor, carcolor, carbrand, sink};
            }
            return {datasource, yolov5n, fashioncolor, carcolor, carbrand, sink};
        }
        case PipelineType::Surveillance_Campus: {
            auto datasource = std::make_shared<PipelineModel>(PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, 0, true});
            datasource->possibleDevices = {startDevice};

            auto yolov5n = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    1,
                    true,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{datasource, -1, {streamName}}}
            });
            yolov5n->possibleDevices = {startDevice, edgeNode};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {edgeNode};
            }
            datasource->downstreams.push_back(PipelineEdge{yolov5n, -1, {streamName}});

            // std::shared_ptr<PipelineModel> yolov5n320 = nullptr;
            std::shared_ptr<PipelineModel> yolov5n512 = nullptr;
            // std::shared_ptr<PipelineModel> yolov5s = nullptr;
            
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {edgeNode};

                // yolov5n320 = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5n320",
                //         ModelType::Yolov5n320,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5n320->possibleDevices = {edgeNode};


                yolov5n512 = std::make_shared<PipelineModel>(PipelineModel{
                        edgeNode,
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        1,
                        true,
                        false,
                        {},
                        {},
                        {},
                        {PipelineEdge{datasource, -1, {streamName}}}
                });
                yolov5n512->possibleDevices = {edgeNode};

                // yolov5s = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5s",
                //         ModelType::Yolov5s,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5s->possibleDevices = {edgeNode};

                // Add downstream connections for the additional YOLO variants in the JLF pipeline
                // datasource->downstreams.push_back(PipelineEdge{yolov5n320, -1, {streamName}});
                datasource->downstreams.push_back(PipelineEdge{yolov5n512, -1, {streamName}});
                // datasource->downstreams.push_back(PipelineEdge{yolov5s, -1, {streamName}});
            }

            auto fashioncolor = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "fashioncolor",
                    ModelType::FashionColor,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, 0, {streamName}}}
            });
            fashioncolor->possibleDevices = { edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});

            if (ctrl_systemName == "jlf") {
                fashioncolor->possibleDevices = {edgeNode};
                fashioncolor->upstreams = {
                    PipelineEdge{yolov5n, 0, {streamName}}, 
                    // PipelineEdge{yolov5n320, 0, {streamName}}, 
                    PipelineEdge{yolov5n512, 0, {streamName}}, 
                    // PipelineEdge{yolov5s, 0, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
            }

            auto carcolor = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "carcolor",
                    ModelType::CarColor,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, -1, {streamName}}}
            });
            carcolor->possibleDevices = {edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{carcolor, -1, {streamName}});

            if (ctrl_systemName == "jlf") {
                carcolor->possibleDevices = {edgeNode};
                carcolor->upstreams = {
                    PipelineEdge{yolov5n, 2, {streamName}}, 
                    // PipelineEdge{yolov5n320, 2, {streamName}}, 
                    PipelineEdge{yolov5n512, 2, {streamName}}, 
                    // PipelineEdge{yolov5s, 2, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{carcolor, 2, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{carcolor, 2, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{carcolor, 2, {streamName}});
            }

            auto carbrand = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "carbrand",
                    ModelType::CarBrand,
                    {},
                    3,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{carcolor, -1, {streamName}}}
            });
            carbrand->possibleDevices = {edgeNode};
            carcolor->downstreams.push_back(PipelineEdge{carbrand, -1, {streamName}});

            auto sink = std::make_shared<PipelineModel>(PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    0,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{fashioncolor, -1, {streamName}}, PipelineEdge{carbrand, -1, {streamName}}}
            });
            sink->possibleDevices = {"sink"};
            fashioncolor->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            carbrand->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});

            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                fashioncolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][fashioncolor->name];
                carbrand->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carbrand->name];
                carcolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][carcolor->name];
            }

            if (ctrl_systemName == "jlf") {
                fashioncolor->possibleDevices = {edgeNode};
                carcolor->possibleDevices = {edgeNode};
                carbrand->possibleDevices = {edgeNode};
                // return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, fashioncolor, carcolor, carbrand, sink};
                return {datasource, yolov5n, yolov5n512, fashioncolor, carcolor, carbrand, sink};
            }
            return {datasource, yolov5n, fashioncolor, carcolor, carbrand, sink};
        }
        case PipelineType::Factory_Robot: {
            auto datasource = std::make_shared<PipelineModel>(PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, 0, true});
            datasource->possibleDevices = {startDevice};

            auto equipmentdet = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "equipmentdet",
                    ModelType::EquipDetect,
                    {},
                    1,
                    true,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{datasource, -1, {streamName}}}
            });
            equipmentdet->possibleDevices = {startDevice, edgeNode};
            datasource->downstreams.push_back(PipelineEdge{equipmentdet, -1, {streamName}});

            auto yolov5n = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    1,
                    true,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{datasource, -1, {streamName}}}
            });
            yolov5n->possibleDevices = {startDevice, edgeNode};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {edgeNode};
            }
            datasource->downstreams.push_back(PipelineEdge{yolov5n, -1, {streamName}});

            // std::shared_ptr<PipelineModel> yolov5n320 = nullptr;
            std::shared_ptr<PipelineModel> yolov5n512 = nullptr;
            // std::shared_ptr<PipelineModel> yolov5s = nullptr;
            
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {edgeNode};

                // yolov5n320 = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5n320",
                //         ModelType::Yolov5n320,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5n320->possibleDevices = {edgeNode};


                yolov5n512 = std::make_shared<PipelineModel>(PipelineModel{
                        edgeNode,
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        1,
                        true,
                        false,
                        {},
                        {},
                        {},
                        {PipelineEdge{datasource, -1, {streamName}}}
                });
                yolov5n512->possibleDevices = {edgeNode};

                // yolov5s = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5s",
                //         ModelType::Yolov5s,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5s->possibleDevices = {edgeNode};

                // Add downstream connections for the additional YOLO variants in the JLF pipeline
                // datasource->downstreams.push_back(PipelineEdge{yolov5n320, -1, {streamName}});
                datasource->downstreams.push_back(PipelineEdge{yolov5n512, -1, {streamName}});
                // datasource->downstreams.push_back(PipelineEdge{yolov5s, -1, {streamName}});
            }

            auto geardet = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "geardet",
                    ModelType::GearDetect,
                    {},
                    1,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, 0, {streamName}}}
            });
            geardet->possibleDevices = {startDevice, edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{geardet, 0, {streamName}});
            
            if (ctrl_systemName == "jlf") {
                geardet->possibleDevices = {edgeNode};
                geardet->upstreams = {
                    PipelineEdge{yolov5n, 0, {streamName}},
                    // PipelineEdge{yolov5n320, 0, {streamName}},
                    PipelineEdge{yolov5n512, 0, {streamName}},
                    // PipelineEdge{yolov5s, 0, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{geardet, 0, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{geardet, 0, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{geardet, 0, {streamName}});
            }

            auto fashioncolor = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "fashioncolor",
                    ModelType::FashionColor,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, 0, {streamName}}}
            });
            fashioncolor->possibleDevices = {edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{fashioncolor, -1, {streamName}});
            
            if (ctrl_systemName == "jlf") {
                fashioncolor->upstreams = {
                    PipelineEdge{yolov5n, 0, {streamName}}, 
                    // PipelineEdge{yolov5n320, 0, {streamName}}, 
                    PipelineEdge{yolov5n512, 0, {streamName}}, 
                    // PipelineEdge{yolov5s, 0, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
            }

            auto sink = std::make_shared<PipelineModel>(PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    0,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {
                        PipelineEdge{equipmentdet, -1, {streamName}}, 
                        PipelineEdge{geardet, -1, {streamName}}, 
                        PipelineEdge{fashioncolor, -1, {streamName}}
                    }
            });
            sink->possibleDevices = {"sink"};
            equipmentdet->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            geardet->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            fashioncolor->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            
            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                equipmentdet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][equipmentdet->name];
                geardet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][geardet->name];
                fashioncolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][fashioncolor->name];
            }

            if (ctrl_systemName == "jlf") {
                equipmentdet->possibleDevices = {edgeNode};
                geardet->possibleDevices = {edgeNode};
                fashioncolor->possibleDevices = {edgeNode};
                // return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, equipmentdet, geardet, fashioncolor, sink};
                return {datasource, yolov5n, yolov5n512, equipmentdet, geardet, fashioncolor, sink};
            }
            return {datasource, yolov5n, equipmentdet, geardet, fashioncolor, sink};
        }
        case PipelineType::Factory_CCTV: {
            auto datasource = std::make_shared<PipelineModel>(PipelineModel{startDevice, "datasource", ModelType::DataSource, {}, 0, true});
            datasource->possibleDevices = {startDevice};

            auto equipmentdet = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "equipmentdet",
                    ModelType::EquipDetect,
                    {},
                    1,
                    true,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{datasource, -1, {streamName}}}
            });
            equipmentdet->possibleDevices = {startDevice, edgeNode};
            datasource->downstreams.push_back(PipelineEdge{equipmentdet, -1, {streamName}});

            auto yolov5n = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "yolov5n",
                    ModelType::Yolov5n,
                    {},
                    1,
                    true,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{datasource, -1, {streamName}}}
            });
            yolov5n->possibleDevices = {startDevice, edgeNode};
            if (ctrl_systemName == "tuti") {
                yolov5n->possibleDevices = {edgeNode};
            }
            datasource->downstreams.push_back(PipelineEdge{yolov5n, -1, {streamName}});

            // std::shared_ptr<PipelineModel> yolov5n320 = nullptr;
            std::shared_ptr<PipelineModel> yolov5n512 = nullptr;
            // std::shared_ptr<PipelineModel> yolov5s = nullptr;
            
            if (ctrl_systemName == "jlf") {
                yolov5n->possibleDevices = {edgeNode};

                // yolov5n320 = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5n320",
                //         ModelType::Yolov5n320,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5n320->possibleDevices = {edgeNode};


                yolov5n512 = std::make_shared<PipelineModel>(PipelineModel{
                        edgeNode,
                        "yolov5n512",
                        ModelType::Yolov5n512,
                        {},
                        1,
                        true,
                        false,
                        {},
                        {},
                        {},
                        {PipelineEdge{datasource, -1, {streamName}}}
                });
                yolov5n512->possibleDevices = {edgeNode};

                // yolov5s = std::make_shared<PipelineModel>(PipelineModel{
                //         edgeNode,
                //         "yolov5s",
                //         ModelType::Yolov5s,
                //         {},
                //         1,
                //         true,
                //         false,
                //         {},
                //         {},
                //         {},
                //         {PipelineEdge{datasource, -1, {streamName}}}
                // });
                // yolov5s->possibleDevices = {edgeNode};

                // Add downstream connections for the additional YOLO variants in the JLF pipeline
                // datasource->downstreams.push_back(PipelineEdge{yolov5n320, -1, {streamName}});
                datasource->downstreams.push_back(PipelineEdge{yolov5n512, -1, {streamName}});
                // datasource->downstreams.push_back(PipelineEdge{yolov5s, -1, {streamName}});
            }

            auto geardet = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "geardet",
                    ModelType::GearDetect,
                    {},
                    1,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, 2, {streamName}}}
            });
            geardet->possibleDevices = {startDevice, edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{geardet, 0, {streamName}});
            
            if (ctrl_systemName == "jlf") {
                geardet->possibleDevices = {edgeNode};
                geardet->upstreams = {
                    PipelineEdge{yolov5n, 0, {streamName}},
                    // PipelineEdge{yolov5n320, 0, {streamName}},
                    PipelineEdge{yolov5n512, 0, {streamName}},
                    // PipelineEdge{yolov5s, 0, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{geardet, 0, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{geardet, 0, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{geardet, 0, {streamName}});
            }

            auto fashioncolor = std::make_shared<PipelineModel>(PipelineModel{
                    edgeNode,
                    "fashioncolor",
                    ModelType::FashionColor,
                    {},
                    2,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {PipelineEdge{yolov5n, 0, {streamName}}}
            });
            fashioncolor->possibleDevices = {edgeNode};
            yolov5n->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
            
            if (ctrl_systemName == "jlf") {
                fashioncolor->upstreams = {
                    PipelineEdge{yolov5n, 0, {streamName}}, 
                    // PipelineEdge{yolov5n320, 0, {streamName}}, 
                    PipelineEdge{yolov5n512, 0, {streamName}}, 
                    // PipelineEdge{yolov5s, 0, {streamName}}
                };
                // yolov5n320->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                yolov5n512->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
                // yolov5s->downstreams.push_back(PipelineEdge{fashioncolor, 0, {streamName}});
            }

            auto sink = std::make_shared<PipelineModel>(PipelineModel{
                    "sink",
                    "sink",
                    ModelType::Sink,
                    {},
                    0,
                    false,
                    false,
                    {},
                    {},
                    {},
                    {
                        PipelineEdge{equipmentdet, -1, {streamName}}, 
                        PipelineEdge{geardet, -1, {streamName}}, 
                        PipelineEdge{fashioncolor, -1, {streamName}}
                    }
            });
            sink->possibleDevices = {"sink"};
            equipmentdet->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            geardet->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            fashioncolor->downstreams.push_back(PipelineEdge{sink, -1, {streamName}});
            
            if (!sourceName.empty()) {
                yolov5n->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][yolov5n->name];
                equipmentdet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][equipmentdet->name];
                geardet->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][geardet->name];
                fashioncolor->arrivalProfiles.arrivalRates = ctrl_initialRequestRates[sourceName][fashioncolor->name];
            }

            if (ctrl_systemName == "jlf") {
                equipmentdet->possibleDevices = {edgeNode};
                geardet->possibleDevices = {edgeNode};
                fashioncolor->possibleDevices = {edgeNode};
                // return {datasource, yolov5n, yolov5n320, yolov5n512, yolov5s, equipmentdet, geardet, fashioncolor, sink};
                return {datasource, yolov5n, yolov5n512, equipmentdet, geardet, fashioncolor, sink};
            }
            return {datasource, yolov5n, equipmentdet, geardet, fashioncolor, sink};
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
        )->second * skipRate + 1;
        maxPersonRate = std::max(maxPersonRate, ctrl_systemFPS * 1.f);
        float maxCarRate = 1.2 * std::max_element(
                initialPerSecondRate[streamName]["car"].begin(),
                initialPerSecondRate[streamName]["car"].end()
        )->second * skipRate + 1;
        maxCarRate = std::max(maxCarRate, ctrl_systemFPS * 1.f);
        stream->insert({"retinamtface", ctrl_systemFPS});
        stream->insert({"yolov5n", ctrl_systemFPS});
        stream->insert({"yolov5s", ctrl_systemFPS});
        stream->insert({"retina1face", std::ceil(maxPersonRate)});
        stream->insert({"movenet", std::ceil(maxPersonRate)});
        stream->insert({"arcface", std::ceil(maxPersonRate)});
        stream->insert({"carbrand", std::ceil(maxCarRate)});
        stream->insert({"platedet", std::ceil(maxCarRate)});
        stream->insert({"age", std::ceil(maxPersonRate)});
        stream->insert({"gender", std::ceil(maxPersonRate)});
        stream->insert({"emotionnet", std::ceil(maxPersonRate)});
        stream->insert({"carcolor", std::ceil(maxCarRate)});
        stream->insert({"cardamage", std::ceil(maxCarRate)});
        stream->insert({"fashioncolor", std::ceil(maxPersonRate)});
        stream->insert({"equipdetect", std::ceil(maxPersonRate)});
        stream->insert({"geardetect", std::ceil(ctrl_systemFPS)});
        stream->insert({"firedetect", std::ceil(ctrl_systemFPS)});
    }
}

PipelineModelListType deepCopyPipelineModelList(const PipelineModelListType& original) {
    PipelineModelListType newList;
    newList.reserve(original.size());
    for (const auto& modelSp : original) {
        if (modelSp) {
            newList.push_back(std::make_shared<PipelineModel>(*modelSp));
        }
    }
    return newList;
}

std::shared_ptr<TaskHandle> Controller::CreatePipelineFromMessage(TaskDesc msg) {
    auto task = std::make_shared<TaskHandle>(msg.name(),
                                             PipelineTypeReverseList[msg.type()],
                                             msg.stream(),
                                             msg.srcdevice(),
                                             msg.slo(),
                                             std::chrono::system_clock::now(),
                                             msg.edgenode());

    std::map<std::string, std::shared_ptr<NodeHandle>> deviceList = devices.getMap();

    if (deviceList.find(msg.srcdevice()) == deviceList.end()) {
        spdlog::error("Device {0:s} is not connected", msg.srcdevice());
        return nullptr;
    }

    while (!deviceList.at(msg.srcdevice())->initialNetworkCheck) {
        spdlog::get("container_agent")->info("Waiting for device {0:s} to finish network check", msg.srcdevice());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    task->tk_src_device = msg.srcdevice();

    if (task->tk_type != PipelineType::None)
        task->tk_pipelineModels = getModelsByPipelineType(task->tk_type, msg.srcdevice(), msg.name(), msg.stream(), msg.edgenode());
    else {
        task->tk_pipelineModels = PipelineModelListType();
        auto datasource = std::shared_ptr<PipelineModel>(new PipelineModel{msg.srcdevice(), 
                                                                                                               "datasource",
                                                                                                               ModelType::DataSource, 
                                                                                                               std::weak_ptr<TaskHandle>(), 
                                                                                                               0, 
                                                                                                               true});
        datasource->possibleDevices = {msg.srcdevice()};
        datasource->canBeCombined = false;

        task->tk_pipelineModels.push_back(datasource);
        
        std::map<std::string, std::vector<controlmessages::PipelineNeighbor>> downstreams, upstreams;
        
        std::map<std::string, std::shared_ptr<PipelineModel>> models;
        models["datasource"] = datasource;
        
        for (auto &m: msg.models()){
            auto model = std::shared_ptr<PipelineModel>(new PipelineModel{
                    m.device(),
                    m.type(),
                    ModelTypeReverseList[m.type()],
                    std::weak_ptr<TaskHandle>(),
                    m.position(),
                    m.issplitpoint(),
                    m.forwardinput()
            });
            task->tk_pipelineModels.push_back(model);
            models[m.type()] = model;
            downstreams[m.type()] = {};
            for (auto &d: m.downstreams())
                downstreams[m.type()].push_back(d);
            upstreams[m.type()] = {};
            for (auto &u: m.upstreams())
                upstreams[m.type()].push_back(u);
            model->possibleDevices = {};
            for (std::string d: m.possibledevices())
                model->possibleDevices.push_back(d);
        }
        
        auto sink = std::shared_ptr<PipelineModel>(new PipelineModel{
                "sink",
                "sink",
                ModelType::Sink,
                std::weak_ptr<TaskHandle>(),
                0,
                false
        });
        sink->possibleDevices = {"sink"};
        
        for (auto &m: task->tk_pipelineModels){
            for (auto &d: downstreams[m->name]) {
                // FIX: Initialize the PipelineEdge struct with the target, class, and the initial stream name
                m->downstreams.push_back(PipelineEdge{models[d.name()], d.classofinterest(), {msg.stream()}});
            }
            for (auto &u: upstreams[m->name]) {
                m->upstreams.push_back(PipelineEdge{models[u.name()], u.classofinterest(), {msg.stream()}});
                if (u.name() == "datasource")
                    datasource->downstreams.push_back(PipelineEdge{m, -1, {msg.stream()}});
            }
            if (m->downstreams.empty()) {
                m->downstreams.push_back(PipelineEdge{sink, -1, {msg.stream()}});
                sink->upstreams.push_back(PipelineEdge{m, -1, {msg.stream()}});
            }
        }
        task->tk_pipelineModels.push_back(sink);
    }

    for (auto &model: task->tk_pipelineModels) {
        model->datasourceName = {msg.stream()};
        
        // implicitly casts the shared_ptr 'task' to a weak_ptr!
        model->task = task; 
        
        if (model->possibleDevices.empty())
            model->possibleDevices = {msg.srcdevice(), msg.edgenode()};
    }
    return task;
}

void Controller::UpdatePipelineFromMessage(std::shared_ptr<TaskHandle> task, TaskDesc msg) {
    if (!task) return;

    // Ensure to reset the naming for existing models
    for (auto &model: task->tk_pipelineModels) {
        if (model->name.find(task->tk_name) != std::string::npos)
            model->name = model->name.substr(model->name.find('_') + 1);
    }

    std::map<std::string, std::vector<controlmessages::PipelineNeighbor>> downstreams, upstreams;
    std::map<std::string, std::shared_ptr<PipelineModel>> models;
    std::shared_ptr<PipelineModel> datasource = nullptr, sink = nullptr;

    // Add new models if necessary
    for (auto &m: msg.models()){
        std::shared_ptr<PipelineModel> found = nullptr;
        for (auto &model: task->tk_pipelineModels) {
            if (datasource == nullptr && model->name.find("datasource") != std::string::npos)
                datasource = model;
            if (sink == nullptr && model->name.find("sink") != std::string::npos)
                sink = model;
            if (model->name.find(m.type()) == std::string::npos)
                continue;
            found = model;
            break;
        }

        if (found != nullptr){
            models[m.type()] = found;
        } else {
            auto model = std::shared_ptr<PipelineModel>(new PipelineModel{
                    m.device(),
                    m.type(),
                    ModelTypeReverseList[m.type()],
                    std::weak_ptr<TaskHandle>(),
                    m.position(),
                    m.issplitpoint(),
                    m.forwardinput()
            });
            task->tk_pipelineModels.push_back(model);
            models[m.type()] = model;
            model->datasourceName = {msg.stream()};
            model->task = task;
            model->possibleDevices = {};
            for (std::string d: m.possibledevices())
                model->possibleDevices.push_back(d);
            if (model->possibleDevices.empty())
                model->possibleDevices = {msg.srcdevice(), msg.edgenode()};
        }
        downstreams[m.type()] = {};
        for (auto &d: m.downstreams())
            downstreams[m.type()].push_back(d);
        upstreams[m.type()] = {};
        for (auto &u: m.upstreams())
            upstreams[m.type()].push_back(u);
    }

    // Restructure pipeline to include the new models and updated connections
    for (auto &m: models){
        
        // FIX: Replaced contains_pair lambda with get_existing_edge to return a pointer to the edge
        // so we can dynamically inject the new streamName into the unordered_set if the edge exists.
        auto get_existing_edge = [](std::vector<PipelineEdge> &vec, std::shared_ptr<PipelineModel> p, int cls) -> PipelineEdge* {
            for (auto &e : vec) {
                if (e.targetNode.lock() == p && e.classOfInterest == cls) return &e;
            }
            return nullptr;
        };

        for (auto &d: downstreams[m.second->name]) {
            auto dst = models[d.name()];
            int cls = d.classofinterest();
            
            auto* existing_down = get_existing_edge(m.second->downstreams, dst, cls);
            if (!existing_down) {
                m.second->downstreams.push_back(PipelineEdge{dst, cls, {msg.stream()}});
            } else {
                existing_down->streamNames.insert(msg.stream()); // Upsert the stream!
            }

            auto* existing_up = get_existing_edge(dst->upstreams, m.second, cls);
            if (!existing_up) {
                dst->upstreams.push_back(PipelineEdge{m.second, cls, {msg.stream()}});
            } else {
                existing_up->streamNames.insert(msg.stream()); // Upsert the stream!
            }
        }
        
        for (auto &u: upstreams[m.second->name]) {
            auto src = (u.name() == "datasource") ? datasource : models[u.name()];
            if (src == nullptr) continue;
            int cls = u.classofinterest();
            
            auto* existing_up = get_existing_edge(m.second->upstreams, src, cls);
            if (!existing_up) {
                m.second->upstreams.push_back(PipelineEdge{src, cls, {msg.stream()}});
            } else {
                existing_up->streamNames.insert(msg.stream()); // Upsert the stream!
            }

            auto* existing_down = get_existing_edge(src->downstreams, m.second, cls);
            if (!existing_down) {
                src->downstreams.push_back(PipelineEdge{m.second, cls, {msg.stream()}});
            } else {
                existing_down->streamNames.insert(msg.stream()); // Upsert the stream!
            }
        }
        
        if (sink != nullptr && m.second->downstreams.empty()) {
            m.second->downstreams.push_back(PipelineEdge{sink, -1, {msg.stream()}});
            sink->upstreams.push_back(PipelineEdge{m.second, -1, {msg.stream()}});
        }
    }

    // Remove models that are no longer present
    std::vector<std::shared_ptr<PipelineModel>> to_delete = {};
    for (auto &model: task->tk_pipelineModels) {
        bool found = false;
        if (model != datasource && model != sink) {
            for (auto &m: models)
                if (m.second == model) {
                    found = true;
                    break;
                }
            if (!found)
                to_delete.push_back(model);
        }
    }
    for (auto &model: to_delete) {
        task->tk_pipelineModels.erase(std::remove(task->tk_pipelineModels.begin(), task->tk_pipelineModels.end(), model), task->tk_pipelineModels.end());
        for (auto &dWk: model->downstreams) {
            if (auto d = dWk.targetNode.lock()) {
                d->upstreams.erase(std::remove_if(
                        d->upstreams.begin(),
                        d->upstreams.end(),
                        [&model](const PipelineEdge &p) {
                            return p.targetNode.lock() == model;
                        }
                ), d->upstreams.end());
            }
        }
        for (auto &uWk: model->upstreams) {
            if (auto u = uWk.targetNode.lock()) {
                u->downstreams.erase(std::remove_if(
                        u->downstreams.begin(),
                        u->downstreams.end(),
                        [&model](const PipelineEdge &p) {
                            return p.targetNode.lock() == model;
                        }
                ), u->downstreams.end());
            }
        }
    }
}
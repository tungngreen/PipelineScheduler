import zmq
import sys
import controlmessages_pb2  # Generated from controlmessages.proto

downstream_map = {"firedetect": [],
                  "yolov5n": ["carcolor", "fashioncolor"],
                  "carcolor": ["carbrand"],
                  "carbrand": ["platedet", "cardamage"],
                  "platedet": [],
                  "cardamage": [],
                  "fashioncolor": ["retina1face"],
                  "retina1face": []}
upstream_map = {"firedetect": ["datasource"],
                "yolov5n": ["datasource"],
                "carcolor": ["yolov5n"],
                "carbrand": ["carcolor"],
                "platedet": ["carbrand"],
                "cardamage": ["carbrand"],
                "fashioncolor": ["yolov5n"],
                "retina1face": ["fashioncolor"]}

def start_task():
    # Create the protobuf message
    request = controlmessages_pb2.TaskDesc()

    request.name = "traffic4"
    request.slo = 300000
    request.stream = "../data/short/traffic4.mp4"
    request.srcDevice = "orinano2"
    request.edgeNode = "server"
    for model in downstream_map.keys():
        desc = controlmessages_pb2.PipeModelDesc()
        desc.device = "orinano2"
        desc.type = model
        desc.position = 1 if (model == "yolov5n") else 2
        desc.isSplitPoint = False
        desc.forwardInput = True if (model == "carbrand") else False
        for d in downstream_map[model]:
            neighbor = controlmessages_pb2.PipelineNeighbor()
            neighbor.name = d
            neighbor.classOfInterest = -1
            desc.downstreams.append(neighbor)
        for u in upstream_map[model]:
            neighbor = controlmessages_pb2.PipelineNeighbor()
            neighbor.name = u
            neighbor.classOfInterest = -1
            desc.upstreams.append(neighbor)
        request.models.append(desc)

    serialized = request.SerializeToString()
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    controller_addr = "tcp://localhost:60000"
    socket.connect(controller_addr)

    try:
        socket.send(b"START " + serialized)
        print(f"[INFO] Sent START request to {controller_addr}")

        # Wait for the controller response
        reply = socket.recv_string()
        if reply.strip().lower() == "success":
            print("✅ Task started successfully.")
        else:
            print("❌ Controller returned error:", reply)

    except Exception as e:
        print(f"[ERROR] Exception during communication: {e}")

    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    start_task()

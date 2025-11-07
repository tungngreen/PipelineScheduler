import zmq
import sys
import controlmessages_pb2  # Generated from controlmessages.proto

downstream_map = {"yolov5n": ["carcolor", "carbrand"],
                  "carcolor": [],
                  "carbrand": ["platedet"],
                  "platedet": []}
upstream_map = {"yolov5n": ["datasource"],
                "carcolor": ["yolov5n"],
                "carbrand": ["yolov5n"],
                "platedet": ["carbrand"]}

def start_task():
    # Create the protobuf message
    request = controlmessages_pb2.TaskDesc()

    request.name = "task"
    request.slo = 200000
    request.stream = "../data/traffic_profiling.mp4"
    request.srcDevice = "orinagx1"
    request.edgeNode = "server"
    for model in ["yolov5n", "carcolor", "carbrand", "platedet"]:
        desc = controlmessages_pb2.PipeModelDesc()
        desc.device = "orinagx1"
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

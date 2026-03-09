import argparse
import json
import logging
import signal
import sys
import threading
import time
import zmq
import os

import controlmessages_pb2 as cm

def parse_arguments():
    """Parses command line arguments similar to absl::Flags."""
    parser = argparse.ArgumentParser(description="Container Agent Initialization")

    parser.add_argument("--json", type=str, default=None, help="configurations for microservices as json")
    parser.add_argument("--json_path", type=str, default=None, help="json for configuration inside a file")
    parser.add_argument("--trt_json", type=str, default=None, help="optional json for TRTConfiguration")
    parser.add_argument("--trt_json_path", type=str, default=None, help="json for TRTConfiguration")
    parser.add_argument("--port", type=int, default=0, help="control port for the service")
    parser.add_argument("--port_offset", type=int, default=0, help="port offset for control communication")
    parser.add_argument("--device", type=int, default=0, help="Index of GPU device")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level 0:trace, 1:debug... 6:off")
    parser.add_argument("--logging_mode", type=int, default=0, help="0:stdout, 1:file, 2:both")
    parser.add_argument("--log_dir", type=str, default="../logs", help="Log path for the container")
    parser.add_argument("--profiling_mode", type=int, default=0, help="0:deployment, 1:profiling, 2:empty_profiling")
    parser.add_argument("--restart", action="store_true", help="Restart container after finishing input stream")

    return parser.parse_args()


class ContainerAgent:
    def __init__(self, configs, args):
        self.configs = configs
        container_configs = configs.get("container", {})

        # Basic configurations
        self.cont_name = container_configs.get("cont_name", "default_container")
        self.cont_system_name = container_configs.get("cont_systemName", "default_system")
        self.cont_task_name = container_configs.get("cont_taskName", "default_task")

        self.pid = os.getpid()
        self.cont_run = True

        self.messaging_ctx = zmq.Context(1)
        self.sending_socket = self.messaging_ctx.socket(zmq.REQ)
        req_address = f"tcp://localhost:{cm.DEVICE_RECEIVE_PORT + args.port_offset}"
        self.sending_socket.connect(req_address)
        self.sending_socket.setsockopt(zmq.RCVTIMEO, 100)
        self.device_message_queue = self.messaging_ctx.socket(zmq.SUB)
        sub_address = f"tcp://localhost:{cm.DEVICE_MESSAGE_QUEUE_PORT + args.port_offset}"
        self.device_message_queue.connect(sub_address)
        self.device_message_queue.setsockopt_string(zmq.SUBSCRIBE, f"{self.cont_name}|")
        self.device_message_queue.setsockopt(zmq.RCVTIMEO, 1000)

        self.handlers = {
            "CONTAINER_STOP": self.stop_execution
        }

        logging.info(f"Container Agent {self.cont_name} initialized with PID {self.pid}.")


    def is_data_source(self):
        """Stub to check if the current container is purely a data source."""
        return self.cont_task_name in ["dsrc", "datasource"]

    # ---------------------------------------------------------
    # Communication Loops
    # ---------------------------------------------------------

    def handle_control_messages(self):
        """Background loop to receive and route ZMQ control messages."""
        while self.cont_run:
            try:
                # Recv with a timeout (handled by ZMQ socket options)
                message = self.device_message_queue.recv()

                # Expected format: b"topic type payload"
                # Split at most 2 times to isolate topic, type, and the raw byte payload
                parts = message.split(b" ", 2)
                if len(parts) >= 2:
                    topic = parts[0].decode('utf-8')
                    msg_type = parts[1].decode('utf-8')
                    payload = parts[2] if len(parts) > 2 else b""

                    if msg_type in self.handlers:
                        # Pass the raw bytes payload to the specific handler
                        self.handlers[msg_type](payload)
                    else:
                        logging.error(f"Received unknown device type: {msg_type} (topic: {topic})")

            except zmq.Again:
                continue
            except Exception as e:
                logging.error(f"Error receiving control message: {e}")

    def send_message_to_device(self, msg_type: str, content: bytes) -> bytes:
        """Sends a serialized protobuf to the device and waits for a response."""
        # Frame the message as bytes: b"TYPE <serialized_payload>"
        msg = msg_type.encode('utf-8') + b" " + content
        try:
            self.sending_socket.send(msg, zmq.DONTWAIT)
            response = self.sending_socket.recv()
            return response
        except zmq.ZMQError as e:
            logging.error(f"Failed to communicate with device on send: {e}")
            return b""

    def report_start(self):
        """Reports the container start to the device and retrieves its assigned PID."""
        request = cm.ProcessData()
        request.msvc_name = self.cont_name

        response_bytes = self.send_message_to_device("MSVC_START_REPORT", request.SerializeToString())

        reply = cm.ProcessData()
        try:
            reply.ParseFromString(response_bytes)
            self.pid = reply.pid
            logging.info(f"Container Agent started and assigned PID: {self.pid}")
        except Exception as e:
            logging.error(f"Failed to parse reply from device agent: {e}")
            self.pid = 0

    # ---------------------------------------------------------
    # Message Handlers
    # ---------------------------------------------------------

    def stop_execution(self, payload: str):
        logging.info("Stopping execution via control message.")
        self.cont_run = False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_arguments()

    configs = {}
    if args.json_path:
        with open(args.json_path, 'r') as f:
            configs = json.load(f)
    elif args.json:
        configs = json.loads(args.json)
    else:
        logging.error("No configurations found. Provide --json or --json_path.")
        sys.exit(1)

    agent = ContainerAgent(configs, args)

    def signal_handler(sig, frame):
        logging.info("Interrupt received, shutting down Container Agent...")
        agent.cont_run = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    listener_thread = threading.Thread(target=agent.handle_control_messages, daemon=True)
    listener_thread.start()
    agent.report_start()

    # Keep the main process alive while the background listener handles commands
    while agent.cont_run:
        time.sleep(1)

    logging.info("Container Agent exited cleanly.")

if __name__ == "__main__":
    main()
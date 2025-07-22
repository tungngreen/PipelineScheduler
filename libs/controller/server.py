import time
import grpc
from concurrent import futures
from google.protobuf.empty_pb2 import Empty
import json
import os
import psycopg2
from psycopg2 import sql

import controlcommands_pb2_grpc
from startcontainer import read_config_file, start_container, initialTasks, create_container_from_json
from controlmessages_pb2 import ConnectionConfigs, SystemInfo, DummyMessage
from controlmessages_pb2_grpc import ControlMessagesServicer
import controlmessages_pb2_grpc

DB_CONFIG = {
    'dbname': 'pipeline',
    'user': 'controller',
    'password': 'agent',
    'host': 'localhost',
    'port': '60004'
}
SCHEMA_NAME = 'st2_st'

def create_schema_if_not_exists(db_config, schema_name):
    try:
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s;",
            (schema_name,)
        )
        result = cur.fetchone()
        if result is None:
            print(f"Schema '{schema_name}' Doesn't exist, creating and setting permissions...")
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name))
            )
            cur.execute(
                sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT ALL PRIVILEGES ON TABLES TO controller;")
                .format(sql.Identifier(schema_name))
            )
            cur.execute(
                sql.SQL("GRANT USAGE ON SCHEMA {} TO device_agent, container_agent;")
                .format(sql.Identifier(schema_name))
            )
            cur.execute(
                sql.SQL("GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA {} TO device_agent, container_agent;")
                .format(sql.Identifier(schema_name))
            )
            cur.execute(
                sql.SQL("GRANT CREATE ON SCHEMA {} TO device_agent, container_agent;")
                .format(sql.Identifier(schema_name))
            )
            cur.execute(
                sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT SELECT, INSERT ON TABLES TO device_agent, container_agent;")
                .format(sql.Identifier(schema_name))
            )
            print("Schema successful！")
        else:
            print(f"Schema '{schema_name}' is already exist.")
        cur.close()
        conn.close()
    except Exception as e:
        print("Error creating schema：", e)


def save_current_platform(request,json_path="./jsons/CurrentPlatform.json"):

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    platform_info = {
        "device_name"        : request.device_name,
        "device_type"        : request.device_type,
        "ip_address"         : request.ip_address,
        "agent_port_offset"  : request.agent_port_offset,
        "processors"         : request.processors,
        "memory"             : list(request.memory),
    }


    data = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            data = content if isinstance(content, list) else [content]
        except Exception as e:
            print(f"[platform_io] Failed to read/parse '{json_path}': {e}")
            data = []

    data.append(platform_info)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[platform_io] Appended platform info; total entries = {len(data)}")
    except Exception as e:
        print(f"[platform_io] Error writing to {json_path}: {e}")


class ControlMessagesServicer(controlmessages_pb2_grpc.ControlMessagesServicer):
    def AdvertiseToController(self, request, context):
        """
        Called when the DeviceAgent sends an AdvertiseToController request.
        """
        print("Received AdvertiseToController request:")
        print(f"  Device Name         : {request.device_name}")
        print(f"  Device Type (ID)    : {request.device_type}")
        print(f"  IP Address          : {request.ip_address}")
        print(f"  Agent Port Offset   : {request.agent_port_offset}")
        print(f"  Processors          : {request.processors}")
        print(f"  Memory (bytes each) : {list(request.memory)}")

        save_current_platform(request, json_path="./jsons/CurrentPlatform.json")
        
        response = SystemInfo(
            name="st",
            experiment="st2",
            # 
            offloading_targets=[]
        )
        return response

    def SendDummyData(self, request, context):

        print("Received SendDummyData request:")
        print(f"  Origin Name: {request.origin_name}")
        print(f"  Generation Time: {request.gen_time}")
        print(f"  Data (bytes): {request.data}")
        return Empty()

def serve():

    create_schema_if_not_exists(DB_CONFIG, SCHEMA_NAME)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    controlmessages_pb2_grpc.add_ControlMessagesServicer_to_server(ControlMessagesServicer(), server)
    server.add_insecure_port('[::]:60001')
    server.start()
    
    print("ControlMessages server started on port 60001...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

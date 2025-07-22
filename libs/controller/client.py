import grpc
import json
import os
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Tuple, Set
import torch.nn.functional as F
import torch.optim as optim
import sys
from typing import Dict, Tuple, List
from controlcommands_pb2_grpc import ControlCommandsStub
from controlcommands_pb2 import ContainerConfig
from controlcommands_pb2 import ContainerLink 
from ST_test import predict_and_aggregate
from ST_add import dynamic_train_and_predict
from STpre_nospatial import predict_and_schedule_nospatial
from distream import distream_scheduling
from LSTM_spatial import distream_Generator
from GNN_LSTM import gnn_decision

_prev_decisions: Dict[str, Dict[str, float]] = {}

def load_pipeline_configs(st_offload_path: str) -> list:

    with open(st_offload_path, 'r', encoding='utf-8') as f:
        main_cfg = json.load(f)

    sink_ip = main_cfg.get('sink_ip')
    exp_name = main_cfg.get('expName')
    system_name = main_cfg.get('systemName')

    pipelines = main_cfg.get('initial_pipelines', [])
    for p in pipelines:
        p['expName'] = exp_name
        p['systemName'] = system_name

    return pipelines,sink_ip


def get_target_devices(current_platform_path: str):

    ip_to_name = {}
    name_to_ip = {}
    if os.path.exists(current_platform_path):
        with open(current_platform_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                ip   = entry.get('ip_address')
                name = entry.get('device_name')
                if ip and name:
                    ip_to_name[ip]   = name
                    name_to_ip[name] = ip
    return ip_to_name, name_to_ip


def prepare_container_config(pipeline: dict, jsons_dir: str, device_ip: str, all_devices: dict,pipelines: list, sink_ip: str=None, sink_port: int=None, port_offset: int = 0) -> ContainerConfig:

    model_type = pipeline['model_type']
    if model_type == 2:
        tpl = 'yolov5_aicity.json'
    elif model_type == 3:
        tpl = 'platedet.json'
    else:
        tpl = None

    start_config = {}
    if tpl:
        tpl_path = os.path.join(jsons_dir, tpl)
        if os.path.exists(tpl_path):
            with open(tpl_path, 'r', encoding='utf-8') as f:
                start_config = json.load(f)

    cont = start_config.setdefault('container', {})
    cont.update({
        'cont_experimentName': pipeline.get('expName'),
        'cont_systemName':     pipeline.get('systemName'),
        'cont_pipeName':       pipeline.get('pipeline'),
        'cont_hostDevice':     pipeline.get('device'),
        'cont_hostDeviceType': re.sub(r'\d+$', '', pipeline.get('device', '')),
        'cont_name':           pipeline.get('container_name'),
        'cont_pipeline':       start_config.get('container', {}).get('cont_pipeline', [])
    })

    cont_pipeline = cont['cont_pipeline']

    if model_type == 2:

        for svc in cont_pipeline:
            if svc.get('msvc_upstreamMicroservices'):
                for up in svc['msvc_upstreamMicroservices']:
                    if up['nb_name'] == 'video_source':
                        up['nb_link'] = [ pipeline['video_source'] ]
        for svc in cont_pipeline:
            if svc.get('msvc_name') == 'inference':
                svc['path'] = pipeline['model_path']
                break

    elif model_type == 3:
        for svc in cont_pipeline:
            if svc.get('msvc_name') == 'inference':
                svc['path'] = pipeline['model_path']
                break  
    
# —— 1) Yolov5 -> RetinaFace —— 
    if model_type == 2:
        for svc in cont_pipeline:
            if svc.get('msvc_name') == 'sender':

                orig = svc['msvc_dnstreamMicroservices'][0]['nb_link'][0]
                try:
                    base_port = int(orig.split(':')[1])
                except:
                    base_port = 50010
                new_port = base_port + port_offset
                svc['msvc_dnstreamMicroservices'][0]['nb_link'] = [
                    f"localhost:{new_port}"
                ]
                break

    # —— 2) RetinaFace <- Yolov5 —— 
    if model_type == 3:
        for svc in cont_pipeline:
            if svc.get('msvc_name') == 'receiver':
                orig = svc['msvc_upstreamMicroservices'][0]['nb_link'][0]
                try:
                    base_port = int(orig.split(':')[1])
                except:
                    base_port = 50010
                new_port = base_port + port_offset
                svc['msvc_upstreamMicroservices'][0]['nb_link'] = [
                    f"0.0.0.0:{new_port}"
                ]
                break

    # 2) RetinaFace -> Sink
    if model_type == 3 and sink_ip and sink_port:
        sender = cont_pipeline[-1]['msvc_dnstreamMicroservices'][0]
        sender['nb_link'] = [f"{sink_ip}:{sink_port}"]


    json_str = json.dumps(start_config)
    if model_type == 2:           
        exe_cmd = "./Container_Yolov5"
    elif model_type == 3:         
        exe_cmd = "./Container_PlateDet"
    else:
        exe_cmd = ""          
    device_int = 0 if pipeline['device'] == 'server' else 0

    if model_type == 2:        # YoloV5
        control_port = 55000
    elif model_type == 3:      # RetinaFace
        control_port = 55001
    else:
        control_port = 55003  

    return ContainerConfig(
        name=pipeline['container_name'],
        json_config=json_str,
        executable=exe_cmd,
        device=device_int,
        control_port=control_port,
        model_type=model_type,
        input_shape=pipeline['input_shape']
    )

def start_container(
    st_offload_path: str,
    current_platform_path: str,
    jsons_dir: str,
    port: int = 60002
):

    pipelines, sink_ip = load_pipeline_configs(st_offload_path)
    sink_port = 55050
    sink_name = f"{pipelines[0]['expName']}_{pipelines[0]['systemName']}_sink"


    sink_pipeline = pipelines[0]['pipeline']
    start_cfg = {
        "experimentName": pipelines[0]['expName'],
        "systemName":   pipelines[0]['systemName'],
        "pipelineName": sink_pipeline
    }
    sink_cfg = ContainerConfig(
        name=sink_name,
        json_config=json.dumps(start_cfg),
        executable="./runSink",
        device=-1,
        control_port=sink_port,
        model_type=4,
        input_shape=[]
    )


    with open(current_platform_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        entries = [entries]
    dev2info: Dict[str, Tuple[str, int]] = {
        e['device_name']: (e['ip_address'], e.get('agent_port_offset', 0))
        for e in entries
    }

    stub_map: Dict[Tuple[str,int], ControlCommandsStub] = {}
    def get_stub(ip: str, offset: int) -> ControlCommandsStub:
        key = (ip, offset)
        if key not in stub_map:
            stub_map[key] = ControlCommandsStub(
                grpc.insecure_channel(f"{ip}:{port + offset}")
            )
        return stub_map[key]

    to_start: List[Tuple[str,int,ContainerConfig]] = []
    for p in pipelines:
        dev = p['device']
        if dev in dev2info:
            ip, offset = dev2info[dev]
            cfg = prepare_container_config(
                p, jsons_dir, ip, {ip: dev}, pipelines,
                sink_ip=sink_ip, sink_port=sink_port, port_offset=offset
            )
            to_start.append((ip, offset, cfg))
    to_start.append((sink_ip, 0, sink_cfg))

    for ip, offset, cfg in to_start:
        stub = get_stub(ip, offset)
        print(f"[StartContainer] Request start of {cfg.name} on {ip}:{port + offset}")
        try:
            stub.StartContainer(cfg)
            print(f"[OK] {cfg.name} started successfully")
        except grpc.RpcError as e:
            print(f"[ERROR] Failed to start {cfg.name} on {ip}:{port + offset} - {e.code()} {e.details()}")
       
def update_downstream_decisions(
    decisions: Dict[Tuple[int, int], float],
    st_offload_path: str,
    nodes_file: str,
    platform_file: str
):
    """
    对比当前 decisions 与上一次 _prev_decisions，先处理相同卸载目标的 Modify，再对剩余不同目标集合进行 Add/Remove/Overwrite。
    decisions: {(src_node_idx, dst_node_idx): percent}
    """
    global _prev_decisions
    pipelines, _ = load_pipeline_configs(st_offload_path)
    cont_device = {p['container_name']: p['device'] for p in pipelines}


    with open(nodes_file, 'r', encoding='utf-8') as f:
        idx2dev: Dict[str, str] = json.load(f)


    with open(platform_file, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    dev2ip = {e['device_name']: e['ip_address'] for e in entries}
    dev2offset = {e['device_name']: e.get('agent_port_offset', 0) for e in entries}
    # cont_ip = {cont: dev2ip.get(dev, '') for cont, dev in cont_device.items()}
    cont_ip: Dict[str, str] = {}
    for cont, dev in cont_device.items():
        ip = dev2ip.get(dev)
        if not ip:
            print(f"[Error] device '{dev}' is not in {platform_file}")
        cont_ip[cont] = ip or ''


    new_map: Dict[str, Dict[str, float]] = {}
    for (src_idx, dst_idx), pct in decisions.items():
        src_dev = idx2dev.get(str(src_idx)); dst_dev = idx2dev.get(str(dst_idx))
        if not src_dev or not dst_dev:
            # print(f"[Error] Nodes {src_idx} or {dst_idx} is not in {nodes_file}")
            print(f"[Skip] 决策 ({src_idx},{dst_idx})：节点不在 {nodes_file} 中")
            continue
        try:
            src_cont = next(
                p['container_name']
                for p in pipelines
                if p['device'] == src_dev and p['model_type'] == 2
            )
            dst_cont = next(
                p['container_name']
                for p in pipelines
                if p['device'] == dst_dev and p['model_type'] == 3
            )
        except StopIteration:
            print(f"[Skip] ({src_idx},{dst_idx}) no device in current platform : device={src_dev}/{dst_dev}")
            continue
        # src_cont = next(p['container_name'] for p in pipelines if p['device']==src_dev and p['model_type']==2)
        # dst_cont = next(p['container_name'] for p in pipelines if p['device']==dst_dev and p['model_type']==3)
        new_map.setdefault(src_cont, {})[dst_cont] = pct


    for src_cont, curr in new_map.items():
        prev = _prev_decisions.get(src_cont, {})
        prev_set = set(prev.keys())
        curr_set = set(curr.keys())
        device_name  = cont_device[src_cont]
        offset       = dev2offset.get(device_name, 0)
        channel_port = 60002 + offset

        common = prev_set & curr_set
        for dst in common:
            old_pct = prev.get(dst)
            new_pct = curr.get(dst)
            if old_pct is not None and new_pct != old_pct:
                link = ContainerLink(
                    mode               = 3,  
                    name               = src_cont,
                    downstream_name    = "name",
                    ip                 = cont_ip.get(dst, ''),
                    port               = 50010,
                    data_portion       = new_pct / 100,
                    old_link           = '',
                    timestamp          = int(time.time_ns()//1_000), 
                    offloading_duration= 15 
                )
                print(f"[Modify] {src_cont}->{dst} {old_pct}% -> {new_pct}% | src_ip={cont_ip[src_cont]}")
                print(link)
                #send grpc
                channel = grpc.insecure_channel(f"{cont_ip[src_cont]}:{channel_port}")
                stub    = ControlCommandsStub(channel)
                try:
                    stub.UpdateDownstream(link, timeout=5.0)
                    print(f"[Sent Modify] {src_cont} -> {dst}")
                except grpc.RpcError as e:
                    print(f"[Error] Modify gRPC failed: {e.code()} {e.details()}")
                finally:
                    channel.close()


        removed = prev_set - curr_set
        added   = curr_set - prev_set


        if added and removed and len(added) == len(removed):
            for old_dst, new_dst in zip(sorted(removed), sorted(added)):
                new_pct = curr[new_dst]
                link = ContainerLink(
                    mode               = 0,  
                    name               = src_cont,
                    downstream_name    = "name",
                    ip                 = cont_ip.get(new_dst, ''),
                    port               = 50010,
                    data_portion       = new_pct / 100,
                    old_link           = f"{cont_ip.get(old_dst, '')}:50010",
                    timestamp          = int(time.time_ns()//1_000),
                    offloading_duration= 15
                )
                print(f"[Overwrite] {src_cont}: {old_dst}-> {new_dst} replaced | src_ip={cont_ip[src_cont]}")
                print(link)
                #send grpc
                channel = grpc.insecure_channel(f"{cont_ip[src_cont]}:{channel_port}")
                stub    = ControlCommandsStub(channel)
                try:
                    stub.UpdateDownstream(link, timeout=5.0)
                    print(f"[Sent overwrite] {src_cont} -> {new_dst}")
                except grpc.RpcError as e:
                    print(f"[Error] overwrite gRPC failed: {e.code()} {e.details()}")
                finally:
                    channel.close()
        else:

            for dst in removed:
                link = ContainerLink(
                    mode               = 2,
                    name               = src_cont,
                    downstream_name    = "name",
                    ip                 = cont_ip.get(dst, ''),
                    port               = 50010,
                    data_portion       = 0,
                    old_link           = '',
                    timestamp          = int(time.time_ns()//1_000),
                    offloading_duration= 15
                )
                print(f"[Remove] {src_cont}->{dst} removed | src_ip={cont_ip[src_cont]}")
                print(link)
                #send grpc
                channel = grpc.insecure_channel(f"{cont_ip[src_cont]}:{channel_port}")
                stub    = ControlCommandsStub(channel)
                try:
                    stub.UpdateDownstream(link, timeout=5.0)
                    print(f"[Sent remove] {src_cont} -> {dst}")
                except grpc.RpcError as e:
                    print(f"[Error] remove gRPC failed: {e.code()} {e.details()}")
                finally:
                    channel.close()


            for dst in added:
                pct = curr[dst]
                link = ContainerLink(
                    mode               = 1,
                    name               = src_cont,
                    downstream_name    = "name",
                    ip                 = cont_ip.get(dst, ''),
                    port               = 50010,
                    data_portion       = pct / 100,
                    old_link           = '',
                    timestamp          = int(time.time_ns()//1_000),
                    offloading_duration= 15
                )
                print(f"[Add] {src_cont}->{dst} {pct}% | src_ip={cont_ip[src_cont]}")
                print(link)
                #send grpc
                channel = grpc.insecure_channel(f"{cont_ip[src_cont]}:{channel_port}")
                stub    = ControlCommandsStub(channel)
                try:
                    stub.UpdateDownstream(link, timeout=5.0)
                    print(f"[Sent Add] {src_cont} -> {dst}")
                except grpc.RpcError as e:
                    print(f"[Error] Add gRPC failed: {e.code()} {e.details()}")
                finally:
                    channel.close()

        _prev_decisions[src_cont] = curr.copy()

if __name__ == '__main__':
    
    base = './jsons'
    st_offload = os.path.join(base, 'SToffloading_aicity.json')
    current_platform = os.path.join(base, 'CurrentPlatform.json')
    Nodes = os.path.join(base, 'Nodes.json')


    start_ts = time.time()
    start_container(
        os.path.join(base, 'SToffloading_aicity.json'),
        os.path.join(base, 'CurrentPlatform.json'),
        base
    )


    print("Warm up 30s")
    time.sleep(30)


    pipelines, _ = load_pipeline_configs(st_offload)
    _, name_to_ip = get_target_devices(current_platform)

    model_file = 'model_super_school.pt'
    cites_file = 'bandwidth.cites'
    data_file = 's36_person.json'
    interval = 15.0
    device_counts = [8, 12, 16]
    change_times  = [60, 120, 180]

    while True:
        elapsed = int(time.time() - start_ts)
        try:
            decisions = predict_and_aggregate(
                model_file,
                cites_file,
                data_file,
                end_second=elapsed
            )

        except Exception as e:
            print(f"[Error] 预测失败: {e}")
            time.sleep(interval)
            continue
        update_downstream_decisions(decisions, st_offload,Nodes, current_platform)

        time.sleep(interval)

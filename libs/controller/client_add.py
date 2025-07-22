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
from distream import distream_scheduling

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


def prepare_container_config(pipeline: dict, jsons_dir: str, device_ip: str, all_devices: dict,pipelines: list, sink_ip: str=None, sink_port: int=None) -> ContainerConfig:
    model_type = pipeline['model_type']
    if model_type == 2:
        tpl = 'yolov5.json'
    elif model_type == 3:
        tpl = 'retinaface.json'
    else:
        tpl = None

    start_config = {}
    if tpl:
        tpl_path = os.path.join(jsons_dir, tpl)
        if os.path.exists(tpl_path):
            with open(tpl_path, 'r', encoding='utf-8') as f:
                start_config = json.load(f)

    # 2. 填写 container 基本信息
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
    
    if model_type == 2:  
        face = next((p for p in pipelines if p['model_type'] == 3), None)
        if face:
            idx = len(cont_pipeline) - 2
            cont_pipeline[idx]['msvc_dnstreamMicroservices'][0]['nb_link'] = [
                f"{device_ip}:{face['recv_port']}"
            ]


    if model_type == 3 and sink_ip and sink_port:

        sender = cont_pipeline[-1]['msvc_dnstreamMicroservices'][0]
        sender['nb_link'] = [f"{sink_ip}:{sink_port}"]
        sender['nb_commMethod'] = 2  


    json_str = json.dumps(start_config)
    if model_type == 2:           
        exe_cmd = "./Container_Yolov5"
    elif model_type == 3:         
        exe_cmd = "./Container_RetinaFace"
    else:
        exe_cmd = ""          
    device_int = 1 if pipeline['device'] == 'server' else 0

    if model_type == 2:        
        control_port = 55000
    elif model_type == 3:      
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



    ip_to_name, name_to_ip = get_target_devices(current_platform_path)

    stub_map = {}
    def get_stub(ip):
        if ip not in stub_map:
            stub_map[ip] = ControlCommandsStub(grpc.insecure_channel(f"{ip}:{port}"))
        return stub_map[ip]

    to_start = []
    for ip, device_name in ip_to_name.items():
        for p in pipelines:
            if p['device'].lower() == device_name.lower():
                cfg = prepare_container_config(
                    p, jsons_dir, ip, ip_to_name, pipelines,
                    sink_ip=sink_ip, sink_port=sink_port
                )
                to_start.append((ip, cfg))
    to_start.append((sink_ip, sink_cfg))

    for ip, cfg in to_start:
        stub = get_stub(ip)
        print(f"create {cfg.name} to {ip}:{port}")
        try:
            stub.StartContainer(cfg)
            print(f"[OK] {cfg.name} create successful")
        except grpc.RpcError as e:
            print(f"[ERR] {cfg.name} create fail: {e.code()} {e.details()}")

def update_downstream_decisions(
    decisions: Dict[Tuple[int, int], float],
    st_offload_path: str,
    nodes_file: str,
    platform_file: str
):
    global _prev_decisions
    pipelines, _ = load_pipeline_configs(st_offload_path)
    cont_device = {p['container_name']: p['device'] for p in pipelines}

    # 2) 读取 idx->device 映射
    with open(nodes_file, 'r', encoding='utf-8') as f:
        idx2dev: Dict[str, str] = json.load(f)

    # 3) 读取 device->ip 映射
    with open(platform_file, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    dev2ip = {e['device_name']: e['ip_address'] for e in entries}
    # cont_ip = {cont: dev2ip.get(dev, '') for cont, dev in cont_device.items()}
    cont_ip: Dict[str, str] = {}
    for cont, dev in cont_device.items():
        ip = dev2ip.get(dev)
        if not ip:
            print(f"[Error] device '{dev}' is not in {platform_file}")
        cont_ip[cont] = ip or ''

    # 4) 构建本次映射: src_cont -> { dst_cont: pct }
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
        new_map.setdefault(src_cont, {})[dst_cont] = pct

    for src_cont, curr in new_map.items():
        prev = _prev_decisions.get(src_cont, {})
        prev_set = set(prev.keys())
        curr_set = set(curr.keys())

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
                channel = grpc.insecure_channel(f"{cont_ip[src_cont]}:60002")
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
                    mode               = 0,  # Overwrite
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
                channel = grpc.insecure_channel(f"{cont_ip[src_cont]}:60002")
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
                channel = grpc.insecure_channel(f"{cont_ip[src_cont]}:60002")
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

                channel = grpc.insecure_channel(f"{cont_ip[src_cont]}:60002")
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
    st_offload = os.path.join(base, 'SToffloading.json')
    current_platform = os.path.join(base, 'CurrentPlatform.json')
    Nodes = os.path.join(base, 'Nodes.json')


    start_ts = time.time()
    start_container(
        os.path.join(base, 'SToffloading.json'),
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
            decisions = dynamic_train_and_predict(device_counts,
                              change_times,
                              end_seconds=elapsed,
                              cites_file="bandwidth.cites",
                              data_file="s36_person.json",
                              fps=15,
                              label_file="gt_campus.csv")
            print("decisions",decisions)
        except Exception as e:
            print(f"[Error] 预测失败: {e}")
            time.sleep(interval)
            continue

        update_downstream_decisions(decisions, st_offload,Nodes, current_platform)

        time.sleep(interval)

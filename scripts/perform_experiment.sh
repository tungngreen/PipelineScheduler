#!/bin/bash

# Remote hosts
agx=("143.248.57.132" "143.248.57.133" "143.248.57.134")
agx_names=("agxavier1" "agxavier2" "agxavier3")
nx=("143.248.57.93" "143.248.57.114" "143.248.57.112" "143.248.57.135" "143.248.57.115")
nx_names=("nx1" "nx2" "nx3" "nx4" "nx5")
orn=("143.248.57.150" "143.248.57.35" "143.248.57.74")
orn_names=("orn1" "orn2" "orn3")

echo "Cleaning Database"
./delete_table_with_keyword.sh "143.248.57.18" "60004" "$1" &

echo "Starting Controller..."
config_path="../jsons/experiments/$2.json"
./run_controller.exp "$config_path" &
controller_pid=$!
sleep 1

for i in "${!agx[@]}"; do
  name="${agx_names[$i]}"
  host="${agx[$i]}"
  echo "Starting DeviceAgent on $name@$host"
  cmd="./DeviceAgent --name $name --device_type agxavier --controller_url 143.248.57.94 --dev_port_offset 0 --dev_verbose 1"
  ssh -t "root@$host" -i ~/.ssh/testbed "rm -r ~/pipe/models/fcpo_learning/hs* ; cd /home/cdsn/FCPO/build_host && $cmd" &
done

for i in "${!nx[@]}"; do
  name="${nx_names[$i]}"
  host="${nx[$i]}"
  echo "Starting DeviceAgent on $name@$host"
  cmd="./DeviceAgent --name $name --device_type nxavier --controller_url 143.248.57.94 --dev_port_offset 0 --dev_verbose 1"
  ssh -t "root@$host" -i ~/.ssh/testbed "rm -r ~/pipe/models/fcpo_learning/hs* ; cd /home/cdsn/FCPO/build_host && $cmd" &
done

for i in "${!orn[@]}"; do
  name="${orn_names[$i]}"
  host="${orn[$i]}"
  echo "Starting DeviceAgent on $name@$host"
  cmd="./DeviceAgent --name $name --device_type orinano --controller_url 143.248.57.94 --dev_port_offset 0 --dev_verbose 1"
  ssh -t "root@$host" -i ~/.ssh/testbed "rm -r ~/pipe/models/fcpo_learning/hs* ; cd /home/cdsn/FCPO/build_host && $cmd" &
done

host="143.248.57.73"
echo "Starting DeviceAgent on onprem1@$host"
cmd="./DeviceAgent --name onprem1 --device_type onprem --controller_url 143.248.57.94 --dev_port_offset 0 --dev_verbose 1"
ssh -t "root@$host" -i ~/.ssh/testbed "rm -r /ssd0/tung/PipePlusPlus/models/fcpo_learning/hs* ; cd /FCPO/build && $cmd" &

echo "Starting local DeviceAgent..."
../build/DeviceAgent --name server --device_type server --controller_url localhost --dev_port_offset 0 --dev_verbose 1 &
sleep 3

wait $controller_pid
echo "Controller process has finished."

./scripts/stop_containers_with_keywords.sh 'hs'

# if $3 exists, store the experiment
if [ -z "$3" ]; then
  echo "No storage path provided. Experiment complete."
  exit 0
fi

echo "Storing Experiment"
ssh -t "root@143.248.55.76" -i ~/.ssh/testbed "mv /ssd0/tung/PipePlusPlus/logs/hs*/fcpo/*  /ssd0/tung/PipePlusPlus/logs/$3"
echo "Experiment complete."

#!/bin/bash


KEYWORD=${1:-KEYWORD}
container_ids=$(docker ps -q --filter "name=${KEYWORD}")

if [ -n "$container_ids" ]; then
  for container_id in $container_ids; do
    docker stop "$container_id"
  done
  echo "Stopped containers matching '${KEYWORD}'"
else
  echo "No containers found matching '${KEYWORD}'"
fi

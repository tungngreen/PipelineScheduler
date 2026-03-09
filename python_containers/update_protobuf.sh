#!/bin/bash

protoc -I=../libs/utils/protobufprotocols/ --python_out=. controlmessages.proto

#!/bin/bash

# Default values
DEFAULT_HOURS="1"
DEFAULT_FILENAME="video1.mp4"
DEFAULT_URL='https://www.youtube.com/watch?v=dQw4w9WgXcQ'

# Command line arguments
HOURS=${1:-$DEFAULT_DB_HOST}
FILENAME=${2:-$DEFAULT_DB_PORT}
URL=${3:-$DEFAULT_PHRASE}

# Setup Environment before running
if [ ! -d "./venv" ]; then
    python3.10 -m venv venv
fi
source ./venv/bin/activate
pip install streamlink -q

# Record the Livestream / Video
streamlink --hls-live-edge 99999 --stream-segment-threads 5 --hls-duration 0$HOURS:00:00 -o $FILENAME $URL 720p
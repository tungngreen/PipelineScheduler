#!/bin/bash

# Check if the script is run with sudo
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo privileges."
    echo "Please run it again using: sudo $0"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    type "$1" &> /dev/null
}

# Check if TC is installed
if ! command_exists tc; then
    echo "TC (Traffic Control) is not installed. Please install it and try again."
    exit 1
fi

# Get parameters from command line
INTERFACE=$1
BANDWIDTH_MBPS=$2

# Check if the interface exists
if ! ip link show "$INTERFACE" &> /dev/null; then
    echo "Interface $INTERFACE does not exist."
    exit 1
fi

# Convert Mbps to kbps
BANDWIDTH_KBPS=$(echo "$BANDWIDTH_MBPS * 1000" | bc)

echo "Removing existing qdisc..."
tc qdisc del dev "$INTERFACE" root 2> /dev/null
echo "Existing qdisc removed."

echo "Adding new qdisc with bandwidth limit of ${BANDWIDTH_MBPS} Mbps..."
TC_OUTPUT=$(tc qdisc add dev "$INTERFACE" root handle 1: htb default 10 2>&1)
TC_STATUS=$?

if [ $TC_STATUS -eq 0 ]; then
    TC_OUTPUT+=$(tc class add dev "$INTERFACE" parent 1: classid 1:10 htb rate "${BANDWIDTH_KBPS}kbit" burst 15k 2>&1)
    TC_STATUS=$?
    if [ $TC_STATUS -eq 0 ]; then
        echo "Bandwidth successfully limited to $BANDWIDTH_MBPS Mbps on interface $INTERFACE"
    else
        echo "Failed to set bandwidth class. Error output:"
        echo "$TC_OUTPUT"
        exit 1
    fi
else
    echo "Failed to set qdisc. Error output:"
    echo "$TC_OUTPUT"
    exit 1
fi

echo "Current qdisc settings:"
tc qdisc show dev "$INTERFACE"

echo "Current class settings:"
tc class show dev "$INTERFACE"

echo "To remove this bandwidth limit, run: sudo tc qdisc del dev $INTERFACE root"

echo "Script completed. You can now test the bandwidth limit using iperf3 in another terminal."
echo "Remember to run iperf3 without sudo to test the applied limit."
import json

timestamps = []
bandwidths = []

# round bandwidths to 2 decimal places
rounded_bandwidths = []
for i in bandwidths:
    rounded_bandwidths.append(round(i, 2))
# read timestamps as hh.mm.ss and calculate the difference to the previous timestamp in seconds as integer
timestamp_uint = []
for i in timestamps:
    timestamp_uint.append(int(i[0:2])*3600 + int(i[3:5])*60 + int(i[6:8]))
timestamps_diff = []
for i, t in enumerate(timestamp_uint):
    if i == 0:
        timestamps_diff.append(0)
    else:
        timestamps_diff.append(t - timestamp_uint[i-1] + timestamps_diff[i-1])

# remove elements with the same timestamp_diff
timestamps_diff_unique = []
bandwidths_unique = []
for i, t in enumerate(timestamps_diff):
    if i == 0:
        timestamps_diff_unique.append(t)
        bandwidths_unique.append(rounded_bandwidths[i])
    else:
        if t != timestamps_diff_unique[-1]:
            timestamps_diff_unique.append(t)
            try:
                bandwidths_unique.append(rounded_bandwidths[i])
            except:
                bandwidths_unique.append(rounded_bandwidths[i-len(bandwidths_unique)+1])
        else:
            bandwidths_unique[-1] = bandwidths_unique[-1] + rounded_bandwidths[i] / 2

# place the values into 30 second chunks and average within them
timestamps_30 = []
bandwidths_30 = []
for i, t in enumerate(timestamps_diff_unique):
    if i == 0:
        timestamps_30.append(t)
        bandwidths_30.append(bandwidths_unique[i])
    else:
        if t < timestamps_30[-1] + 30:
            bandwidths_30[-1] = (bandwidths_30[-1] + bandwidths_unique[i]) / 2
        else:
            timestamps_30.append(t)
            bandwidths_30.append(bandwidths_unique[i])

# make sure each bandwidths_unique is larger than 0.01
for i, b in enumerate(bandwidths_30):
    if b < 0.01:
        bandwidths_30[i] = 0.01
    else:
        bandwidths_30[i] = round(b, 2)

# create json file like this:
# {
#     "interface": "eth0",
#     "bandwidth_limits": [
#       {"time": 0, "mbps":  bitrates[0]},
#       {"time": 0 + timestamp[1]-timestamp[0] in seconds, "mbps": bitrates[1]},
# ...
#       {"time": time[n-1] + timestamp[n]-timestamp[n-1] in seconds, "mbps": bitrates[n]},
#     ]
#   }
data = {"interface": "eth0", "bandwidth_limits": []}
for i, t in enumerate(timestamps_30):
    data["bandwidth_limits"].append({"time": t, "mbps": bandwidths_30[i]})
with open("bandwidth_limits.json", "w") as f:
    json.dump(data, f, indent=4)

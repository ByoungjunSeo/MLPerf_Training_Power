#!/bin/bash
while true; do
    output=$(sudo ipmitool dcmi power reading)
    power=$(echo "$output" | grep "Instantaneous" | awk '{print $4}')
    timestamp=$(echo "$output" | grep "IPMI timestamp" | cut -d':' -f2- | xargs)
    echo "Instantaneous power reading:                   ${power} Watts     IPMI timestamp:                           ${timestamp}"
    sleep 1
done >> IPMIPower.txt
```

출력 예시:
```
Instantaneous power reading:                   315 Watts     IPMI timestamp:                           Tue Jan 27 05:16:52 2026

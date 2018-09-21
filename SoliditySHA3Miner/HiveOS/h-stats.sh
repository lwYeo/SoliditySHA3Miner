#!/usr/bin/env bash

#######################
# MAIN script body
#######################

. /hive/custom/$CUSTOM_NAME/h-manifest.conf

# Stats from API
stats_raw=`curl --connect-timeout 3 --silent --noproxy '*' http://127.0.0.1:4078/`

if [[ $? -ne 0 || -z ${stats_raw} ]]; then
  echo -e "${YELLOW}Failed to read $miner from http://127.0.0.1:4078/${NOCOLOR}"

else
  khs=$(echo ${stats_raw} | jq .TotalHashRate)                               # Total hashrate
  local hashunit=$(echo ${stats_raw} | jq .HashRateUnit)                     # Hashrate unit

  if [[ "$hashunit" == *"GH"* ]]; then
    hashunit="ghs"
    khs=$(echo $khs*1000000 | jq -nf /dev/stdin)

  elif [[ "$hashunit" == *"MH"* ]]; then
    hashunit="mhs"
    khs=$(echo $khs*1000 | jq -nf /dev/stdin)

  elif [[ "$hashunit" == *"KH"* ]]; then
    hashunit="khs"

  else
    hashunit="hs"
    khs=$(echo $khs/1000 | jq -nf /dev/stdin)
  fi

  local temp=$(echo ${stats_raw} | jq [.Miners[].CurrentTemperatureC])       # GPU temp
  local fan=$(echo ${stats_raw} | jq [.Miners[].SettingFanLevelPercent])     # GPU fan speed
  local ac=$(echo ${stats_raw} | jq .AcceptedShares)                         # Accepted shares
  local rj=$(echo ${stats_raw} | jq .RejectedShares)                         # Rejected shares
  local algo="SoliditySHA3"                                                  # Algo
  local hs=$(echo ${stats_raw} | jq [.Miners[].HashRate])                    # GPU hashrate
  local uptime=$(echo ${stats_raw} | jq .Uptime)                             # Miner uptime

  # make JSON
  stats=$(jq -n \
             --argjson hs "$hs" \
             --arg hs_units "$hashunit" \
             --argjson temp "$temp" \
             --argjson fan "$fan" \
             --arg uptime "$uptime" \
             --arg ac "$ac" \
             --arg rj "$rj" \
             --arg algo "$algo" \
          '{$hs, $hs_units, $temp, $fan, $uptime, ar: [$ac, $rj], $algo}')
fi

[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats="null"

# debug outputs
#echo khs: "$khs"
#echo stats: "$stats"

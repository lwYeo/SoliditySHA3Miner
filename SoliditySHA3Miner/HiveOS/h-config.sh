#!/usr/bin/env bash
# This code is included in /hive/bin/custom function

[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${YELLOW}CUSTOM_TEMPLATE is empty${NOCOLOR}" && return 1
[[ -z $CUSTOM_URL ]] && echo -e "${YELLOW}CUSTOM_URL is empty${NOCOLOR}" && return 1

# make JSON
conf=$(jq -n \
  --arg isLogFile false \
  --arg minerJsonAPI "http://127.0.0.1:4078" \
  --arg minerCcminerAPI "127.0.0.1:4068" \
  --arg web3api "https://mainnet.infura.io/ANueYSYQTstCr2mFJjPE" \
  --arg contractAddress "0xB6eD7644C69416d67B522e20bC294A9a9B405B31" \
  --arg abiFile "0xBTC.abi" \
  --arg overrideMaxTarget "0x40000000000000000000000000000000000000000000000000000000000" \
  --arg customDifficulty 0 \
  --arg submitStale false \
  --arg maxScanRetry 5 \
  --arg pauseOnFailedScans 3 \
  --arg networkUpdateInterval 15000 \
  --arg hashrateUpdateInterval 30000 \
  --arg kingAddress "" \
  --arg minerAddress "${CUSTOM_TEMPLATE}" \
  --arg primaryPool "${CUSTOM_URL}" \
  --arg secondaryPool "" \
  --arg privateKey "" \
  --arg gasToMine 5.0 \
  --arg cpuMode false \
  --argjson cpuDevices [] \
  --arg allowIntel true \
  --argjson intelDevices [] \
  --arg allowAMD true \
  --argjson amdDevices [] \
  --arg allowCUDA true \
  --argjson cudaDevices [] \
'{$isLogFile, $minerJsonAPI, $minerCcminerAPI, $web3api, $contractAddress, $abiFile, $overrideMaxTarget, $customDifficulty, $submitStale, $maxScanRetry, $pauseOnFailedScans, $networkUpdateInterval, $hashrateUpdateInterval, $kingAddress, $minerAddress, $primaryPool, $secondaryPool, $privateKey, $gasToMine, $cpuMode, $cpuDevices, $allowIntel, $intelDevices, $allowAMD, $amdDevices, $allowCUDA, $cudaDevices}')

#replace tpl values in whole file
[[ -z $EWAL && -z $ZWAL && -z $DWAL ]] && echo -e "${RED}No WAL address is set${NOCOLOR}"
[[ ! -z $EWAL ]] && conf=$(sed "s/%EWAL%/$EWAL/g" <<< "$conf") #|| echo "${RED}EWAL not set${NOCOLOR}"
[[ ! -z $DWAL ]] && conf=$(sed "s/%DWAL%/$DWAL/g" <<< "$conf") #|| echo "${RED}DWAL not set${NOCOLOR}"
[[ ! -z $ZWAL ]] && conf=$(sed "s/%ZWAL%/$ZWAL/g" <<< "$conf") #|| echo "${RED}ZWAL not set${NOCOLOR}"
[[ ! -z $EMAIL ]] && conf=$(sed "s/%EMAIL%/$EMAIL/g" <<< "$conf")
[[ ! -z $WORKER_NAME ]] && conf=$(sed "s/%WORKER_NAME%/$WORKER_NAME/g" <<< "$conf") #|| echo "${RED}WORKER_NAME not set${NOCOLOR}"

[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME is set${NOCOLOR}" && return 1
echo "$conf" > $CUSTOM_CONFIG_FILENAME
echo "$CUSTOM_USER_CONFIG" > ${CUSTOM_CONFIG_FILENAME}.param

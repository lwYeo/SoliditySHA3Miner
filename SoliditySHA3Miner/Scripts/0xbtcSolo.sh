#!/usr/bin/env bash

command -v dotnet >/dev/null 2>&1 ||
{
 echo >&2 ".NET Core is not found or not installed,"
 echo >&2 "run 'sh install-deps.sh' to install dependencies.";
 read -p "Press any key to continue...";
 exit 1;
}
while : ; do
  if [ -f SoliditySHA3Miner.conf ] ; then
    rm -f SoliditySHA3Miner.conf
  fi
  dotnet SoliditySHA3Miner.dll allowCPU=false allowIntel=true allowAMD=true allowCUDA=true web3api=https://mainnet.infura.io/ANueYSYQTstCr2mFJjPE abiFile=0xBTC.abi contract=0xB6eD7644C69416d67B522e20bC294A9a9B405B31 gasToMine=3 gasApiMax=7 gasLimit=600000 gasApiURL=https://ethgasstation.info/json/ethgasAPI.json gasApiPath=$.safeLow gasApiMultiplier=0.1 gasApiOffset=0.5 privateKey=YOUR_ETH_PRIVATE_KEY
  [[ $? -eq 22 ]] || break
done

#!/usr/bin/env bash

command -v dotnet >/dev/null 2>&1 ||
{
 echo >&2 ".NET Core is not found or not installed,"
 echo >&2 "download and install from https://www.microsoft.com/net/download/linux/run";
 read -p "Press any key to continue...";
 exit 1;
}
  if [ -f SoliditySHA3Miner.conf ] ; then
    rm -f SoliditySHA3Miner.conf
  fi
dotnet SoliditySHA3Miner.dll web3api=https://mainnet.infura.io/ANueYSYQTstCr2mFJjPE abiFile=CLM.abi contract=0xc2981fc938A0e9D8de03e6f48740562B9e429D65 gasToMine=5 gasLimit=1704624 gasApiURL=https://ethgasstation.info/json/ethgasAPI.json gasApiPath=$.safeLow gasApiMultiplier=0.1 gasApiOffset=0.5 privateKey=YOUR_ETH_PRIVATE_KEY

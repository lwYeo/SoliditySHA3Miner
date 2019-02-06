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
  dotnet SoliditySHA3Miner.dll masterMode=false allowCPU=false allowIntel=true allowAMD=true allowCUDA=true masterURL=http://192.168.0.1:4080/ slaveUpdateInterval=5000
  [[ $? -eq 22 ]] || break
done

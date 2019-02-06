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
  dotnet SoliditySHA3Miner.dll allowCPU=false allowIntel=true allowAMD=true allowCUDA=true abiFile=0xBTC.abi contract=0xB6eD7644C69416d67B522e20bC294A9a9B405B31 overrideMaxTarget=27606985387162255149739023449108101809804435888681546220650096895197184 pool=http://mike.rs:8080 address=0x9172ff7884CEFED19327aDaCe9C470eF1796105c
  [[ $? -eq 22 ]] || break
done

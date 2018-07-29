@echo off
pushd %~dp0
:start
SoliditySHA3Miner.exe abiFile=0xBTC.abi contract=0xB6eD7644C69416d67B522e20bC294A9a9B405B31 gasToMine=5 privateKey=YOUR_ETH_PRIVATE_KEY
goto start
pause
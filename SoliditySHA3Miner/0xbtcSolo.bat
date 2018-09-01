@echo off
pushd %~dp0
dotnet SoliditySHA3Miner.dll web3api=https://mainnet.infura.io/ANueYSYQTstCr2mFJjPE abiFile=0xBTC.abi contract=0xB6eD7644C69416d67B522e20bC294A9a9B405B31 gasToMine=5 privateKey=YOUR_ETH_PRIVATE_KEY
pause
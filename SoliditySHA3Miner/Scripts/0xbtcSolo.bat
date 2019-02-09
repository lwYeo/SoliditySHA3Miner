@echo off
pushd %~dp0

for %%X in (dotnet.exe) do (set FOUND=%%~$PATH:X)
if defined FOUND (goto dotNetFound) else (goto dotNetNotFound)

:dotNetNotFound
echo .NET Core is not found or not installed,
echo download and install from https://www.microsoft.com/net/download/windows/run
goto end

:dotNetFound
:startMiner
DEL /F /Q SoliditySHA3Miner.conf

SoliditySHA3Miner.exe ^
allowCPU=false ^
allowIntel=true ^
allowAMD=true ^
allowCUDA=true ^
web3api=https://mainnet.infura.io/ANueYSYQTstCr2mFJjPE ^
abiFile=0xBTC.abi ^
contract=0xB6eD7644C69416d67B522e20bC294A9a9B405B31 ^
gasToMine=3 ^
gasApiMax=7 ^
gasLimit=600000 ^
gasApiURL=https://ethgasstation.info/json/ethgasAPI.json ^
gasApiPath=$.safeLow ^
gasApiMultiplier=0.1 ^
gasApiOffset=1.0 ^
privateKey=YOUR_ETH_PRIVATE_KEY

if %errorlevel% EQU 22 (
  goto startMiner
)
:end
pause
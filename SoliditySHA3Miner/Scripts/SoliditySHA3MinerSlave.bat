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

dotnet SoliditySHA3Miner.dll ^
masterMode=false ^
allowCPU=false ^
allowIntel=true ^
allowAMD=true ^
allowCUDA=true ^
masterURL=http://192.168.0.1:4080/ ^
slaveUpdateInterval=5000

if %errorlevel% EQU 22 (
  goto startMiner
)
:end
pause
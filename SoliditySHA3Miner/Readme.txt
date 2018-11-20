SoliditySHA3Miner
All-in-one mixed multi-GPU (nVidia, AMD, Intel) & CPU miner solves proof of work to mine supported EIP918 tokens in a single instance (with API).

Current latest public release version: 2.0.6
Runs on Windows 10, HiveOS, EthOS, and Ubuntu.

Built with .NET Core 2.1.5 SDK, VC++ 2017, gcc 4.8.5, nVidia CUDA SDK 9.2 64-bits, and AMD APP SDK v3.0.130.135 (OpenCL)
.NET Core 2.1 can be downloaded from https://www.microsoft.com/net/download
VC++ 2017 can be downloaded from https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads
CUDA 9.2 requires a minimum nVidia driver version of 396 [https://www.nvidia.com/drivers/beta]

LICENSE
SoliditySHA3Miner is licensed under the Apache License, Version 2.0 (the "License");
Libraries are included in the Software under the following license terms:
- libkeccak-tiny https://github.com/coruus/keccak-tiny/
- Nethereum https://github.com/Nethereum/Nethereum/blob/master/LICENSE.md
- Json.NET https://github.com/JamesNK/Newtonsoft.Json/blob/master/LICENSE.md
- Common Infrastructure Libraries for .NET http://netcommon.sourceforge.net/license.html
- Bouncy Castle https://www.bouncycastle.org/licence.html

Donation addresses
ETH (or any ERC 20/918 tokens)	: 0x9172ff7884CEFED19327aDaCe9C470eF1796105c
BTC                             : 3GS5J5hcG6Qcu9xHWGmJaV5ftWLmZuR255
LTC                             : LbFkAto1qYt8RdTFHL871H4djendcHyCyB

Usage: SoliditySHA3Miner [OPTIONS]
Options:
  help                    Display this help text and exit
  allowCPU                Allow to use CPU, may slow down system (default: false)
  cpuAffinity             Comma separated list of CPU affinity ID to use (default: all logical CPUs except first)
  allowIntel              Allow to use Intel GPU (OpenCL) (default: true)
  allowAMD                Allow to use AMD GPU (OpenCL) (default: true)
  allowCUDA               Allow to use Nvidia GPU (CUDA) (default: true)
  intelIntensity          GPU (Intel OpenCL) intensity (default: 17, decimals allowed)
  listAmdDevices          List of all AMD (OpenCL) devices in this system and exit (device ID: GPU name)
  amdDevice               Comma separated list of AMD (OpenCL) devices to use (default: all devices)
  amdIntensity            GPU (AMD OpenCL) intensity (default: 24.056, decimals allowed)
  listCudaDevices         List of all CUDA devices in this system (device ID: GPU name)
  cudaDevice              Comma separated list of CUDA devices to use (default: all devices)
  cudaIntensity           GPU (CUDA) intensity (default: auto, decimals allowed)
  minerJsonAPI            'http://IP:port/' for the miner JSON-API (default: http://127.0.0.1:4078), 0 disabled
  minerCcminerAPI         'IP:port' for the ccminer-style API (default: 127.0.0.1:4068), 0 disabled
  overrideMaxTarget       (Pool only) Use maximum target and skips query from web3
  customDifficulty        (Pool only) Set custom difficulity (check with your pool operator)
  maxScanRetry            Number of retries to scan for new work (default: 3)
  pauseOnFailedScans      Pauses mining after number of connection fails, including secondary and retries (default: 3)
  submitStale             Submit stale jobs, may create more rejected shares (default: false)
  abiFile                 Token abi in a file (default: 'ERC-541.abi' in the same folder as this miner)
  web3api                 User-defined web3 provider URL (default: Infura mainnet provider)
  contract                Token contract address (default: 0xbtc contract address)
  hashrateUpdateInterval  Interval (miliseconds) for GPU hashrate logs (default: 30000)
  networkUpdateInterval   Interval (miliseconds) to scan for new work (default: 15000)
  kingAddress             Add MiningKing address to nonce, only CPU mining supported (default: none)
  address                 (Pool only) Miner's ethereum address (default: developer's address)
  privateKey              (Solo only) Miner's private key
  gasToMine               (Solo only) Gas price to mine in GWei (default: 3, decimals allowed; note: will override lower dynamic gas price)
  gasLimit                (Solo only) Gas limit to submit proof of work (default: 1704624)
  gasApiURL               (Solo only) Get dynamic gas price to mine from this JSON API URL (note: leave empty to disable)
  gasApiPath              (Solo only) JSON path expression to retrieve dynamic gas price value from 'gasApiURL'
  gasApiMultiplier        (Solo only) Multiplier to dynamic gas price value from 'gasApiURL' => 'gasApiPath' (note: use 0.1 for EthGasStation API)
  gasApiOffset            (Solo only) Offset to dynamic gas price value from 'gasApiURL' => 'gasApiPath' (after 'gasApiMultiplier', decimals allowed)
  gasApiMax               (Solo only) Maximum gas price to mine in GWei from API (default: 7, decimals allowed)
  pool                    (Pool only) URL of pool mining server (default: http://mike.rs:8080)
  secondaryPool           (Optional) URL of failover pool mining server
  logFile                 Enables logging of console output to '{appPath}\\Log\\{yyyy-MM-dd}.log' (default: false)
  devFee                  Set developer fee in percentage (default: 2.0%, minimum: 1.5%)

NOTES
For HiveOS, refer to 'GuideForHiveOS.txt' on how to get started.
For EthOS, refer to 'GuideForEthOS.txt' on how to get started.
Do refer to 'GuideForPoolMining.txt' and 'GuideForSoloMining.txt' on how to get started.
Configuration is based on CLI (similar to ccminer), except ".abi" files are required for new tokens (You can manually create one and copy from etherscan.com -> Contract -> Code -> Contract ABI).
Note that there is a configuration file "SoliditySHA3Miner.conf" that saves previous CLI parameters/settings, delete it prior to changing CLI parameters.
A sample CLI launch parameter can be found in the ".bat" file found together with this miner, please refer to it if you need help.
You will have to supply your own Ethereum address (or Private key if you solo mine). It is your own responsibility to mine to the correct address/account.
It is recommended to use your own web3api (e.g. Geth / Parity) if you solo mine.
There is a default of 2.0% dev fee (Once every 50th nonce: starting from 11th if Pool mine, or starting from 50th if Solo mine).
You can set to the lowest 1.5% with "devFee=1.5" (the formula is "(nonce mod (100 / devFee)) = 0").
Dev fee in solo mining is by sending the current reward amount after the successful minted block, using the same gas fee as provided in 'gasToMine'.
In the case if the compute load for your GPU is not >= 99%, you can adjust the intensity via (amdIntensity/cudaIntensity/intelIntensity).

Please feedback your results and suggestions so that I can improve the miner. You can either add an issue in the repository, or find me in discord (Amano7). Thanks for trying out this miner!

CREDITS
Donations are encouraged to help support further development of this miner!
Many thanks to the following developers and testers in the 0xBitcoin discord :
Azlehria
mining-visualizer
LtTofu/Mag517
Infernal Toast
0x1d00ffff
Mikers
TwenteMining
Ghorge
BRob
Sly

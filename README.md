# SoliditySHA3Miner
Solves proof of work to mine supported ERC20/918 tokens.

Built with C#.NET 4.7.1, VC++ 2017 and nVidia CUDA SDK 9.2 64-bits (Windows 10 64-bit)

- .NET 4.7.1 can be downloaded from [https://microsoft.com/en-us/download/details.aspx?id=56116]

- VC++ 2017 can be downloaded from [https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads]

- CUDA 9.2 requires a minimum nVidia driver version of 396.xx [https://nvidia.com/drivers/results/134068]

### Releases can be found [here](https://github.com/lwYeo/SoliditySHA3Miner/releases).


## LICENSE

SoliditySHA3Miner is licensed under the [MIT license](https://github.com/lwYeo/SoliditySHA3Miner/blob/master/LICENSE).

Include CUDA kernel from Mikers, Azlehria and LtTofu (Mag517)

Libraries are included in the Software under the following license terms:

    Satoshi Nakamoto and The Bitcoin Core developers (uint256) [https://github.com/bitcoin/bitcoin/blob/master/COPYING]
    
    Projet RNRT SAPHIR (sphlib) [http://www.saphir2.com/sphlib/]
    
    Nethereum [https://github.com/Nethereum/Nethereum/blob/master/LICENSE.md]
    
    Json.NET [https://github.com/JamesNK/Newtonsoft.Json/blob/master/LICENSE.md]
    
    Common Infrastructure Libraries for .NET [http://netcommon.sourceforge.net/license.html]
    
    Bouncy Castle [https://www.bouncycastle.org/licence.html]
    

### Donation addresses

    ETH (or any ERC 20/918 tokens)  : 0x9172ff7884cefed19327adace9c470ef1796105c
    
    BTC                             : 3GS5J5hcG6Qcu9xHWGmJaV5ftWLmZuR255
    
    LTC                             : LbFkAto1qYt8RdTFHL871H4djendcHyCyB
    

Usage: SoliditySHA3Miner [OPTIONS]

Options:

    help                    Display this help text and exit
    
    listCudaDevices         List of all CUDA devices in this system (device ID: GPU name)
    
    cudaDevice              Comma separated list of CUDA devices to use (default: all devices)
    
    cudaIntensity           GPU (CUDA) intensity (default: auto, decimals allowed)
    
    minerJsonAPI            'http://IP:port/' for the miner JSON-API (default: http://127.0.0.1:4078), 0 disabled
    
    minerCcminerAPI         'IP:port' for the ccminer-style API (default: 127.0.0.1:4068), 0 disabled
	
    overrideMaxDiff         (Pool only) Use maximum difficulty and skips query from web3
    
    customDifficulty        (Pool only) Set custom difficulity (check with your pool operator)
    
    maxScanRetry            Number of retries to scan for new work (default: 5)
    
    pauseOnFailedScans      Pauses mining when connection fails, including secondary and retries (default: true)
    
    submitStale             Submit stale jobs, may create more rejected shares (default: true)
    
    abiFile                 Token abi in a file (default: '0xbtc.abi' in the same folder as this miner)
    
    web3api                 User-defined web3 provider URL (default: Infura mainnet provider)
    
    contract                Token contract address (default: 0xbtc contract address)
    
    hashrateUpdateInterval  Interval (miliseconds) for GPU hashrate logs (default: 30000)
    
    networkUpdateInterval   Interval (miliseconds) to scan for new work (default: 15000)
    
    address                 Miner's ethereum address (default: developer's address)
    
    pool                    (Pool only) URL of pool mining server (default: http://mike.rs:8080)
    
    secondaryPool           (Optional) URL of failover pool mining server
    
    donate                  Set donaton in percentage (default: 2%, minimum: 1.5)
    

using Nethereum.Hex.HexTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SoliditySHA3Miner
{
    public class Config
    {
        public bool isLogFile { get; set; }
        public string minerJsonAPI { get; set; }
        public string minerCcminerAPI { get; set; }
        public string web3api { get; set; }
        public string contractAddress { get; set; }
        public string abiFile { get; set; }
        public HexBigInteger overrideMaxTarget { get; set; }
        public ulong customDifficulty { get; set; }
        public bool submitStale { get; set; }
        public int maxScanRetry { get; set; }
        public int pauseOnFailedScans { get; set; }
        public int networkUpdateInterval { get; set; }
        public int hashrateUpdateInterval { get; set; }
        public string kingAddress { get; set; }
        public string minerAddress { get; set; }
        public string primaryPool { get; set; }
        public string secondaryPool { get; set; }
        public string privateKey { get; set; }
        public float gasToMine { get; set; }
        public bool cpuMode { get; set; }
        public Miner.Device[] cpuDevices { get; set; }
        public bool allowIntel { get; set; }
        public Miner.Device[] intelDevices { get; set; }
        public bool allowAMD { get; set; }
        public Miner.Device[] amdDevices { get; set; }
        public bool allowCUDA { get; set; }
        public Miner.Device[] cudaDevices { get; set; }

        public Config() // set defaults
        {
            isLogFile = false;
            minerJsonAPI = Defaults.JsonAPIPath;
            minerCcminerAPI = Defaults.CcminerAPIPath;
            web3api= Defaults.InfuraAPI_mainnet;
            contractAddress = Defaults.Contract0xBTC_mainnet;
            abiFile = Defaults.AbiFile0xBTC;
            overrideMaxTarget = new HexBigInteger(BigInteger.Zero);
            customDifficulty = 0u;
            submitStale = Defaults.SubmitStale;
            maxScanRetry = Defaults.MaxScanRetry;
            pauseOnFailedScans = Defaults.PauseOnFailedScan;
            networkUpdateInterval = Defaults.NetworkUpdateInterval;
            hashrateUpdateInterval = Defaults.HashrateUpdateInterval;
            kingAddress = string.Empty;
            minerAddress = string.Empty;
            primaryPool = string.Empty;
            secondaryPool = string.Empty;
            privateKey = string.Empty;
            gasToMine = Defaults.GasToMine;
            cpuMode = false;
            cpuDevices = new Miner.Device[] { };
            allowIntel = true;
            intelDevices = new Miner.Device[] { };
            allowAMD = true;
            amdDevices = new Miner.Device[] { };
            allowCUDA = true;
            cudaDevices = new Miner.Device[] { };
        }

        private static void PrintHelp()
        {
            var help =
                "Usage: SoliditySHA3Miner [OPTIONS]\n" +
                "Options:\n" +
                "  help                    Display this help text and exit\n" +
                "  cpuMode                 Set this miner to run in CPU mode only, disables GPU (default: false)\n" +
                "  cpuID                   Comma separated list of CPU thread ID to use (default: all logical CPUs except first)\n" +
                "  allowIntel              Allow to use Intel GPU (OpenCL) (default: true)\n" +
                "  allowAMD                Allow to use AMD GPU (OpenCL) (default: true)\n" +
                "  allowCUDA               Allow to use Nvidia GPU (CUDA) (default: true)\n" +
                "  intelIntensity          GPU (Intel OpenCL) intensity (default: 21, decimals allowed)\n" +
                "  listAmdDevices          List of all AMD (OpenCL) devices in this system and exit (device ID: GPU name)\n" +
                "  amdDevice               Comma separated list of AMD (OpenCL) devices to use (default: all devices)\n" +
                "  amdIntensity            GPU (AMD OpenCL) intensity (default: 24.223, decimals allowed)\n" +
                "  listCudaDevices         List of all CUDA devices in this system and exit (device ID: GPU name)\n" +
                "  cudaDevice              Comma separated list of CUDA devices to use (default: all devices)\n" +
                "  cudaIntensity           GPU (CUDA) intensity (default: auto, decimals allowed)\n" +
                "  minerJsonAPI            'http://IP:port/' for the miner JSON-API (default: " + Defaults.JsonAPIPath + "), 0 disabled\n" +
                "  minerCcminerAPI         'IP:port' for the ccminer-style API (default: " + Defaults.CcminerAPIPath + "), 0 disabled\n" +
                "  overrideMaxTarget       (Pool only) Use maximum target and skips query from web3\n" +
                "  customDifficulty        (Pool only) Set custom difficulity (check with your pool operator)\n" +
                "  maxScanRetry            Number of retries to scan for new work (default: " + Defaults.MaxScanRetry + ")\n" +
                "  pauseOnFailedScans      Pauses mining when connection fails, including secondary and retries (default: true)\n" +
                "  submitStale             Submit stale jobs, may create more rejected shares (default: " + Defaults.SubmitStale.ToString().ToLower() + ")\n" +
                "  abiFile                 Token abi in a file (default: '0xbtc.abi' in the same folder as this miner)\n" +
                "  web3api                 User-defined web3 provider URL (default: Infura mainnet provider)\n" +
                "  contract                Token contract address (default: 0xbtc contract address)\n" +
                "  hashrateUpdateInterval  Interval (miliseconds) for GPU hashrate logs (default: " + Defaults.HashrateUpdateInterval + ")\n" +
                "  networkUpdateInterval   Interval (miliseconds) to scan for new work (default: " + Defaults.NetworkUpdateInterval + ")\n" +
                "  kingAddress             Add MiningKing address to nonce, only CPU mining supported (default: none)\n" +
                "  address                 (Pool only) Miner's ethereum address (default: developer's address)\n" +
                "  privateKey              (Solo only) Miner's private key\n" +
                "  gasToMine               (Solo only) Gas price to mine in GWei (default: " + Defaults.GasToMine + ")\n" +
                "  pool                    (Pool only) URL of pool mining server (default: " + Defaults.PoolPrimary + ")\n" +
                "  secondaryPool           (Optional) URL of failover pool mining server\n" +
                "  logFile                 Enables logging of console output to '{appPath}\\Log\\{yyyy-MM-dd}.log' (default: false)\n" +
                "  devFee                  Set dev fee in percentage (default: " + DevFee.Percent + "%, minimum: " + DevFee.MinimumPercent + "%)\n";
            Console.WriteLine(help);
        }

        private static void PrintAmdDevices()
        {
            Miner.OpenCL.PreInitialize(true, out string initErrorMessage);
            if (!string.IsNullOrWhiteSpace(initErrorMessage)) Console.WriteLine(initErrorMessage);

            var amdDevices = Miner.OpenCL.GetDevices("AMD Accelerated Parallel Processing", out string getDevicesErrorMessage);
            if (!string.IsNullOrWhiteSpace(getDevicesErrorMessage)) Console.WriteLine(getDevicesErrorMessage);

            Console.WriteLine(amdDevices);
        }

        private static void PrintCudaDevices()
        {
            var cudaDevices = Miner.CUDA.GetDevices(out string errorMessage);
            Console.WriteLine(string.IsNullOrWhiteSpace(errorMessage) ? cudaDevices : errorMessage);
        }

        private void PrepareCpuDeviceList()
        {
            var cpuDeviceCount = Miner.CPU.GetLogicalProcessorCount();
            cpuDevices = (Miner.Device[])Array.CreateInstance(typeof(Miner.Device), cpuDeviceCount);
            for (int i = 0; i < cpuDevices.Length; i++)
            {
                cpuDevices[i].Type = "CPU";
                cpuDevices[i].DeviceID = -1;
            }
        }

        private void SetCpuDevices(string[] sCpuIDs)
        {
            PrepareCpuDeviceList();

            for (int i = 0; i < sCpuIDs.Length; i++)
            {
                if (string.IsNullOrEmpty(sCpuIDs[i])) continue;
                else cpuDevices[i].DeviceID = int.Parse(sCpuIDs[i]);
            }
        }

        private void PrepareAmdDeviceList()
        {
            var cudaDeviceCount = Miner.OpenCL.GetDeviceCount("AMD Accelerated Parallel Processing", out string cudaCountErrorMessage);
            amdDevices = (Miner.Device[])Array.CreateInstance(typeof(Miner.Device), cudaDeviceCount);
            for (int i = 0; i < amdDevices.Length; i++)
            {
                amdDevices[i].Type = "OpenCL";
                amdDevices[i].Platform = "AMD Accelerated Parallel Processing";
                amdDevices[i].DeviceID = -1;
            }
        }

        private bool SetAmdDevices(string[] sAmdDevices)
        {
            PrepareAmdDeviceList();

            for (int i = 0; i < amdDevices.Length; i++)
            {
                if (string.IsNullOrEmpty(sAmdDevices[i])) continue;
                else
                {
                    var deviceID = int.Parse(sAmdDevices[i]);
                    amdDevices[i].Name = Miner.CUDA.GetDeviceName(deviceID, out string errorMessage);
                    if (!string.IsNullOrEmpty(errorMessage))
                    {
                        Program.Print("OpenCL [ERROR] " + errorMessage);
                        return false;
                    }
                    amdDevices[i].DeviceID = int.Parse(sAmdDevices[i]);
                }
            }
            return true;
        }

        private void SetAmdIntensities(string[] sAmdIntensities)
        {
            if (amdDevices == null || !amdDevices.Any() || sAmdIntensities == null) return;

            if (sAmdIntensities.Length > 1)
                for (int i = 0; i < sAmdIntensities.Length; i++)
                    amdDevices[i].Intensity = float.Parse(sAmdIntensities[i]);
            else
                for (int i = 0; i < sAmdIntensities.Length; i++)
                    amdDevices[i].Intensity = float.Parse(sAmdIntensities[0]);
        }

        private void SetIntelIntensities(string[] sIntelIntensities)
        {
            if (intelDevices == null || !intelDevices.Any() || sIntelIntensities == null) return;

            if (sIntelIntensities.Length > 1)
                for (int i = 0; i < sIntelIntensities.Length; i++)
                    intelDevices[i].Intensity = float.Parse(sIntelIntensities[i]);
            else
                for (int i = 0; i < sIntelIntensities.Length; i++)
                    intelDevices[i].Intensity = float.Parse(sIntelIntensities[0]);
        }

        private void PrepareCudaDeviceList()
        {
            var cudaDeviceCount = Miner.CUDA.GetDeviceCount(out string cudaCountErrorMessage);
            cudaDevices = (Miner.Device[])Array.CreateInstance(typeof(Miner.Device), cudaDeviceCount);
            for (int i = 0; i < cudaDevices.Length; i++)
            {
                cudaDevices[i].Type = "CUDA";
                cudaDevices[i].DeviceID = -1;
            }
        }

        private bool SetCudaDevices(string[] sCudaDevices)
        {
            PrepareCudaDeviceList();

            for (int i = 0; i < sCudaDevices.Length; i++)
            {
                if (string.IsNullOrEmpty(sCudaDevices[i])) continue;
                else
                {
                    var deviceID = int.Parse(sCudaDevices[i]);
                    cudaDevices[i].Name = Miner.CUDA.GetDeviceName(deviceID, out string errorMessage);
                    if (!string.IsNullOrEmpty(errorMessage))
                    {
                        Program.Print("CUDA [ERROR] " + errorMessage);
                        return false;
                    }
                    cudaDevices[i].DeviceID = int.Parse(sCudaDevices[i]);
                }
            }
            return true;
        }

        private void SetCudaIntensities(string[] sCudaIntensities)
        {
            if (cudaDevices == null || !cudaDevices.Any() || sCudaIntensities == null) return;

            if (sCudaIntensities.Length > 1)
                for (int i = 0; i < sCudaIntensities.Length; i++)
                    cudaDevices[i].Intensity = float.Parse(sCudaIntensities[i]);
            else
                for (int i = 0; i < cudaDevices.Length; i++)
                    cudaDevices[i].Intensity = float.Parse(sCudaIntensities[0]);
        }

        private void CheckCPUConfig(string[] args)
        {
            if ((cpuDevices == null || !cpuDevices.Any()) && args.All(a => !a.StartsWith("cpuID")))
            {
                Program.Print("CPU [INFO] IDs not specified, default assign all logical CPUs except first.");
                var cpuCount = Miner.CPU.GetLogicalProcessorCount();
                if (cpuCount <= 0) cpuCount = 1;
                cpuDevices = (Miner.Device[])Array.CreateInstance(typeof(Miner.Device), cpuCount);

                for (int i = 0; i < cpuCount; i++)
                {
                    cpuDevices[i].Type = "CPU";
                    cpuDevices[i].DeviceID = (i < 1) ? -1 : i;
                }
            }
        }

        private void CheckAMDConfig(string[] args)
        {
            if (allowAMD || allowIntel)
            {
                try
                {
                    Miner.OpenCL.PreInitialize(allowIntel, out var openCLInitErrorMessage);
                    
                    if (!string.IsNullOrWhiteSpace(openCLInitErrorMessage))
                    {
                        if (openCLInitErrorMessage.Contains("Unable to load shared library"))
                            Program.Print("OpenCL [WARN] OpenCL not installed.");
                        else
                            Program.Print("OpenCL [ERROR] " + openCLInitErrorMessage);
                    }
                    else
                    {
                        if (allowIntel)
                        {
                            Program.Print("OpenCL [INFO] Assign all Intel(R) OpenCL devices.");
                            var deviceCount = Miner.OpenCL.GetDeviceCount("Intel(R) OpenCL", out var openCLerrorMessage);
                            if (!string.IsNullOrWhiteSpace(openCLerrorMessage)) Program.Print("OpenCL [WARN] " + openCLerrorMessage);
                            else
                            {
                                var tempIntelList = new List<Miner.Device>();
                                for (int i = 0; i < deviceCount; i++)
                                {
                                    var tempName = Miner.OpenCL.GetDeviceName("Intel(R) OpenCL", i, out var openCLdeviceErrorMessage);
                                    if (!string.IsNullOrWhiteSpace(openCLdeviceErrorMessage))
                                    {
                                        Program.Print("OpenCL [WARN] " + openCLdeviceErrorMessage);
                                        continue;
                                    }

                                    tempIntelList.Add(new Miner.Device
                                    {
                                        Type = "OpenCL",
                                        Platform = "Intel(R) OpenCL",
                                        DeviceID = i,
                                        Name = tempName
                                    });
                                }
                                intelDevices = tempIntelList.ToArray();
                            }
                        }

                        if (allowAMD && (amdDevices == null || !amdDevices.Any()) && args.All(a => !a.StartsWith("openclDevices")))
                        {
                            Program.Print("OpenCL [INFO] Device not specified, default assign all AMD APP devices.");

                            var deviceCount = Miner.OpenCL.GetDeviceCount("AMD Accelerated Parallel Processing", out var openCLerrorMessage);
                            if (!string.IsNullOrWhiteSpace(openCLerrorMessage)) Program.Print("OpenCL [WARN] " + openCLerrorMessage);
                            else
                            {
                                var tempAmdList = new List<Miner.Device>();
                                for (int i = 0; i < deviceCount; i++)
                                {
                                    var tempName = Miner.OpenCL.GetDeviceName("AMD Accelerated Parallel Processing", i, out var openCLdeviceErrorMessage);
                                    if (!string.IsNullOrWhiteSpace(openCLdeviceErrorMessage))
                                    {
                                        Program.Print("OpenCL [WARN] " + openCLdeviceErrorMessage);
                                        continue;
                                    }

                                    tempAmdList.Add(new Miner.Device
                                    {
                                        Type = "OpenCL",
                                        Platform = "AMD Accelerated Parallel Processing",
                                        DeviceID = i,
                                        Name = tempName
                                    });
                                }
                                amdDevices = tempAmdList.ToArray();
                            }
                        }
                    }
                }
                catch (DllNotFoundException)
                {
                    Program.Print("OpenCL [WARN] OpenCL not found.");
                }
                catch (Exception ex)
                {
                    Program.Print(ex.ToString());
                }
            }
        }

        private void CheckCUDAConfig(string[] args)
        {
            if (allowCUDA && (cudaDevices == null || !cudaDevices.Any()) && args.All(a => !a.StartsWith("cudaDevice")))
            {
                Program.Print("CUDA [INFO] Device not specified, default assign all CUDA devices.");
                var cudaDeviceCount = Miner.CUDA.GetDeviceCount(out string cudaCountErrorMessage);
                if (!string.IsNullOrWhiteSpace(cudaCountErrorMessage)) Program.Print("CUDA [ERROR] " + cudaCountErrorMessage);

                if (cudaDeviceCount > 0)
                {
                    cudaDevices = (Miner.Device[])Array.CreateInstance(typeof(Miner.Device), cudaDeviceCount);
                    for (int i = 0; i < cudaDevices.Length; i++)
                    {
                        cudaDevices[i].Type = "CUDA";
                        cudaDevices[i].DeviceID = i;
                        cudaDevices[i].Name = Miner.CUDA.GetDeviceName(i, out string errorMessage);
                    }
                }
                else
                {
                    Program.Print("CUDA [WARN] Device not found.");
                }
            }
        }

        private bool CheckConfig(string[] args)
        {
            try
            {
                if (networkUpdateInterval < 1000) networkUpdateInterval = 1000;
                if (hashrateUpdateInterval < 1000) hashrateUpdateInterval = 1000;
                
                if (string.IsNullOrEmpty(kingAddress))
                    Program.Print("[INFO] King making disabled.");
                else
                    Program.Print("[INFO] King making enabled, address: " + kingAddress);

                if (string.IsNullOrWhiteSpace(minerAddress) && string.IsNullOrWhiteSpace(privateKey))
                {
                    Program.Print("[INFO] Miner address not specified, donating 100% to dev.");
                    minerAddress = DevFee.Address;
                }

                if (!string.IsNullOrWhiteSpace(privateKey))
                {
                    Program.Print("[INFO] Solo mining mode.");
                }
                else if (string.IsNullOrWhiteSpace(primaryPool))
                {
                    Program.Print("[INFO] Primary pool not specified, using " + Defaults.PoolPrimary);
                    primaryPool = Defaults.PoolPrimary;
                }
                else
                {
                    Program.Print("[INFO] Primary pool specified, using " + primaryPool);

                    if (!string.IsNullOrWhiteSpace(secondaryPool))
                        Program.Print("[INFO] Secondary pool specified, using " + secondaryPool);
                }

                if (cpuMode)
                {
                    CheckCPUConfig(args);
                }
                else
                {
                    if (allowAMD || allowIntel) CheckAMDConfig(args);
                    if (allowCUDA) CheckCUDAConfig(args);

                    foreach (var arg in args)
                    {
                        try
                        {
                            switch (arg.Split('=')[0])
                            {
                                case "intelIntensity":
                                    if (!allowIntel || arg.EndsWith('=')) break;
                                    SetIntelIntensities(arg.Split('=')[1].Split(','));
                                    break;

                                case "amdIntensity":
                                    if (!allowAMD || arg.EndsWith('=')) break;
                                    SetAmdIntensities(arg.Split('=')[1].Split(','));
                                    break;

                                case "cudaIntensity":
                                    if (!allowCUDA || arg.EndsWith('=')) break;
                                    SetCudaIntensities(arg.Split('=')[1].Split(','));
                                    break;
                            }
                        }
                        catch (Exception)
                        {
                            Program.Print("[ERROR] Failed parsing argument: " + arg);
                            return false;
                        }
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                Program.Print("[ERROR] " + ex.Message);
                return false;
            }
        }

        public bool ParseArgumentsToConfig(string[] args)
        {
            foreach (var arg in args)
                {
                    try
                    {
                        switch (arg.Split('=')[0])
                        {
                            case "help":
                                PrintHelp();
                                Environment.Exit(0);
                                break;

                            case "cpuMode":
                                cpuMode = bool.Parse(arg.Split('=')[1]);
                                break;

                            case "cpuID":
                                SetCpuDevices(arg.Split('=')[1].Split(','));
                                break;

                            case "allowIntel":
                                allowIntel = bool.Parse(arg.Split('=')[1]);
                                break;

                            case "allowAMD":
                                allowAMD = bool.Parse(arg.Split('=')[1]);
                                break;

                            case "allowCUDA":
                                allowCUDA = bool.Parse(arg.Split('=')[1]);
                                break;

                            case "listAmdDevices":
                                PrintAmdDevices();
                                Environment.Exit(0);
                                break;

                            case "amdDevice":
                                if (!SetAmdDevices(arg.Split('=')[1].Split(',')))
                                    return false;
                                break;

                            case "listCudaDevices":
                                PrintCudaDevices();
                                Environment.Exit(0);
                                break;

                            case "cudaDevice":
                                if (!SetCudaDevices(arg.Split('=')[1].Split(',')))
                                    return false;
                                break;

                            case "minerJsonAPI":
                                minerJsonAPI = arg.Split('=')[1];
                                break;

                            case "minerCcminerAPI":
                                minerCcminerAPI = arg.Split('=')[1];
                                break;

                            case "overrideMaxTarget":
                                var strValue = arg.Split('=')[1];
                                overrideMaxTarget = strValue.StartsWith("0x")
                                                            ? new HexBigInteger(strValue)
                                                            : new HexBigInteger(BigInteger.Parse(strValue));
                                break;

                            case "customDifficulty":
                                customDifficulty = uint.Parse(arg.Split('=')[1]);
                                break;

                            case "maxScanRetry":
                                maxScanRetry = int.Parse(arg.Split('=')[1]);
                                break;

                            case "pauseOnFailedScans":
                                pauseOnFailedScans = int.Parse(arg.Split('=')[1]);
                                break;

                            case "submitStale":
                                submitStale = bool.Parse(arg.Split('=')[1]);
                                break;

                            case "abiFile":
                                abiFile = arg.Split('=')[1];
                                break;

                            case "web3api":
                                web3api = arg.Split('=')[1];
                                break;

                            case "contract":
                                contractAddress = arg.Split('=')[1];
                                break;

                            case "networkUpdateInterval":
                                networkUpdateInterval = int.Parse(arg.Split('=')[1]);
                                break;

                            case "hashrateUpdateInterval":
                                hashrateUpdateInterval = int.Parse(arg.Split('=')[1]);
                                break;

                            case "kingAddress":
                                kingAddress = arg.Split('=')[1];
                                break;

                            case "address":
                                minerAddress = arg.Split('=')[1];
                                break;

                            case "privateKey":
                                privateKey = arg.Split('=')[1];
                                break;

                            case "gasToMine":
                                gasToMine = float.Parse(arg.Split('=')[1]);
                                break;

                            case "pool":
                                primaryPool = arg.Split('=')[1];
                                break;

                            case "secondaryPool":
                                secondaryPool = arg.Split('=')[1];
                                break;

                            case "devFee":
                                DevFee.UserPercent = float.Parse(arg.Split('=')[1]);
                                break;
                        }
                    }
                    catch (Exception)
                    {
                        Program.Print("[ERROR] Failed parsing argument: " + arg);
                        return false;
                    }
                }

                return CheckConfig(args);
        }

        public static class Defaults
        {
            public const string InfuraAPI_mainnet = "https://mainnet.infura.io/ANueYSYQTstCr2mFJjPE";
            public const string InfuraAPI_ropsten = "https://ropsten.infura.io/ANueYSYQTstCr2mFJjPE";
            public const string Contract0xBTC_mainnet = "0xB6eD7644C69416d67B522e20bC294A9a9B405B31";
            public const string Contract0xBTC_ropsten = "0x9D2Cc383E677292ed87f63586086CfF62a009010";
            public const string AbiFile0xBTC = "0xbtc.abi";

            public const string PoolPrimary = "http://mike.rs:8080";
            public const string PoolSecondary = "http://mike.rs:8080";
            public const string JsonAPIPath = "http://127.0.0.1:4078";
            public const string CcminerAPIPath = "127.0.0.1:4068";

            public const bool SubmitStale = false;
            public const float GasToMine = 5.0f;
            public const int MaxScanRetry = 3;
            public const int PauseOnFailedScan = 3;
            public const int NetworkUpdateInterval = 15000;
            public const int HashrateUpdateInterval = 30000;

            public const bool LogFile = false;
        }
    }
}
            
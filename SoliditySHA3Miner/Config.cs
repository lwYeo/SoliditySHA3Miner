/*
   Copyright 2018 Lip Wee Yeo Amano

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

using Nethereum.Hex.HexTypes;
using SoliditySHA3Miner.Miner.Helper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;

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
        public ulong gasLimit { get; set; }
        public string gasApiURL { get; set; }
        public string gasApiPath { get; set; }
        public float gasApiMultiplier { get; set; }
        public float gasApiOffset { get; set; }
        public float gasApiMax { get; set; }
        public bool allowCPU { get; set; }
        public Miner.Device.CPU cpuDevice { get; set; }
        public bool allowIntel { get; set; }
        public Miner.Device.OpenCL[] intelDevices { get; set; }
        public bool allowAMD { get; set; }
        public Miner.Device.OpenCL[] amdDevices { get; set; }
        public bool allowCUDA { get; set; }
        public Miner.Device.CUDA[] cudaDevices { get; set; }

        public Config() // set defaults
        {
            isLogFile = false;
            minerJsonAPI = Defaults.JsonAPIPath;
            minerCcminerAPI = Defaults.CcminerAPIPath;
            web3api = Defaults.InfuraAPI_mainnet;
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
            gasLimit = Defaults.GasLimit;
            gasApiMax = Defaults.GasApiMax;
            allowCPU = false;
            cpuDevice = new Miner.Device.CPU();
            allowIntel = true;
            intelDevices = new Miner.Device.OpenCL[] { };
            allowAMD = true;
            amdDevices = new Miner.Device.OpenCL[] { };
            allowCUDA = true;
            cudaDevices = new Miner.Device.CUDA[] { };
        }

        private static void PrintHelp()
        {
            var help =
                "Usage: SoliditySHA3Miner [OPTIONS]\n" +
                "Options:\n" +
                "  help                    Display this help text and exit\n" +
                "  allowCPU                Allow to use CPU, may slow down system (default: false)\n" +
                "  cpuAffinity             Comma separated list of CPU affinity ID to use (default: all odd number logical processors)\n" +
                "  allowIntel              Allow to use Intel GPU (OpenCL) (default: true)\n" +
                "  allowAMD                Allow to use AMD GPU (OpenCL) (default: true)\n" +
                "  allowCUDA               Allow to use Nvidia GPU (CUDA) (default: true)\n" +
                "  intelIntensity          GPU (Intel OpenCL) intensity (default: 17, decimals allowed)\n" +
                "  listAmdDevices          List of all AMD (OpenCL) devices in this system and exit (device ID: GPU name)\n" +
                "  amdDevice               Comma separated list of AMD (OpenCL) devices to use (default: all devices)\n" +
                "  amdIntensity            GPU (AMD OpenCL) intensity (default: 24.056, decimals allowed)\n" +
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
                "  abiFile                 Token abi in a file (default: 'ERC-541.abi' in the same folder as this miner)\n" +
                "  web3api                 User-defined web3 provider URL (default: Infura mainnet provider)\n" +
                "  contract                Token contract address (default: 0xbtc contract address)\n" +
                "  hashrateUpdateInterval  Interval (miliseconds) for GPU hashrate logs (default: " + Defaults.HashrateUpdateInterval + ")\n" +
                "  networkUpdateInterval   Interval (miliseconds) to scan for new work (default: " + Defaults.NetworkUpdateInterval + ")\n" +
                "  kingAddress             Add MiningKing address to nonce, only CPU mining supported (default: none)\n" +
                "  address                 (Pool only) Miner's ethereum address (default: developer's address)\n" +
                "  privateKey              (Solo only) Miner's private key\n" +
                "  gasToMine               (Solo only) Gas price to mine in GWei (default: " + Defaults.GasToMine + ", decimals allowed; note: will override lower dynamic gas price)\n" +
                "  gasLimit                (Solo only) Gas limit to submit proof of work (default: " + Defaults.GasLimit + ")\n" +
                "  gasApiURL               (Solo only) Get dynamic gas price to mine from this JSON API URL (note: leave empty to disable)\n" +
                "  gasApiPath              (Solo only) JSON path expression to retrieve dynamic gas price value from 'gasApiURL'\n" +
                "  gasApiMultiplier        (Solo only) Multiplier to dynamic gas price value from 'gasApiURL' => 'gasApiPath' (note: use 0.1 for EthGasStation API)\n" +
                "  gasApiOffset            (Solo only) Offset to dynamic gas price value from 'gasApiURL' => 'gasApiPath' (after 'gasApiMultiplier', decimals allowed)\n" +
                "  gasApiMax               (Solo only) Maximum gas price to mine in GWei from API (default: " + Defaults.GasApiMax + ", decimals allowed)" +
                "  pool                    (Pool only) URL of pool mining server (default: " + Defaults.PoolPrimary + ")\n" +
                "  secondaryPool           (Optional) URL of failover pool mining server\n" +
                "  logFile                 Enables logging of console output to '{appPath}\\Log\\{yyyy-MM-dd}.log' (default: false)\n" +
                "  devFee                  Set dev fee in percentage (default: " + DevFee.Percent + "%, minimum: " + DevFee.MinimumPercent + "%)\n";
            Console.WriteLine(help);
        }

        private static void PrintAmdDevices()
        {
            var maxPlatformCount = 5u;
            var maxDeviceCount = 64u;

            var platformCount = 0u;
            var platformPointer = IntPtr.Zero;
            var amdDevices = new StringBuilder();
            var errorMessage = new StringBuilder(1024);

            OpenCL.Solver.PreInitialize(null, null, 0, 0);
            OpenCL.Solver.GetPlatforms(ref platformPointer, maxPlatformCount, ref platformCount, errorMessage);
            if (errorMessage.Length > 0)
            {
                Program.Print("OpenCL [ERROR] " + errorMessage.ToString());
                return;
            }

            var platforms = (Structs.DeviceCL.Platform[])Array.CreateInstance(typeof(Structs.DeviceCL.Platform), platformCount);
            unsafe
            {
                Structs.DeviceCL.Platform* tempPlatforms = (Structs.DeviceCL.Platform*)platformPointer.ToPointer();
                for (var i = 0; i < platformCount; i++)
                    platforms[i] = tempPlatforms[i];
            }

            var amdPlatform = platforms.FirstOrDefault(p => string.Concat(p.NameToString()).IndexOf("AMD Accelerated Parallel Processing", StringComparison.OrdinalIgnoreCase) > -1);
            if (amdPlatform.ID != IntPtr.Zero)
            {
                var deviceCount = 0u;
                var devicesPointer = IntPtr.Zero;

                OpenCL.Solver.GetDevicesByPlatform(amdPlatform, maxDeviceCount, ref deviceCount, ref devicesPointer, errorMessage);
                if (errorMessage.Length > 0)
                {
                    var errMessage = errorMessage.ToString();

                    if (errMessage.IndexOf("CL_DEVICE_NOT_FOUND") > -1)
                        Program.Print("No AMD device(s) found.");
                    else
                        Program.Print("OpenCL [ERROR] " + errMessage);

                    return;
                }

                var devices = (Structs.DeviceCL[])Array.CreateInstance(typeof(Structs.DeviceCL), deviceCount);
                unsafe
                {
                    Structs.DeviceCL* tempDevices = (Structs.DeviceCL*)platformPointer.ToPointer();
                    for (var i = 0; i < deviceCount; i++)
                        devices[i] = tempDevices[i];
                }

                for (var i = 0; i < deviceCount; i++)
                    amdDevices.AppendLine(string.Format("{0}: {1}", i, devices[i].NameToString()));
            }

            if (amdDevices.Length > 0)
                Console.WriteLine(amdDevices.ToString());
            else
                Console.WriteLine("No AMD device(s) found.");
        }

        private static void PrintCudaDevices()
        {
            var cudaDevices = GetCudaDeviceList(out string errorMessage);

            if (string.IsNullOrWhiteSpace(errorMessage))
            {
                var outputString = string.Empty;

                foreach (var device in cudaDevices)
                {
                    if (!string.IsNullOrWhiteSpace(outputString)) outputString += Environment.NewLine;
                    outputString += string.Format("{0}: {1}", device.Item1, device.Item2);
                }
                Console.WriteLine(outputString);
            }
            else { Console.WriteLine(errorMessage); }
        }

        private static List<Tuple<int, string>> GetCudaDeviceList(out string errorMessage)
        {
            errorMessage = string.Empty;
            var deviceList = new List<Tuple<int, string>>();

            var errMsg = new StringBuilder(1024);
            var deviceName = new StringBuilder(256);

            var cudaCount = 0;
            CUDA.Solver.GetDeviceCount(ref cudaCount, errMsg);
            errorMessage = errMsg.ToString();

            if (!string.IsNullOrEmpty(errorMessage)) return deviceList;

            for (int i = 0; i < cudaCount; i++)
            {
                errMsg.Clear();
                deviceName.Clear();

                CUDA.Solver.GetDeviceName(i, deviceName, errMsg);
                errorMessage = errMsg.ToString();

                if (!string.IsNullOrEmpty(errorMessage)) return deviceList;

                deviceList.Add(new Tuple<int, string>(i, deviceName.ToString()));
            }
            return deviceList;
        }

        private void SetCpuAffinity(int[] iCpuAffinity)
        {
            cpuDevice.Affinities = iCpuAffinity.Distinct().Where(i => i > -1).OrderBy(i => i).ToArray();
            cpuDevice.AllowDevice = cpuDevice.Affinities.Any();
        }

        private void SetAmdDevices(uint[] iAmdDevices)
        {
            for (uint i = 0; i < amdDevices.Length; i++)
                amdDevices[i].AllowDevice = iAmdDevices.Any(id => id == i);
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

        private void SetCudaDevices(uint[] iCudaDevices)
        {
            for (uint i = 0; i < cudaDevices.Length; i++)
                cudaDevices[i].AllowDevice = iCudaDevices.Any(id => id == i);
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
            if (!Program.AllowCPU)
                return;

            if (args.All(a => !a.StartsWith("cpuAffinity")))
                Program.Print("CPU [INFO] Processor affinity not specified, default assign all odd number logical processors.");

            if ((cpuDevice.Affinities == null || !cpuDevice.Affinities.Any()) && args.All(a => !a.StartsWith("cpuAffinity")))
            {
                cpuDevice.Type = "CPU";
                cpuDevice.Affinities = Environment.ProcessorCount > 1
                                     ? (int[])Array.CreateInstance(typeof(int), Environment.ProcessorCount / 2)
                                     : Enumerable.Empty<int>().ToArray();
                cpuDevice.AllowDevice = cpuDevice.Affinities.Any();

                for (int i = 0; i < cpuDevice.Affinities.Length; i++)
                    cpuDevice.Affinities[i] = (i * 2) + 1;
            }
        }

        private void CheckOpenCLConfig(string[] args)
        {
            if (!Program.AllowAMD && !Program.AllowIntel)
                return;

            if (args.All(a => !a.StartsWith("amdDevice")))
                Program.Print("OpenCL [INFO] AMD APP device not specified, default assign all AMD APP devices.");

            try
            {
                var maxPlatformCount = 5u;
                var maxDeviceCount = 64u;

                var sha3Kernel = new StringBuilder(Properties.Resources.ResourceManager.GetString("sha3Kernel"));
                var sha3KingKernel = new StringBuilder(Properties.Resources.ResourceManager.GetString("sha3KingKernel"));
                OpenCL.Solver.PreInitialize(sha3Kernel, sha3KingKernel, (ulong)sha3Kernel.Length, (ulong)sha3KingKernel.Length);

                var platformCount = 0u;
                var platformPointer = IntPtr.Zero;
                var errorMessage = new StringBuilder(1024);

                OpenCL.Solver.GetPlatforms(ref platformPointer, maxPlatformCount, ref platformCount, errorMessage);
                if (errorMessage.Length > 0)
                {
                    Program.Print("OpenCL [ERROR] " + errorMessage.ToString());
                    return;
                }

                var platforms = (Structs.DeviceCL.Platform[])Array.CreateInstance(typeof(Structs.DeviceCL.Platform), platformCount);
                unsafe
                {
                    Structs.DeviceCL.Platform* tempPlatforms = (Structs.DeviceCL.Platform*)platformPointer.ToPointer();
                    for (var i = 0; i < platformCount; i++)
                        platforms[i] = tempPlatforms[i];
                }

                var amdPlatform = platforms.FirstOrDefault(p => string.Concat(p.NameToString()).IndexOf("AMD Accelerated Parallel Processing", StringComparison.OrdinalIgnoreCase) > -1);
                var intelPlatform = platforms.FirstOrDefault(p => string.Concat(p.NameToString()).IndexOf("Intel(R) OpenCL", StringComparison.OrdinalIgnoreCase) > -1);
                var cudaPlatform = platforms.FirstOrDefault(p => string.Concat(p.NameToString()).IndexOf("NVIDIA CUDA", StringComparison.OrdinalIgnoreCase) > -1);

                if (Program.AllowAMD && amdPlatform.ID != IntPtr.Zero)
                {
                    var deviceCount = 0u;
                    var devicesPointer = IntPtr.Zero;
                    OpenCL.Solver.GetDevicesByPlatform(amdPlatform, maxDeviceCount, ref deviceCount, ref devicesPointer, errorMessage);

                    if (errorMessage.Length > 0)
                    {
                        var errMessage = errorMessage.ToString();

                        if (errMessage.IndexOf("CL_DEVICE_NOT_FOUND") > -1)
                            Program.Print("OpenCL [WARN] " + errMessage);
                        else
                            Program.Print("OpenCL [ERROR] " + errMessage);

                        errorMessage.Clear();
                    }
                    else
                    {
                        if (deviceCount < 1)
                            intelDevices = (Miner.Device.OpenCL[])Array.CreateInstance(typeof(Miner.Device.OpenCL), 0);
                        else
                        {
                            var devices = (Structs.DeviceCL[])Array.CreateInstance(typeof(Structs.DeviceCL), deviceCount);
                            unsafe
                            {
                                Structs.DeviceCL* tempDevices = (Structs.DeviceCL*)devicesPointer.ToPointer();
                                for (var i = 0; i < deviceCount; i++)
                                    devices[i] = tempDevices[i];
                            }

                            var tempAmdList = new List<Miner.Device.OpenCL>((int)deviceCount);
                            for (int i = 0; i < deviceCount; i++)
                            {
                                var userDevice = amdDevices?.FirstOrDefault(d => d.DeviceID.Equals(i));

                                tempAmdList.Add(new Miner.Device.OpenCL
                                {
                                    AllowDevice = userDevice?.AllowDevice ?? true,
                                    DeviceCL_Struct = devices[i],
                                    Type = "OpenCL",
                                    Platform = "AMD Accelerated Parallel Processing",
                                    DeviceID = i,
                                    PciBusID = userDevice?.PciBusID ?? 0,
                                    Name = devices[i].NameToString(),
                                    Intensity = userDevice?.Intensity ?? 0
                                });
                            }
                            amdDevices = tempAmdList.ToArray();
                        }
                    }
                }

                if (Program.AllowIntel && intelPlatform.ID != IntPtr.Zero)
                {
                    Program.Print("OpenCL [INFO] Assign all Intel(R) OpenCL devices.");

                    var deviceCount = 0u;
                    var devicesPointer = IntPtr.Zero;
                    OpenCL.Solver.GetDevicesByPlatform(intelPlatform, maxDeviceCount, ref deviceCount, ref devicesPointer, errorMessage);

                    if (errorMessage.Length > 0)
                    {
                        var errMessage = errorMessage.ToString();

                        if (errMessage.IndexOf("CL_DEVICE_NOT_FOUND") > -1)
                            Program.Print("OpenCL [WARN] " + errMessage);
                        else
                            Program.Print("OpenCL [ERROR] " + errMessage);

                        errorMessage.Clear();
                    }
                    else
                    {
                        if (deviceCount < 1)
                            intelDevices = (Miner.Device.OpenCL[])Array.CreateInstance(typeof(Miner.Device.OpenCL), 0);
                        else
                        {
                            var devices = (Structs.DeviceCL[])Array.CreateInstance(typeof(Structs.DeviceCL), deviceCount);
                            unsafe
                            {
                                Structs.DeviceCL* tempDevices = (Structs.DeviceCL*)devicesPointer.ToPointer();
                                for (var i = 0; i < deviceCount; i++)
                                    devices[i] = tempDevices[i];
                            }

                            var tempIntelList = new List<Miner.Device.OpenCL>((int)deviceCount);
                            for (int i = 0; i < deviceCount; i++)
                            {
                                var userDevice = intelDevices?.FirstOrDefault(d => d.DeviceID.Equals(i));

                                tempIntelList.Add(new Miner.Device.OpenCL
                                {
                                    AllowDevice = userDevice?.AllowDevice ?? true,
                                    DeviceCL_Struct = devices[i],
                                    Type = "OpenCL",
                                    Platform = "Intel(R) OpenCL",
                                    DeviceID = i,
                                    Name = devices[i].NameToString(),
                                    Intensity = userDevice?.Intensity ?? 0
                                });
                            }
                            intelDevices = tempIntelList.ToArray();
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

        private void CheckCUDAConfig(string[] args)
        {
            if (!Program.AllowCUDA)
                return;

            if (args.All(a => !a.StartsWith("cudaDevice")))
                Program.Print("CUDA [INFO] Device not specified, default assign all CUDA devices.");

            var errorMessage = new StringBuilder(1024);

            var cudaDeviceCount = 0;
            CUDA.Solver.GetDeviceCount(ref cudaDeviceCount, errorMessage);

            if (errorMessage.Length > 0) Program.Print("CUDA [ERROR] " + errorMessage.ToString());

            if (cudaDeviceCount > 0)
            {
                var tempCudaList = new List<Miner.Device.CUDA>();
                for (int i = 0; i < cudaDeviceCount; i++)
                {
                    var userDevice = cudaDevices?.FirstOrDefault(d => d.DeviceID.Equals(i));
                    var userIntensity = userDevice?.Intensity ?? 0;
                    var userAllowDevice = userDevice?.AllowDevice ?? true;
                    var userPciBusID = userDevice?.PciBusID ?? 0;

                    errorMessage.Clear();
                    var deviceName = new StringBuilder(256);
                    CUDA.Solver.GetDeviceName(i, deviceName, errorMessage);

                    if (errorMessage.Length > 0) Program.Print("CUDA [ERROR] " + errorMessage.ToString());

                    tempCudaList.Add(new Miner.Device.CUDA
                    {
                        AllowDevice = true && userAllowDevice,
                        Type = "CUDA",
                        DeviceID = i,
                        PciBusID = userPciBusID,
                        Name = deviceName.ToString(),
                        Intensity = userIntensity
                    });
                }
                cudaDevices = tempCudaList.ToArray();
            }
            else
            {
                Program.Print("CUDA [WARN] Device not found.");
                cudaDevices = (Miner.Device.CUDA[])Array.CreateInstance(typeof(Miner.Device.CUDA), 0);
            }
        }

        private bool CheckConfig(string[] args)
        {
            try
            {
                Program.AllowCPU = allowCPU;
                Program.AllowIntel = allowIntel;
                Program.AllowAMD = allowAMD;
                Program.AllowCUDA = allowCUDA;

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

                CheckCUDAConfig(args);
                CheckOpenCLConfig(args);
                CheckCPUConfig(args);

                foreach (var arg in args)
                {
                    try
                    {
                        switch (arg.Split('=')[0])
                        {
                            case "cpuAffinity":
                                SetCpuAffinity(arg.Split('=')[1].Split(',').Select(s => int.Parse(s)).ToArray());
                                break;

                            case "amdDevice":
                                SetAmdDevices(arg.Split('=')[1].Split(',').Select(s => uint.Parse(s)).ToArray());
                                break;

                            case "cudaDevice":
                                SetCudaDevices(arg.Split('=')[1].Split(',').Select(s => uint.Parse(s)).ToArray());
                                break;
                        }
                    }
                    catch (Exception)
                    {
                        Program.Print("[ERROR] Failed parsing argument: " + arg);
                        return false;
                    }
                }

                foreach (var arg in args)
                {
                    try
                    {
                        switch (arg.Split('=')[0])
                        {
                            case "intelIntensity":
                                if (!Program.AllowIntel || arg.EndsWith("=")) break;
                                SetIntelIntensities(arg.Split('=')[1].Split(','));
                                break;

                            case "amdIntensity":
                                if (!Program.AllowAMD || arg.EndsWith("=")) break;
                                SetAmdIntensities(arg.Split('=')[1].Split(','));
                                break;

                            case "cudaIntensity":
                                if (!Program.AllowCUDA || arg.EndsWith("=")) break;
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

                        case "allowCPU":
                            allowCPU = bool.Parse(arg.Split('=')[1]);
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

                        case "listCudaDevices":
                            PrintCudaDevices();
                            Environment.Exit(0);
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

                        case "gasLimit":
                            gasLimit = ulong.Parse(arg.Split('=')[1]);
                            break;

                        case "gasApiURL":
                            gasApiURL = arg.Split('=')[1];
                            break;

                        case "gasApiPath":
                            gasApiPath = arg.Split('=')[1];
                            break;

                        case "gasApiMultiplier":
                            gasApiMultiplier = float.Parse(arg.Split('=')[1]);
                            break;

                        case "gasApiOffset":
                            gasApiOffset = float.Parse(arg.Split('=')[1]);
                            break;

                        case "gasApiMax":
                            gasApiMax = float.Parse(arg.Split('=')[1]);
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
            public const string AbiFile0xBTC = "ERC-541.abi";

            public const string PoolPrimary = "http://mike.rs:8080";
            public const string PoolSecondary = "http://mike.rs:8080";
            public const string JsonAPIPath = "http://127.0.0.1:4078";
            public const string CcminerAPIPath = "127.0.0.1:4068";

            public const bool SubmitStale = false;
            public const float GasToMine = 3.0f;
            public const ulong GasLimit = 1704624ul;
            public const float GasApiMax = 7.0f;
            public const int MaxScanRetry = 3;
            public const int PauseOnFailedScan = 3;
            public const int NetworkUpdateInterval = 15000;
            public const int HashrateUpdateInterval = 30000;

            public const bool LogFile = false;
        }
    }
}

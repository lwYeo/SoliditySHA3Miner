using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using Nethereum.Hex.HexTypes;

namespace SoliditySHA3Miner
{
    public class Donation
    {
        public const string Address = "0x9172ff7884CEFED19327aDaCe9C470eF1796105c";
        public const float Percent = 2.0F;
        public const float MinimumPercent = 1.5F;
        public static float UserPercent
        {
            get => (m_UserPercent < MinimumPercent) ? Percent : m_UserPercent;
            set => m_UserPercent = value;
        }
        private static float m_UserPercent = Percent;
    }

    public class Defaults
    {
        public const string InfuraAPI_mainnet = "https://mainnet.infura.io/ANueYSYQTstCr2mFJjPE";
        public const string InfuraAPI_ropsten = "https://ropsten.infura.io/ANueYSYQTstCr2mFJjPE";
        public const string Contract0xBTC_mainnet = "0xB6eD7644C69416d67B522e20bC294A9a9B405B31";
        public const string Contract0xBTC_ropsten = "0x9D2Cc383E677292ed87f63586086CfF62a009010";
        public const string AbiFile0xBTC = "0xbtc.abi";

        public const string PoolPrimary = "http://mike.rs:8080";//"http://0xbtc.wolfpool.io:8080";
        public const string PoolSecondary = "http://mike.rs:8080";
        public const string JsonAPIPath = "http://127.0.0.1:4078";
        public const string CcminerAPIPath = "127.0.0.1:4068";

        public const bool SubmitStale = true;
        public const int MaxScanRetry = 5;
        public const int PauseOnFailedScan = 3;
        public const int NetworkUpdateInterval = 15000;
        public const int HashrateUpdateInterval = 30000;

    }

    class Program
    {
        #region closing handler

        [DllImport("Kernel32")]
        private static extern bool SetConsoleCtrlHandler(EventHandler handler, bool add);

        enum CtrlType
        {
            CTRL_C_EVENT = 0,
            CTRL_BREAK_EVENT = 1,
            CTRL_CLOSE_EVENT = 2,
            CTRL_LOGOFF_EVENT = 5,
            CTRL_SHUTDOWN_EVENT = 6
        }
        private delegate bool EventHandler(CtrlType sig);

        static EventHandler m_handler;

        private static bool Handler(CtrlType sig)
        {
            foreach(var miner in m_allMiners)
            {
                try { if (miner != null) miner.Dispose(); }
                catch (Exception ex) { Print(ex.Message); }
            }

            if (m_waitCheckTimer != null) m_waitCheckTimer.Stop();
            if (m_manualResetEvent != null) m_manualResetEvent.Set();

            return true;
        }

        #endregion

        public static readonly DateTime LaunchTime = DateTime.Now;

        public static ulong WaitSeconds { get; private set; }

        public static string GetApplicationName() => typeof(Program).Assembly.GetName().Name;

        public static string GetApplicationVersion() => typeof(Program).Assembly.GetCustomAttribute<AssemblyFileVersionAttribute>().Version;

        public static string GetApplicationYear() => File.GetCreationTime(typeof(Program).Assembly.Location).Year.ToString();

        public static string GetCurrentTimestamp() => string.Format("{0:s}", DateTime.Now);

        public static void Print(string message) => Console.WriteLine(string.Format("[{0}] {1}", GetCurrentTimestamp(), message));

        private static System.Timers.Timer m_waitCheckTimer;
        private static ManualResetEvent m_manualResetEvent;
        private static Miner.CUDA m_cudaMiner;
        private static Miner.OpenCL m_openCLMiner;
        private static Miner.IMiner[] m_allMiners;
        private static API.Json m_apiJson;

        private static string GetHeader()
        {
            return "\n" +
                "*** " + GetApplicationName() + " " + GetApplicationVersion() + " by lwYeo@github (" + GetApplicationYear() + ") ***\n" +
                "*** Built with C#.NET 4.7.1, VC++ 2017 and nVidia CUDA SDK 9.2 64-bits\n" +
                "*** Include kernel from Mikers, Azlehria and LtTofu (Mag517)\n" +
                "\n" +
                "Donation addresses:\n" +
                "ETH (or any ERC 20/918 tokens)	: 0x9172ff7884cefed19327adace9c470ef1796105c\n" +
                "BTC                             : 3GS5J5hcG6Qcu9xHWGmJaV5ftWLmZuR255\n" +
                "LTC                             : LbFkAto1qYt8RdTFHL871H4djendcHyCyB\n";
        }

        private static void PrintHelp()
        {
            var help =
                "Usage: SoliditySHA3Miner [OPTIONS]\n" +
                "Options:\n" +
                "  help                    Display this help text and exit\n" +
                "  allowIntel              Allow to use Intel iGPU (default: true)\n" +
                "  intelIntensity          iGPU (Intel OpenCL) intensity (default: 21, decimals allowed)\n" +
                "  listAmdDevices          List of all AMD (OpenCL) devices in this system and exit (device ID: GPU name)\n" +
                "  amdDevice               Comma separated list of AMD (OpenCL) devices to use (default: all devices)\n" +
                "  amdIntensity            GPU (AMD OpenCL) intensity (default: 25, decimals allowed)\n" +
                "  listCudaDevices         List of all CUDA devices in this system and exit (device ID: GPU name)\n" +
                "  cudaDevice              Comma separated list of CUDA devices to use (default: all devices)\n" +
                "  cudaIntensity           GPU (CUDA) intensity (default: auto, decimals allowed)\n" +
                "  minerJsonAPI            'http://IP:port/' for the miner JSON-API (default: " + Defaults.JsonAPIPath + "), 0 disabled\n" +
                "  minerCcminerAPI         'IP:port' for the ccminer-style API (default: " + Defaults.CcminerAPIPath + "), 0 disabled\n" +
                "  overrideMaxDiff         (Pool only) Use maximum difficulty and skips query from web3\n" +
                "  customDifficulty        (Pool only) Set custom difficulity (check with your pool operator)\n" +
                "  maxScanRetry            Number of retries to scan for new work (default: " + Defaults.MaxScanRetry + ")\n" +
                "  pauseOnFailedScans      Pauses mining when connection fails, including secondary and retries (default: true)\n" +
                "  submitStale             Submit stale jobs, may create more rejected shares (default: " + Defaults.SubmitStale.ToString().ToLower() + ")\n" +
                "  abiFile                 Token abi in a file (default: '0xbtc.abi' in the same folder as this miner)\n" +
                "  web3api                 User-defined web3 provider URL (default: Infura mainnet provider)\n" +
                "  contract                Token contract address (default: 0xbtc contract address)\n" +
                "  hashrateUpdateInterval  Interval (miliseconds) for GPU hashrate logs (default: " + Defaults.HashrateUpdateInterval + ")\n" +
                "  networkUpdateInterval   Interval (miliseconds) to scan for new work (default: " + Defaults.NetworkUpdateInterval + ")\n" +
                "  address                 Miner's ethereum address (default: developer's address)\n" +
                //"  privateKey              (Solo only) Miner's private key\n" +
                "  pool                    (Pool only) URL of pool mining server (default: " + Defaults.PoolPrimary + ")\n" +
                "  secondaryPool           (Optional) URL of failover pool mining server\n" +
                "  donate                  Set donation in percentage (default: " + Donation.Percent + "%, minimum: " + Donation.MinimumPercent + "%)\n";
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

        private static void PrepareAmdDeviceList(ref Miner.Device[] amdDevices)
        {
            if (amdDevices == null || !amdDevices.Any())
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
        }

        private static void SetAmdDevices(string[] sAmdDevices, ref Miner.Device[] amdDevices)
        {
            PrepareAmdDeviceList(ref amdDevices);

            for (int i = 0; i < amdDevices.Length; i++)
            {
                if (string.IsNullOrEmpty(sAmdDevices[i])) continue;
                else
                {
                    var deviceID = int.Parse(sAmdDevices[i]);
                    amdDevices[i].Name = Miner.CUDA.GetDeviceName(deviceID, out string errorMessage);
                    if (!string.IsNullOrEmpty(errorMessage))
                    {
                        Print(errorMessage);
                        m_manualResetEvent.Set();
                        Environment.Exit(1);
                    }
                    amdDevices[i].DeviceID = int.Parse(sAmdDevices[i]);
                }
            }
        }

        private static void SetAmdIntensities(string[] sAmdIntensities, ref Miner.Device[] amdDevices)
        {
            if (amdDevices == null || !amdDevices.Any()) return;

            if (sAmdIntensities.Length > 1)
                for (int i = 0; i < sAmdIntensities.Length; i++)
                    amdDevices[i].Intensity = float.Parse(sAmdIntensities[i]);
            else
                for (int i = 0; i < sAmdIntensities.Length; i++)
                    amdDevices[i].Intensity = float.Parse(sAmdIntensities[0]);
        }

        private static void SetIntelIntensities(string[] sIntelIntensities, ref Miner.Device[] IntelDevices)
        {
            if (IntelDevices == null || !IntelDevices.Any()) return;

            if (sIntelIntensities.Length > 1)
                for (int i = 0; i < sIntelIntensities.Length; i++)
                    IntelDevices[i].Intensity = float.Parse(sIntelIntensities[i]);
            else
                for (int i = 0; i < sIntelIntensities.Length; i++)
                    IntelDevices[i].Intensity = float.Parse(sIntelIntensities[0]);
        }

        private static void PrintCudaDevices()
        {
            var cudaDevices = Miner.CUDA.GetDevices(out string errorMessage);
            Console.WriteLine(string.IsNullOrWhiteSpace(errorMessage) ? cudaDevices : errorMessage);
        }

        private static void PrepareCudaDeviceList(ref Miner.Device[] cudaDevices)
        {
            if (cudaDevices == null || !cudaDevices.Any())
            {
                var cudaDeviceCount = Miner.CUDA.GetDeviceCount(out string cudaCountErrorMessage);
                cudaDevices = (Miner.Device[])Array.CreateInstance(typeof(Miner.Device), cudaDeviceCount);
                for (int i = 0; i < cudaDevices.Length; i++)
                {
                    cudaDevices[i].Type = "CUDA";
                    cudaDevices[i].DeviceID = -1;
                }
            }
        }

        private static void SetCudaDevices(string[] sCudaDevices, ref Miner.Device[] cudaDevices)
        {
            PrepareCudaDeviceList(ref cudaDevices);

            for (int i = 0; i < sCudaDevices.Length; i++)
            {
                if (string.IsNullOrEmpty(sCudaDevices[i])) continue;
                else
                {
                    var deviceID = int.Parse(sCudaDevices[i]);
                    cudaDevices[i].Name = Miner.CUDA.GetDeviceName(deviceID, out string errorMessage);
                    if (!string.IsNullOrEmpty(errorMessage))
                    {
                        Print(errorMessage);
                        m_manualResetEvent.Set();
                        Environment.Exit(1);
                    }
                    cudaDevices[i].DeviceID = int.Parse(sCudaDevices[i]);
                }
            }
        }

        private static void SetCudaIntensities(string[] sCudaIntensities, ref Miner.Device[] cudaDevices)
        {
            if (cudaDevices == null || !cudaDevices.Any()) return;

            if (sCudaIntensities.Length > 1)
                for (int i = 0; i < sCudaIntensities.Length; i++)
                    cudaDevices[i].Intensity = float.Parse(sCudaIntensities[i]);
            else
                for (int i = 0; i < cudaDevices.Length; i++)
                    cudaDevices[i].Intensity = float.Parse(sCudaIntensities[0]);
        }

        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            Console.Title = string.Format("{0} {1} by lwYeo@github ({2})", GetApplicationName(), GetApplicationVersion(), GetApplicationYear());
            Console.WriteLine(GetHeader());

            m_handler += new EventHandler(Handler);
            SetConsoleCtrlHandler(m_handler, true);
            m_manualResetEvent = new ManualResetEvent(false);

            Miner.Device[] amdDevices = new Miner.Device[] { };
            Miner.Device[] intelDevices = new Miner.Device[] { }; ;
            Miner.Device[] cudaDevices = new Miner.Device[] { };
            var minerJsonAPI = string.Empty;
            var minerCcminerAPI = string.Empty;
            var overrideMaxDiff = new HexBigInteger(BigInteger.Zero);
            var customDifficulty = 0u;
            var maxScanRetry = Defaults.MaxScanRetry;
            var pauseOnFailedScans = Defaults.PauseOnFailedScan;
            var submitStale = Defaults.SubmitStale;
            var abiFile = Defaults.AbiFile0xBTC;
            var web3api = Defaults.InfuraAPI_mainnet;
            var contractAddress = Defaults.Contract0xBTC_mainnet;
            var networkUpdateInterval = Defaults.NetworkUpdateInterval;
            var hashrateUpdateInterval = Defaults.HashrateUpdateInterval;
            var minerAddress = string.Empty;
            var primaryPool = string.Empty;
            var secondaryPool = string.Empty;
            var privateKey = string.Empty;
            var allowIntel = true;

            foreach (var arg in args)
            {
                try
                {
                    switch (arg.Split('=')[0])
                    {
                        case "help":
                            PrintHelp();
                            m_manualResetEvent.Set();
                            Environment.Exit(0);
                            break;
                        case "allowIntel":
                            allowIntel = bool.Parse(arg.Split('=')[1]);
                            break;
                        case "listAmdDevices":
                            PrintAmdDevices();
                            m_manualResetEvent.Set();
                            break;
                        case "amdDevice":
                            SetAmdDevices(arg.Split('=')[1].Split(','), ref amdDevices);
                            break;
                        case "listCudaDevices":
                            PrintCudaDevices();
                            m_manualResetEvent.Set();
                            Environment.Exit(0);
                            break;
                        case "cudaDevice":
                            SetCudaDevices(arg.Split('=')[1].Split(','), ref cudaDevices);
                            break;
                        case "minerJsonAPI":
                            minerJsonAPI = arg.Split('=')[1];
                            break;
                        case "minerCcminerAPI":
                            minerCcminerAPI = arg.Split('=')[1];
                            break;
                        case "overrideMaxDiff":
                            overrideMaxDiff = new HexBigInteger(BigInteger.Parse((arg.Split('=')[1])));
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
                        case "address":
                            minerAddress = arg.Split('=')[1];
                            break;
                        case "privateKey":
                            privateKey = arg.Split('=')[1];
                            break;
                        case "pool":
                            primaryPool = arg.Split('=')[1];
                            break;
                        case "secondaryPool":
                            secondaryPool = arg.Split('=')[1];
                            break;
                        case "donate":
                            Donation.UserPercent = float.Parse(arg.Split('=')[1]);
                            break;
                    }
                }
                catch (Exception)
                {
                    Print("[ERROR] Failed parsing argument: " + arg);
                    m_manualResetEvent.Set();
                    Environment.Exit(1);
                }
            }

            if (string.IsNullOrWhiteSpace(minerAddress))
            {
                Print("[INFO] Miner address not specified, donating 100% to dev.");
                minerAddress = Donation.Address;
            }

            if (!string.IsNullOrWhiteSpace(privateKey))
            {
                Print("[INFO] Solo mining mode.");

                //TODO: Solo mining
            }
            else if (string.IsNullOrWhiteSpace(primaryPool))
            {
                Print("[INFO] Primary pool not specified, using " + Defaults.PoolPrimary);
                primaryPool = Defaults.PoolPrimary;
            }

            if ((cudaDevices == null || !cudaDevices.Any()) && args.All(a => !a.StartsWith("cudaDevice")))
            {
                Print("[INFO] CUDA device not specified, default assign all CUDA devices.");
                var cudaDeviceCount = Miner.CUDA.GetDeviceCount(out string cudaCountErrorMessage);
                if (!string.IsNullOrWhiteSpace(cudaCountErrorMessage)) Print(cudaCountErrorMessage);

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
                else Print("[WARN] CUDA device not found.");
            }

            Miner.OpenCL.PreInitialize(allowIntel, out var openCLInitErrorMessage);

            if (!string.IsNullOrWhiteSpace(openCLInitErrorMessage)) Print("[ERROR] " + openCLInitErrorMessage);
            else
            {
                if (allowIntel)
                {
                    Print("[INFO] Assign all Intel(R) OpenCL devices.");
                    var deviceCount = Miner.OpenCL.GetDeviceCount("Intel(R) OpenCL", out var openCLerrorMessage);
                    if (!string.IsNullOrWhiteSpace(openCLerrorMessage)) Print("[WARN] " + openCLerrorMessage);
                    else
                    {
                        var tempIntelList = new List<Miner.Device>();
                        for (int i = 0; i < deviceCount; i++)
                        {
                            var tempName = Miner.OpenCL.GetDeviceName("Intel(R) OpenCL", i, out var openCLdeviceErrorMessage);
                            if (!string.IsNullOrWhiteSpace(openCLdeviceErrorMessage))
                            {
                                Print("[WARN] " + openCLdeviceErrorMessage);
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

                if ((amdDevices == null || !amdDevices.Any()) && args.All(a => !a.StartsWith("openclDevices")))
                {
                    Print("[INFO] OpenCL device not specified, default assign all AMD APP devices.");

                    var deviceCount = Miner.OpenCL.GetDeviceCount("AMD Accelerated Parallel Processing", out var openCLerrorMessage);
                    if (!string.IsNullOrWhiteSpace(openCLerrorMessage)) Print("[WARN] " + openCLerrorMessage);
                    else
                    {
                        var tempAmdList = new List<Miner.Device>();
                        for (int i = 0; i < deviceCount; i++)
                        {
                            var tempName = Miner.OpenCL.GetDeviceName("AMD Accelerated Parallel Processing", i, out var openCLdeviceErrorMessage);
                            if (!string.IsNullOrWhiteSpace(openCLdeviceErrorMessage))
                            {
                                Print("[WARN] " + openCLdeviceErrorMessage);
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

            foreach (var arg in args)
            {
                try
                {
                    switch (arg.Split('=')[0])
                    {
                        case "intelIntensity":
                            SetIntelIntensities(arg.Split('=')[1].Split(','), ref intelDevices);
                            break;
                        case "amdIntensity":
                            SetAmdIntensities(arg.Split('=')[1].Split(','), ref amdDevices);
                            break;
                        case "cudaIntensity":
                            SetCudaIntensities(arg.Split('=')[1].Split(','), ref cudaDevices);
                            break;
                    }
                }
                catch (Exception)
                {
                    Print("[ERROR] Failed parsing argument: " + arg);
                    m_manualResetEvent.Set();
                    Environment.Exit(1);
                }
            }

            try
            {
                var web3Interface = new NetworkInterface.Web3Interface(web3api, contractAddress, minerAddress, privateKey, abiFile);

                var secondaryPoolInterface = string.IsNullOrWhiteSpace(secondaryPool) ? null : new NetworkInterface.PoolInterface(minerAddress, secondaryPool, maxScanRetry);
                var primaryPoolInterface = new NetworkInterface.PoolInterface(minerAddress, primaryPool, maxScanRetry, secondaryPoolInterface);

                if (overrideMaxDiff.Value > 0u)
                    Print("[INFO] Override maximum difficulty: " + overrideMaxDiff.Value);

                m_cudaMiner = new Miner.CUDA(primaryPoolInterface, cudaDevices,
                                             (overrideMaxDiff.Value > 0u) ? overrideMaxDiff : web3Interface.GetMaxDifficulity(),
                                             customDifficulty, submitStale, pauseOnFailedScans);

                if (m_cudaMiner.HasAssignedDevices) m_cudaMiner.StartMining(networkUpdateInterval < 1000 ? Defaults.NetworkUpdateInterval : networkUpdateInterval,
                                                                            hashrateUpdateInterval < 1000 ? Defaults.HashrateUpdateInterval : hashrateUpdateInterval);

                m_openCLMiner = new Miner.OpenCL(primaryPoolInterface, intelDevices.Union(amdDevices).ToArray(),
                                                 (overrideMaxDiff.Value > 0u) ? overrideMaxDiff : web3Interface.GetMaxDifficulity(),
                                                 customDifficulty, submitStale, pauseOnFailedScans);

                if (m_openCLMiner.HasAssignedDevices) m_openCLMiner.StartMining(networkUpdateInterval < 1000 ? Defaults.NetworkUpdateInterval : networkUpdateInterval,
                                                                                hashrateUpdateInterval < 1000 ? Defaults.HashrateUpdateInterval : hashrateUpdateInterval);

                //TODO: CPU

                m_allMiners = new Miner.IMiner[] { m_openCLMiner, m_cudaMiner };

                if (m_allMiners.All(m => !m.HasAssignedDevices))
                {
                    Print("[ERROR] No miner assigned.");
                    m_manualResetEvent.Set();
                    Environment.Exit(1);
                }

                m_apiJson = new API.Json(m_allMiners);
                if (m_apiJson.IsSupported) m_apiJson.Start(minerJsonAPI);

                API.Ccminer.StartListening(minerCcminerAPI, m_allMiners);

                m_waitCheckTimer = new System.Timers.Timer(1000);
                m_waitCheckTimer.Elapsed += delegate { if (m_allMiners.All(m => m != null && !m.IsMining)) WaitSeconds++; };
                m_waitCheckTimer.Start();
                WaitSeconds = (ulong)(LaunchTime - DateTime.Now).TotalSeconds;
            }
            catch (Exception ex)
            {
                Print("[ERROR] " + ex.Message);
                m_manualResetEvent.Set();
                Environment.Exit(1);
            }

            m_manualResetEvent.WaitOne();
            Print("[INFO] Exiting application...");

            API.Ccminer.StopListening();
            m_waitCheckTimer.Stop();
        }
    }
}

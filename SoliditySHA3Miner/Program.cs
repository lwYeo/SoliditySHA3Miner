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

using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace SoliditySHA3Miner
{
    public class DevFee
    {
        public const string Address = "0x9172ff7884CEFED19327aDaCe9C470eF1796105c";
        public const float Percent = 2.0f;
        public const float MinimumPercent = 1.5f;

        public static float UserPercent
        {
            get => (m_UserPercent < MinimumPercent) ? MinimumPercent : m_UserPercent;
            set => m_UserPercent = value;
        }

        private static float m_UserPercent = Percent;
    }

    internal class Program
    {
        #region closing handler

        [DllImport("Kernel32")]
        private static extern bool SetConsoleCtrlHandler(EventHandler handler, bool add);

        private enum CtrlType
        {
            CTRL_C_EVENT = 0,
            CTRL_BREAK_EVENT = 1,
            CTRL_CLOSE_EVENT = 2,
            CTRL_LOGOFF_EVENT = 5,
            CTRL_SHUTDOWN_EVENT = 6
        }

        private delegate bool EventHandler(CtrlType sig);

        private static EventHandler m_handler;

        private static bool Handler(CtrlType sig)
        {
            if (m_handler == null)
                m_handler += new EventHandler(Handler);

            lock (m_handler)
            {
                try
                {
                    if (m_allMiners != null)
                    {
                        Task.WaitAll(m_allMiners.Select(m => Task.Factory.StartNew(() => m.StopMining())).ToArray());
                        Task.WaitAll(m_allMiners.Select(m => Task.Factory.StartNew(() => m.Dispose())).ToArray());
                    }

                    if (m_waitCheckTimer != null) m_waitCheckTimer.Stop();
                }
                catch { }

                if (m_manualResetEvent != null) m_manualResetEvent.Set();

                return true;
            }
        }

        #endregion closing handler

        public static readonly DateTime LaunchTime = DateTime.Now;

        public static Config Config { get; private set; }

        public static ulong WaitSeconds { get; private set; }

        public static string LogFileFormat => $"{DateTime.Today:yyyy-MM-dd}.log";

        public static string AppDirPath => Path.GetDirectoryName(typeof(Program).Assembly.Location);

        public static string GetApplicationName() => typeof(Program).Assembly.GetName().Name;

        public static string GetCompanyName() => typeof(Program).Assembly.GetCustomAttribute<AssemblyCompanyAttribute>().Company;

        public static string GetApplicationVersion() => typeof(Program).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>().InformationalVersion;

        public static string GetApplicationYear() => File.GetCreationTime(typeof(Program).Assembly.Location).Year.ToString();

        public static string GetAppConfigPath() => Path.Combine(AppDirPath, GetApplicationName() + ".conf");

        public static bool AllowIntel { get; set; }

        public static bool AllowAMD { get; set; }

        public static bool AllowCUDA { get; set; }

        public static bool AllowCPU { get; set; }

        public static string GetCurrentTimestamp() => string.Format("{0:s}", DateTime.Now);

        public static void Print(string message, bool excludePrefix = false)
        {
            new TaskFactory().StartNew(() =>
            {
                message = message.Replace("Accelerated Parallel Processing", "APP").Replace("\n", Environment.NewLine);
                if (!excludePrefix) message = string.Format("[{0}] {1}", GetCurrentTimestamp(), message);

                if (Config.isLogFile)
                {
                    var logFilePath = Path.Combine(AppDirPath, "Log", LogFileFormat);

                    lock (m_manualResetEvent)
                    {
                        try
                        {
                            if (!Directory.Exists(Path.GetDirectoryName(logFilePath))) Directory.CreateDirectory(Path.GetDirectoryName(logFilePath));

                            using (var logStream = File.AppendText(logFilePath))
                            {
                                logStream.WriteLine(message);
                                logStream.Close();
                            }
                        }
                        catch (Exception)
                        {
                            Console.WriteLine(string.Format("[ERROR] Failed writing to log file '{0}'", logFilePath));
                        }
                    }
                }

                Console.WriteLine(message);

                if (message.Contains("Kernel launch failed"))
                {
                    Task.Delay(5000);
                    Environment.Exit(22);
                }
            });
        }

        private static ManualResetEvent m_manualResetEvent = new ManualResetEvent(false);
        private static System.Timers.Timer m_waitCheckTimer;
        private static Miner.CPU m_cpuMiner;
        private static Miner.CUDA m_cudaMiner;
        private static Miner.OpenCL m_openCLMiner;
        private static Miner.IMiner[] m_allMiners;
        private static API.Json m_apiJson;

        private static string GetHeader()
        {
            return "\n" +
                "*** " + GetApplicationName() + " " + GetApplicationVersion() + " by " + GetCompanyName() + " (" + GetApplicationYear() + ") ***\n" +
                "*** Built with .NET Core 2.1.5 SDK, VC++ 2017, gcc 4.8.5, nVidia CUDA SDK 9.2 64-bits, and AMD APP SDK v3.0.130.135 (OpenCL)\n" +
                "\n" +
                "Donation addresses:\n" +
                "ETH (or any ERC 20/918 tokens)	: 0x9172ff7884CEFED19327aDaCe9C470eF1796105c\n" +
                "BTC                             : 3GS5J5hcG6Qcu9xHWGmJaV5ftWLmZuR255\n" +
                "LTC                             : LbFkAto1qYt8RdTFHL871H4djendcHyCyB\n";
        }

        private static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            Console.Title = string.Format("{0} {1} by {2} ({3})", GetApplicationName(), GetApplicationVersion(), GetCompanyName(), GetApplicationYear());

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                m_handler += new EventHandler(Handler);
                SetConsoleCtrlHandler(m_handler, true);
            }
            else
            {
                AppDomain.CurrentDomain.ProcessExit += (sender, e) =>
                {
                    Handler(CtrlType.CTRL_CLOSE_EVENT);
                };
                Console.CancelKeyPress += (s, ev) =>
                {
                    Handler(CtrlType.CTRL_C_EVENT);
                };
            }

            if (File.Exists(GetAppConfigPath()))
            {
                Config = Utils.Json.DeserializeFromFile<Config>(GetAppConfigPath());
                if (Config == null)
                {
                    Console.WriteLine(string.Format("[ERROR] Failed to read config file at {0}", GetAppConfigPath()));
                    if (args.Any())
                        Config = new Config();
                    else
                        Environment.Exit(1);
                }
            }
            else
            {
                Config = new Config();
                if (Utils.Json.SerializeToFile(Config, GetAppConfigPath()))
                {
                    Console.WriteLine(string.Format("[INFO] Config file created at {0}", GetAppConfigPath()));
                    if (!args.Any())
                    {
                        Console.WriteLine("[INFO] Update the config file, especially miner details.");
                        Console.WriteLine("[INFO] Exiting application...");
                        Environment.Exit(0);
                    }
                }
                else
                {
                    Console.WriteLine(string.Format("[ERROR] Failed to write config file at {0}", GetAppConfigPath()));
                    if (!args.Any())
                        Environment.Exit(1);
                }
            }

            foreach (var arg in args)
            {
                try
                {
                    switch (arg.Split('=')[0])
                    {
                        case "logFile":
                            Config.isLogFile = bool.Parse(arg.Split('=')[1]);
                            break;
                    }
                }
                catch (Exception)
                {
                    Console.WriteLine("[ERROR] Failed parsing argument: " + arg);
                    Environment.Exit(1);
                }
            }

            Print(GetHeader(), excludePrefix: true);

            Config.allowCPU = true;
            Config.allowCUDA = false;

            if (!Config.ParseArgumentsToConfig(args)) Environment.Exit(1);

            try
            {
                Config.networkUpdateInterval = Config.networkUpdateInterval < 1000 ? Config.Defaults.NetworkUpdateInterval : Config.networkUpdateInterval;
                Config.hashrateUpdateInterval = Config.hashrateUpdateInterval < 1000 ? Config.Defaults.HashrateUpdateInterval : Config.hashrateUpdateInterval;

                Miner.Work.SolutionTemplate = Miner.Helper.CPU.GetSolutionTemplate(kingAddress: Config.kingAddress);

                var web3Interface = new NetworkInterface.Web3Interface(Config.web3api, Config.contractAddress, Config.minerAddress, Config.privateKey, Config.gasToMine,
                                                                       Config.abiFile, Config.networkUpdateInterval, Config.hashrateUpdateInterval,
                                                                       Config.gasLimit, Config.gasApiURL, Config.gasApiPath, Config.gasApiMultiplier, Config.gasApiOffset, Config.gasApiMax);

                web3Interface.OverrideMaxTarget(Config.overrideMaxTarget);

                if (Config.customDifficulty > 0)
                    Print("[INFO] Custom difficulity: " + Config.customDifficulty.ToString());

                NetworkInterface.INetworkInterface mainNetworkInterface = null;
                var isSoloMining = !(string.IsNullOrWhiteSpace(Config.privateKey));

                if (isSoloMining) { mainNetworkInterface = web3Interface; }
                else
                {
                    var secondaryPoolInterface = string.IsNullOrWhiteSpace(Config.secondaryPool)
                                               ? null
                                               : new NetworkInterface.PoolInterface(Config.minerAddress, Config.secondaryPool, Config.maxScanRetry,
                                                                                    -1, -1, Config.customDifficulty, true, web3Interface.GetMaxTarget());

                    var primaryPoolInterface = new NetworkInterface.PoolInterface(Config.minerAddress, Config.primaryPool, Config.maxScanRetry,
                                                                                  Config.networkUpdateInterval, Config.hashrateUpdateInterval,
                                                                                  Config.customDifficulty, false, web3Interface.GetMaxTarget(), secondaryPoolInterface);
                    mainNetworkInterface = primaryPoolInterface;
                }

                if (AllowCUDA && Config.cudaDevices.Any(d => d.AllowDevice))
                    m_cudaMiner = new Miner.CUDA(mainNetworkInterface, Config.cudaDevices, Config.submitStale, Config.pauseOnFailedScans);

                if ((AllowAMD || AllowIntel) && Config.intelDevices.Union(Config.amdDevices).Any(d => d.AllowDevice))
                    m_openCLMiner = new Miner.OpenCL(mainNetworkInterface, Config.intelDevices, Config.amdDevices, Config.submitStale, Config.pauseOnFailedScans);

                if (AllowCPU && Config.cpuDevice.AllowDevice)
                    m_cpuMiner = new Miner.CPU(mainNetworkInterface, Config.cpuDevice, Config.submitStale, Config.pauseOnFailedScans);

                m_allMiners = new Miner.IMiner[] { m_openCLMiner, m_cudaMiner, m_cpuMiner }.Where(m => m != null).ToArray();

                if (!m_allMiners.Any() || m_allMiners.All(m => !m.HasAssignedDevices))
                {
                    Console.WriteLine("[ERROR] No miner assigned.");
                    Environment.Exit(1);
                }

                if (!Utils.Json.SerializeToFile(Config, GetAppConfigPath()))
                    Print(string.Format("[ERROR] Failed to write config file at {0}", GetAppConfigPath()));

                m_apiJson = new API.Json(m_allMiners);
                if (m_apiJson.IsSupported) m_apiJson.Start(Config.minerJsonAPI);

                API.Ccminer.StartListening(Config.minerCcminerAPI, m_allMiners);

                if (m_cudaMiner != null && m_cudaMiner.HasAssignedDevices)
                    m_cudaMiner.StartMining(Config.networkUpdateInterval, Config.hashrateUpdateInterval);

                if (m_openCLMiner != null && m_openCLMiner.HasAssignedDevices)
                    m_openCLMiner.StartMining(Config.networkUpdateInterval, Config.hashrateUpdateInterval);

                if (m_cpuMiner != null && m_cpuMiner.HasAssignedDevices)
                    m_cpuMiner.StartMining(Config.networkUpdateInterval, Config.hashrateUpdateInterval);

                m_waitCheckTimer = new System.Timers.Timer(1000);
                m_waitCheckTimer.Elapsed +=
                    delegate
                    {
                        if (m_allMiners.All(m => m != null && (!m.IsMining || m.IsPause))) WaitSeconds++;
                    };
                m_waitCheckTimer.Start();
                WaitSeconds = (ulong)(LaunchTime - DateTime.Now).TotalSeconds;
            }
            catch (Exception ex)
            {
                Console.WriteLine("[ERROR] " + ex.ToString());
                if (ex.InnerException != null)
                    Console.WriteLine(ex.InnerException.ToString());

                Environment.Exit(1);
            }

            m_manualResetEvent.WaitOne();

            m_waitCheckTimer.Stop();
            m_apiJson.Stop();
            m_apiJson.Dispose();
            API.Ccminer.StopListening();

            Console.WriteLine("[INFO] Exiting application...");
        }
    }
}
using System;
using System.Linq;
using System.Text;
using System.Timers;
using System.Threading.Tasks;
using OpenCLSolver;
using Nethereum.Hex.HexTypes;

namespace SoliditySHA3Miner.Miner
{
    public class OpenCL : IMiner
    {
        #region static

        public static void PreInitialize(bool allowIntel, out string errorMessage)
        {
            errorMessage = string.Empty;
            Solver.preInitialize(allowIntel, ref errorMessage);
        }

        public static string[] GetPlatformNames()
        {
            return Solver.getPlatformNames().Split('\n');
        }

        public static int GetDeviceCount(string platformName, out string errorMessage)
        {
            errorMessage = string.Empty;
            return Solver.getDeviceCount(platformName, ref errorMessage);
        }

        public static string GetDeviceName(string platformName, int deviceEnum, out string errorMessage)
        {
            errorMessage = string.Empty;
            return Solver.getDeviceName(platformName, deviceEnum, ref errorMessage);
        }

        public static string GetDevices(string platformName, out string errorMessage)
        {
            errorMessage = string.Empty;
            var devicesString = new StringBuilder();
            var deviceCount = Solver.getDeviceCount(platformName, ref errorMessage);

            if (!string.IsNullOrEmpty(errorMessage)) return string.Empty;

            for (int i = 0; i < deviceCount; i++)
            {
                var deviceName = Solver.getDeviceName(platformName, i, ref errorMessage);
                if (!string.IsNullOrEmpty(errorMessage)) return string.Empty;

                devicesString.AppendLine(string.Format("{0}: {1}", i, deviceName));
            }
            return devicesString.ToString();
        }

        #endregion

        private Timer m_hashPrintTimer;
        private Timer m_updateMinerTimer;
        private int m_pauseOnFailedScan;
        private int m_failedScanCount;

        public Solver Solver { get; }

        #region IMiner

        public NetworkInterface.INetworkInterface NetworkInterface { get; }

        public bool HasAssignedDevices => (bool)Solver?.isAssigned();

        public Device[] Devices { get; }

        public bool IsAnyInitialised => (bool)Solver?.isAnyInitialised();

        public bool IsMining => (bool)Solver?.isMining();

        public bool IsPaused => (bool)Solver?.isPaused();

        private NetworkInterface.MiningParameters lastMiningParameters;

        public void Dispose()
        {
            try
            {
                m_updateMinerTimer.Dispose();
                Solver.Dispose();
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        public long GetDifficulty()
        {
            try { return (long)lastMiningParameters.MiningDifficulty.Value; }
            catch (OverflowException) { return long.MaxValue; }
        }

        public ulong GetHashrateByDevice(string platformName, int deviceID)
        {
            return Solver.getHashRateByDevice(platformName, deviceID);
        }

        public ulong GetTotalHashrate()
        {
            return Solver.getTotalHashRate();
        }

        public void StartMining(int networkUpdateInterval, int hashratePrintInterval)
        {
            try
            {
                if (NetworkInterface.IsPool) Program.Print("[INFO] Waiting for pool to respond...");
                else Program.Print("[INFO] Waiting for network to respond...");
                UpdateMiner(Solver).Wait();

                m_updateMinerTimer = new Timer(networkUpdateInterval);
                m_updateMinerTimer.Elapsed += m_updateMinerTimer_Elapsed;
                m_updateMinerTimer.Start();

                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += m_hashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();

                Solver.startFinding();
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
                StopMining();
            }
        }

        public void StopMining()
        {
            try
            {
                m_updateMinerTimer.Stop();
                Solver.stopFinding();
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        #endregion

        public OpenCL(NetworkInterface.INetworkInterface networkInterface, Device[] devices, HexBigInteger maxDifficulty, uint customDifficulty, bool isSubmitStale, int pauseOnFailedScans)
        {
            try
            {
                Devices = devices;
                NetworkInterface = networkInterface;
                m_pauseOnFailedScan = pauseOnFailedScans;
                m_failedScanCount = 0;

                Solver = new Solver(maxDifficulty.HexValue)
                {
                    OnMessageHandler = m_openCLSolver_OnMessage,
                    OnSolutionHandler = m_openCLSolver_OnSolution
                };
                if (customDifficulty > 0u) Solver.setCustomDifficulty(customDifficulty);
                Solver.setSubmitStale(isSubmitStale);

                if (devices.All(d => d.DeviceID == -1))
                {
                    Program.Print("[INFO] OpenCL device not set.");
                    return;
                }

                for (int i = 0; i < Devices.Length; i++)
                    if (Devices[i].DeviceID > -1)
                        Solver.assignDevice(Devices[i].Platform, Devices[i].DeviceID, Devices[i].Intensity);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        private async Task UpdateMiner(Solver solver)
        {
            await Task.Factory.StartNew(() =>
            {
                lock (NetworkInterface)
                {
                    try
                    {
                        var miningParameters = NetworkInterface.GetMiningParameters();
                        if (miningParameters == null) return;

                        lastMiningParameters = miningParameters;

                        solver.updatePrefix(lastMiningParameters.ChallengeNumberByte32String + lastMiningParameters.EthAddress.Replace("0x", string.Empty));
                        solver.updateTarget(lastMiningParameters.MiningTargetByte32String);
                        solver.updateDifficulty(lastMiningParameters.MiningDifficulty.HexValue);

                        if (!NetworkInterface.IsPool &&
                        ((NetworkInterface.Web3Interface)NetworkInterface).IsChallengedSubmitted(miningParameters.ChallengeNumberByte32String))
                        {
                            if (!solver.isPaused()) solver.pauseFinding(true);
                        }
                        else if (solver.isPaused()) solver.pauseFinding(false);

                        if (m_failedScanCount > m_pauseOnFailedScan && solver.isPaused())
                        {
                            m_failedScanCount = 0;
                            solver.pauseFinding(false);
                        }
                    }
                    catch (Exception ex)
                    {
                        try
                        {
                            Program.Print(string.Format("[ERROR] {0}", ex.Message));

                            m_failedScanCount += 1;
                            if (m_failedScanCount > m_pauseOnFailedScan && solver.isMining()) solver.pauseFinding(true);
                        }
                        catch (Exception) { }
                    }
                }
            });
        }

        private void m_hashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var hashString = new StringBuilder();
            hashString.Append("OpenCL [INFO] Hashrates:");

            foreach (var device in Devices)
                if (device.DeviceID > -1)
                    hashString.AppendFormat(" {0} MH/s", Solver.getHashRateByDevice(device.Platform, device.DeviceID) / 1000000.0f);

            Program.Print(hashString.ToString());
            Program.Print(string.Format("OpenCL [INFO] Total Hashrate: {0} MH/s", Solver.getTotalHashRate() / 1000000.0f));
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, false);
        }

        private void m_updateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var updateTask = UpdateMiner(Solver);
        }

        private void m_openCLSolver_OnSolution(string digest, string address, string challenge, string difficulty, string target, string solution, bool isCustomDifficulty)
        {
            NetworkInterface.SubmitSolution(digest, address, challenge, difficulty, target, solution, isCustomDifficulty);
            UpdateMiner(Solver).Wait();
        }

        private void m_openCLSolver_OnMessage(string platformName, int deviceEnum, string type, string message)
        {
            var sFormat = new StringBuilder();
            if (!string.IsNullOrWhiteSpace(platformName)) sFormat.Append(platformName + " ");
            if (deviceEnum > -1) sFormat.Append("ID: {0} ");

            switch (type.ToUpperInvariant())
            {
                case "INFO":
                    sFormat.Append(deviceEnum > -1 ? "[INFO] {1}" : "[INFO] {0}");
                    break;
                case "WARN":
                    sFormat.Append(deviceEnum > -1 ? "[WARN] {1}" : "[WARN] {0}");
                    break;
                case "ERROR":
                    sFormat.Append(deviceEnum > -1 ? "[ERROR] {1}" : "[ERROR] {0}");
                    break;
                case "DEBUG":
                default:
#if DEBUG
                    sFormat.Append(deviceEnum > -1 ? "[DEBUG] {1}" : "[DEBUG] {0}");
                    break;
#else
                    return;
#endif
            }
            Program.Print(deviceEnum > -1
                ? string.Format(sFormat.ToString(), deviceEnum, message)
                : string.Format(sFormat.ToString(), message));
        }
    }
}

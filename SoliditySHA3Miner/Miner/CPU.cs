using System;
using System.Linq;
using System.Text;
using System.Timers;
using System.Threading.Tasks;
using CPUSolver;
using Nethereum.Hex.HexTypes;

namespace SoliditySHA3Miner.Miner
{
    public class CPU : IMiner
    {
        #region Static

        public static uint GetLogicalProcessorCount()
        {
            return Solver.getLogicalProcessorsCount();
        }

        public static string GetSolutionTemplate(string solutionTemplate = "")
        {
            return Solver.getSolutionTemplate(solutionTemplate);
        }

        #endregion

        private Timer m_hashPrintTimer;
        private Timer m_updateMinerTimer;
        private int m_pauseOnFailedScan;
        private int m_failedScanCount;

        public Solver Solver { get; }

        private NetworkInterface.MiningParameters lastMiningParameters;

        #region IMiner

        public NetworkInterface.INetworkInterface NetworkInterface { get; }

        public bool HasAssignedDevices => Solver != null && Devices.Any(d => d.DeviceID > -1);

        public bool HasMonitoringAPI => false;

        public Device[] Devices { get; }

        public bool IsAnyInitialised => true; // CPU is always initialised

        public bool IsMining
        {
            get
            {
                try { return Solver == null ? false : Solver.isMining(); }
                catch (Exception) { return false; }
            }
        }

        public bool IsPaused
        {
            get
            {
                try { return Solver == null ? false : Solver.isPaused(); }
                catch (Exception) { return false; }
            }
        }

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
            catch (Exception) { return 0; }
        }

        public ulong GetHashrateByDevice(string platformName, int deviceID)
        {
            try { return (ulong)Solver?.getHashRateByThreadID((uint)deviceID); }
            catch (Exception) { return 0u; }
        }

        public ulong GetTotalHashrate()
        {
            try { return Solver.getTotalHashRate(); }
            catch (Exception) { return 0u; }
        }

        public void StartMining(int networkUpdateInterval, int hashratePrintInterval)
        {
            try
            {
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

        public CPU(NetworkInterface.INetworkInterface networkInterface, Device[] devices, string solutionTemplate, string kingAddress, HexBigInteger maxDifficulty, uint customDifficulty, bool isSubmitStale, int pauseOnFailedScans)
        {
            try
            {
                Devices = devices;
                NetworkInterface = networkInterface;
                m_pauseOnFailedScan = pauseOnFailedScans;
                m_failedScanCount = 0;

                var devicesStr = string.Empty;
                foreach (var device in Devices)
                {
                    if (device.DeviceID < 0) continue;

                    if (!string.IsNullOrEmpty(devicesStr)) devicesStr += ',';
                    devicesStr += device.DeviceID.ToString("X64");
                }

                Solver = new Solver(maxDifficulty.HexValue, devicesStr, solutionTemplate, kingAddress)
                {
                    OnMessageHandler = m_cpuSolver_OnMessage,
                    OnSolutionHandler = m_cpuSolver_OnSolution
                };

                if (customDifficulty > 0u) Solver.setCustomDifficulty(customDifficulty);
                Solver.setSubmitStale(isSubmitStale);

                if (string.IsNullOrWhiteSpace(devicesStr))
                {
                    Program.Print("[WARN] No CPU assigned.");
                    return;
                }
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
                    hashString.AppendFormat(" {0} MH/s", Solver.getHashRateByThreadID((uint)device.DeviceID) / 1000000.0f);

            Program.Print(hashString.ToString());
            Program.Print(string.Format("OpenCL [INFO] Total Hashrate: {0} MH/s", Solver.getTotalHashRate() / 1000000.0f));
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, false);
        }

        private void m_updateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var updateTask = UpdateMiner(Solver);
        }

        private void m_cpuSolver_OnSolution(string digest, string address, string challenge, string difficulty, string target, string solution, bool isCustomDifficulty)
        {
            NetworkInterface.SubmitSolution(digest, address, challenge, difficulty, target, solution, isCustomDifficulty);
            UpdateMiner(Solver).Wait();
        }

        private void m_cpuSolver_OnMessage(int threadID, string type, string message)
        {

            var sFormat = new StringBuilder();
            if (threadID > -1) sFormat.Append("CPU Thread: {0} ");

            switch (type.ToUpperInvariant())
            {
                case "INFO":
                    sFormat.Append(threadID > -1 ? "[INFO] {1}" : "[INFO] {0}");
                    break;
                case "WARN":
                    sFormat.Append(threadID > -1 ? "[WARN] {1}" : "[WARN] {0}");
                    break;
                case "ERROR":
                    sFormat.Append(threadID > -1 ? "[ERROR] {1}" : "[ERROR] {0}");
                    break;
                case "DEBUG":
                default:
#if DEBUG
                    sFormat.Append(threadID > -1 ? "[DEBUG] {1}" : "[DEBUG] {0}");
                    break;
#else
                    return;
#endif
            }
            Program.Print(threadID > -1
                ? string.Format(sFormat.ToString(), threadID, message)
                : string.Format(sFormat.ToString(), message));
        }
    }
}

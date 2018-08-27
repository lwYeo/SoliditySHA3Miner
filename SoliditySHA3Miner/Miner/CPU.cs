using System;
using System.Linq;
using System.Text;
using System.Timers;
using CPUSolver;

namespace SoliditySHA3Miner.Miner
{
    public class CPU : IMiner
    {
        #region Static

        public static uint GetLogicalProcessorCount()
        {
            return Solver.getLogicalProcessorsCount();
        }

        public static string GetNewSolutionTemplate(string solutionTemplate = "")
        {
            return Solver.getNewSolutionTemplate(solutionTemplate);
        }

        #endregion

        private Timer m_hashPrintTimer;
        private int m_pauseOnFailedScan;
        private int m_failedScanCount;

        public Solver Solver { get; }
        
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
                Solver.Dispose();
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
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
                NetworkInterface.UpdateMiningParameters();

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
                m_hashPrintTimer.Stop();
                Solver.stopFinding();
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        #endregion

        public CPU(NetworkInterface.INetworkInterface networkInterface, Device[] devices, bool isSubmitStale, int pauseOnFailedScans)
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

                unsafe
                {
                    Solver = new Solver(devicesStr)
                    {
                        OnGetKingAddressHandler = Work.GetKingAddress,
                        OnGetSolutionTemplateHandler = Work.GetSolutionTemplate,
                        OnGetWorkPositionHandler = Work.GetPosition,
                        OnResetWorkPositionHandler = Work.ResetPosition,
                        OnIncrementWorkPositionHandler = Work.IncrementPosition,
                        OnMessageHandler = m_cpuSolver_OnMessage,
                        OnSolutionHandler = m_cpuSolver_OnSolution
                    };
                }

                NetworkInterface.OnGetMiningParameterStatusEvent += NetworkInterface_OnGetMiningParameterStatusEvent;
                NetworkInterface.OnNewMessagePrefixEvent += NetworkInterface_OnNewMessagePrefixEvent;
                NetworkInterface.OnNewTargetEvent += NetworkInterface_OnNewTargetEvent;

                Solver.setSubmitStale(isSubmitStale);

                if (string.IsNullOrWhiteSpace(devicesStr))
                {
                    Program.Print("[INFO] No CPU assigned.");
                    return;
                }
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        private void m_hashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var hashString = new StringBuilder();
            hashString.Append("OpenCL [INFO] Hashrates:");

            for (uint threadID = 0; threadID < Devices.Count(d => d.DeviceID > -1); threadID++)
                hashString.AppendFormat(" {0} MH/s", Solver.getHashRateByThreadID(threadID) / 1000000.0f);
            
            Program.Print(hashString.ToString());
            Program.Print(string.Format("OpenCL [INFO] Total Hashrate: {0} MH/s", Solver.getTotalHashRate() / 1000000.0f));
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, false);
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

        private void NetworkInterface_OnNewMessagePrefixEvent(NetworkInterface.INetworkInterface sender, string messagePrefix)
        {
            try
            {
                if (Solver != null)
                    Solver.updatePrefix(messagePrefix);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        private void NetworkInterface_OnNewTargetEvent(NetworkInterface.INetworkInterface sender, string target)
        {
            try
            {
                if (Solver != null)
                    Solver.updateTarget(target);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        private void NetworkInterface_OnGetMiningParameterStatusEvent(NetworkInterface.INetworkInterface sender,
                                                                      bool success, NetworkInterface.MiningParameters miningParameters)
        {
            try
            {
                if (Solver != null)
                {
                    if (success)
                    {
                        var isPause = Solver.isPaused();

                        if (!NetworkInterface.IsPool &&
                                ((NetworkInterface.Web3Interface)NetworkInterface).IsChallengedSubmitted(miningParameters.ChallengeNumberByte32String))
                        {
                            isPause = true;
                        }
                        else if (isPause)
                        {
                            if (m_failedScanCount > m_pauseOnFailedScan)
                                m_failedScanCount = 0;

                            isPause = false;
                        }
                        Solver.pauseFinding(isPause);
                    }
                    else
                    {
                        m_failedScanCount += 1;

                        if (m_failedScanCount > m_pauseOnFailedScan && Solver.isMining())
                            Solver.pauseFinding(true);
                    }
                }
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        private void m_cpuSolver_OnSolution(string digest, string address, string challenge, string target, string solution)
        {
            var difficulty = NetworkInterface.Difficulty.ToString("X64");

            NetworkInterface.SubmitSolution(digest, address, challenge, difficulty, target, solution, this);
        }
    }
}

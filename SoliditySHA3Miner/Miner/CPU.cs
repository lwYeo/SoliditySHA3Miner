using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Timers;

namespace SoliditySHA3Miner.Miner
{
    public class CPU : IMiner
    {
        #region P/Invoke interface

        public static class Solver
        {
            public const string SOLVER_NAME = "CPUSoliditySHA3Solver";

            public unsafe delegate void GetSolutionTemplateCallback(byte* solutionTemplate);

            public unsafe delegate void GetKingAddressCallback(byte* kingAddress);

            public delegate void GetWorkPositionCallback(ref ulong lastWorkPosition);

            public delegate void ResetWorkPositionCallback(ref ulong lastWorkPosition);

            public delegate void IncrementWorkPositionCallback(ref ulong lastWorkPosition, ulong incrementSize);

            public delegate void MessageCallback([In]int threadID, [In]StringBuilder type, [In]StringBuilder message);

            public delegate void SolutionCallback([In]StringBuilder digest, [In]StringBuilder address, [In]StringBuilder challenge, [In]StringBuilder target, [In]StringBuilder solution);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetLogicalProcessorsCount(ref uint processorCount);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetNewSolutionTemplate(StringBuilder kingAddress, StringBuilder solutionTemplate);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern IntPtr GetInstance(StringBuilder threads);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void DisposeInstance(IntPtr instance);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static unsafe extern GetSolutionTemplateCallback SetOnGetSolutionTemplateHandler(IntPtr instance, GetSolutionTemplateCallback getSolutionTemplateCallback);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static unsafe extern GetKingAddressCallback SetOnGetKingAddressHandler(IntPtr instance, GetKingAddressCallback getKingAddressCallback);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern GetWorkPositionCallback SetOnGetWorkPositionHandler(IntPtr instance, GetWorkPositionCallback getWorkPositionCallback);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern ResetWorkPositionCallback SetOnResetWorkPositionHandler(IntPtr instance, ResetWorkPositionCallback resetWorkPositionCallback);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern IncrementWorkPositionCallback SetOnIncrementWorkPositionHandler(IntPtr instance, IncrementWorkPositionCallback incrementWorkPositionCallback);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern MessageCallback SetOnMessageHandler(IntPtr instance, MessageCallback messageCallback);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern SolutionCallback SetOnSolutionHandler(IntPtr instance, SolutionCallback solutionCallback);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void SetSubmitStale(IntPtr instance, bool submitStale);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void IsMining(IntPtr instance, ref bool isMining);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void IsPaused(IntPtr instance, ref bool isPaused);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetHashRateByThreadID(IntPtr instance, uint threadID, ref ulong hashRate);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetTotalHashRate(IntPtr instance, ref ulong totalHashRate);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void UpdatePrefix(IntPtr instance, StringBuilder prefix);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void UpdateTarget(IntPtr instance, StringBuilder target);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PauseFinding(IntPtr instance, bool pause);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void StartFinding(IntPtr instance);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void StopFinding(IntPtr instance);
        }

        private Solver.GetSolutionTemplateCallback m_GetSolutionTemplateCallback;
        private Solver.GetKingAddressCallback m_GetKingAddressCallback;
        private Solver.GetWorkPositionCallback m_GetWorkPositionCallback;
        private Solver.ResetWorkPositionCallback m_ResetWorkPositionCallback;
        private Solver.IncrementWorkPositionCallback m_IncrementWorkPositionCallback;
        private Solver.MessageCallback m_MessageCallback;
        private Solver.SolutionCallback m_SolutionCallback;

        #endregion P/Invoke interface

        #region Static

        public static uint GetLogicalProcessorCount()
        {
            var processorCount = 0u;
            Solver.GetLogicalProcessorsCount(ref processorCount);
            return processorCount;
        }

        public static string GetNewSolutionTemplate(string kingAddress = "")
        {
            var solutionTemplate = new StringBuilder(32 * 2 + 2);
            Solver.GetNewSolutionTemplate(new StringBuilder(kingAddress), solutionTemplate);
            return solutionTemplate.ToString();
        }

        #endregion Static

        private const int MAX_SUBMIT_DTM_COUNT = 100;
        private readonly List<DateTime> m_submitDateTimeList;

        private Timer m_hashPrintTimer;
        private int m_pauseOnFailedScan;
        private int m_failedScanCount;

        public readonly IntPtr m_instance;

        #region IMiner

        public NetworkInterface.INetworkInterface NetworkInterface { get; }

        public bool HasAssignedDevices => m_instance != null && m_instance.ToInt64() != 0 && Devices.Any(d => d.DeviceID > -1);

        public bool HasMonitoringAPI => false;

        public Device[] Devices { get; }

        public bool IsAnyInitialised => true; // CPU is always initialised

        public bool IsMining
        {
            get
            {
                var isMining = false;

                if (m_instance != null && m_instance.ToInt64() != 0)
                    Solver.IsMining(m_instance, ref isMining);

                return isMining;
            }
        }

        public bool IsPaused
        {
            get
            {
                var isPaused = false;

                if (m_instance != null && m_instance.ToInt64() != 0)
                    Solver.IsPaused(m_instance, ref isPaused);

                return isPaused;
            }
        }

        public void Dispose()
        {
            try
            {
                if (m_instance != null && m_instance.ToInt64() != 0)
                    Solver.DisposeInstance(m_instance);

                m_GetSolutionTemplateCallback = null;
                m_GetKingAddressCallback = null;
                m_GetWorkPositionCallback = null;
                m_ResetWorkPositionCallback = null;
                m_IncrementWorkPositionCallback = null;
                m_MessageCallback = null;
                m_SolutionCallback = null;
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        public ulong GetHashrateByDevice(string platformName, int deviceID)
        {
            var hashrate = 0ul;

            if (m_instance != null && m_instance.ToInt64() != 0)
                Solver.GetHashRateByThreadID(m_instance, (uint)deviceID, ref hashrate);

            return hashrate;
        }

        public ulong GetTotalHashrate()
        {
            var hashrate = 0ul;

            if (m_instance != null && m_instance.ToInt64() != 0)
                Solver.GetTotalHashRate(m_instance, ref hashrate);

            return hashrate;
        }

        public ulong GetEffectiveHashrate()
        {
            var hashrate = 0ul;

            //if (m_submitDateTimeList.Count() > 1)
            //    hashrate = (ulong)((m_submitDateTimeList.Last() - m_submitDateTimeList.First()).TotalSeconds) * diff;

            return hashrate;
        }

        public void StartMining(int networkUpdateInterval, int hashratePrintInterval)
        {
            try
            {
                NetworkInterface.UpdateMiningParameters();

                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += m_hashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();

                m_submitDateTimeList.Clear();
                m_submitDateTimeList.Add(DateTime.Now);

                Solver.StartFinding(m_instance);
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
                m_submitDateTimeList.Clear();
                Solver.StopFinding(m_instance);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        #endregion IMiner

        public CPU(NetworkInterface.INetworkInterface networkInterface, Device[] devices, bool isSubmitStale, int pauseOnFailedScans)
        {
            try
            {
                Devices = devices;
                NetworkInterface = networkInterface;
                m_pauseOnFailedScan = pauseOnFailedScans;
                m_failedScanCount = 0;
                m_submitDateTimeList = new List<DateTime>(MAX_SUBMIT_DTM_COUNT);

                var devicesStr = string.Empty;
                foreach (var device in Devices)
                {
                    if (device.DeviceID < 0) continue;

                    if (!string.IsNullOrEmpty(devicesStr)) devicesStr += ',';
                    devicesStr += device.DeviceID.ToString("X64");
                }

                m_instance = Solver.GetInstance(new StringBuilder(devicesStr));
                unsafe
                {
                    m_GetSolutionTemplateCallback = Solver.SetOnGetSolutionTemplateHandler(m_instance, Work.GetSolutionTemplate);
                    m_GetKingAddressCallback = Solver.SetOnGetKingAddressHandler(m_instance, Work.GetKingAddress);
                }
                m_GetWorkPositionCallback = Solver.SetOnGetWorkPositionHandler(m_instance, Work.GetPosition);
                m_ResetWorkPositionCallback = Solver.SetOnResetWorkPositionHandler(m_instance, Work.ResetPosition);
                m_IncrementWorkPositionCallback = Solver.SetOnIncrementWorkPositionHandler(m_instance, Work.IncrementPosition);
                m_MessageCallback = Solver.SetOnMessageHandler(m_instance, m_instance_OnMessage);
                m_SolutionCallback = Solver.SetOnSolutionHandler(m_instance, m_instance_OnSolution);

                NetworkInterface.OnGetMiningParameterStatusEvent += NetworkInterface_OnGetMiningParameterStatusEvent;
                NetworkInterface.OnNewMessagePrefixEvent += NetworkInterface_OnNewMessagePrefixEvent;
                NetworkInterface.OnNewTargetEvent += NetworkInterface_OnNewTargetEvent;

                Solver.SetSubmitStale(m_instance, isSubmitStale);

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
            var hashrate = 0ul;
            var hashString = new StringBuilder();
            hashString.Append("OpenCL [INFO] Hashrates:");

            for (uint threadID = 0; threadID < Devices.Count(d => d.DeviceID > -1); threadID++)
            {
                Solver.GetHashRateByThreadID(m_instance, threadID, ref hashrate);
                hashString.AppendFormat(" {0} MH/s", hashrate / 1000000.0f);
            }
            Program.Print(hashString.ToString());

            Solver.GetTotalHashRate(m_instance, ref hashrate);
            Program.Print(string.Format("OpenCL [INFO] Total Hashrate: {0} MH/s", hashrate / 1000000.0f));

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, false);
        }

        private void m_instance_OnMessage(int threadID, StringBuilder type, StringBuilder message)
        {
            var sFormat = new StringBuilder();
            if (threadID > -1) sFormat.Append("CPU Thread: {0} ");

            switch (type.ToString().ToUpperInvariant())
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
                ? string.Format(sFormat.ToString(), threadID, message.ToString())
                : string.Format(sFormat.ToString(), message.ToString()));
        }

        private void NetworkInterface_OnNewMessagePrefixEvent(NetworkInterface.INetworkInterface sender, string messagePrefix)
        {
            try
            {
                if (m_instance != null && m_instance.ToInt64() != 0)
                    Solver.UpdatePrefix(m_instance, new StringBuilder(messagePrefix));
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
                if (m_instance != null && m_instance.ToInt64() != 0)
                    Solver.UpdateTarget(m_instance, new StringBuilder(target));
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
                if (m_instance != null && m_instance.ToInt64() != 0)
                {
                    if (success)
                    {
                        var isPause = false;
                        Solver.IsPaused(m_instance, ref isPause);

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
                        Solver.PauseFinding(m_instance, isPause);
                    }
                    else
                    {
                        m_failedScanCount += 1;

                        var isMining = false;
                        Solver.IsMining(m_instance, ref isMining);

                        if (m_failedScanCount > m_pauseOnFailedScan && IsMining)
                            Solver.PauseFinding(m_instance, true);
                    }
                }
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        private void m_instance_OnSolution(StringBuilder digest, StringBuilder address, StringBuilder challenge, StringBuilder target, StringBuilder solution)
        {
            if (m_submitDateTimeList.Count >= MAX_SUBMIT_DTM_COUNT) m_submitDateTimeList.RemoveAt(0);
            m_submitDateTimeList.Add(DateTime.Now);

            var difficulty = NetworkInterface.Difficulty.ToString("X64");

            NetworkInterface.SubmitSolution(digest.ToString(), address.ToString(), challenge.ToString(), difficulty, target.ToString(), solution.ToString(), this);
        }
    }
}
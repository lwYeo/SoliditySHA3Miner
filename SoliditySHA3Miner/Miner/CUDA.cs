using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Timers;

namespace SoliditySHA3Miner.Miner
{
    public class CUDA : IMiner
    {
        #region P/Invoke interface

        public static class Solver
        {
            public const string SOLVER_NAME = "CudaSoliditySHA3Solver";

            public unsafe delegate void GetSolutionTemplateCallback(byte* solutionTemplate);

            public unsafe delegate void GetKingAddressCallback(byte* kingAddress);

            public delegate void GetWorkPositionCallback(ref ulong lastWorkPosition);

            public delegate void ResetWorkPositionCallback(ref ulong lastWorkPosition);

            public delegate void IncrementWorkPositionCallback(ref ulong lastWorkPosition, ulong incrementSize);

            public delegate void MessageCallback([In]int deviceID, [In]StringBuilder type, [In]StringBuilder message);

            public delegate void SolutionCallback([In]StringBuilder digest, [In]StringBuilder address, [In]StringBuilder challenge, [In]StringBuilder target, [In]StringBuilder solution);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void FoundNvAPI64(ref bool hasNvAPI64);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCount(ref int deviceCount, StringBuilder errorMessage, ref ulong errorSize);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceName(int deviceID, StringBuilder deviceName, ref ulong nameSize, StringBuilder errorMessage, ref ulong errorSize);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern IntPtr GetInstance();

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
            public static extern void AssignDevice(IntPtr instance, int deviceID, ref float intensity);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void IsAssigned(IntPtr instance, ref bool isAssigned);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void IsAnyInitialised(IntPtr instance, ref bool isAnyInitialised);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void IsMining(IntPtr instance, ref bool isMining);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void IsPaused(IntPtr instance, ref bool isPaused);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetHashRateByDeviceID(IntPtr instance, uint deviceID, ref ulong hashRate);

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

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingMaxCoreClock(IntPtr instance, int deviceID, ref int coreClock);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingMaxMemoryClock(IntPtr instance, int deviceID, ref int memoryClock);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingPowerLimit(IntPtr instance, int deviceID, ref int powerLimit);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingThermalLimit(IntPtr instance, int deviceID, ref int thermalLimit);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingFanLevelPercent(IntPtr instance, int deviceID, ref int fanLevel);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentFanTachometerRPM(IntPtr instance, int deviceID, ref int tachometerRPM);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentTemperature(IntPtr instance, int deviceID, ref int temperature);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentCoreClock(IntPtr instance, int deviceID, ref int coreClock);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentMemoryClock(IntPtr instance, int deviceID, ref int memoryClock);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentUtilizationPercent(IntPtr instance, int deviceID, ref int utilization);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentPstate(IntPtr instance, int deviceID, ref int pState);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentThrottleReasons(IntPtr instance, int deviceID, StringBuilder throttleReasons, ref ulong reasonSize);
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

        public static int GetDeviceCount(out string errorMessage)
        {
            errorMessage = string.Empty;
            var errMsg = new StringBuilder(1024);
            var deviceCount = 0;
            var errSize = 0ul;

            Solver.GetDeviceCount(ref deviceCount, errMsg, ref errSize);

            errorMessage = errMsg.ToString();
            return deviceCount;
        }

        public static string GetDeviceName(int deviceID, out string errorMessage)
        {
            errorMessage = string.Empty;
            var errMsg = new StringBuilder(1024);
            var deviceName = new StringBuilder(256);
            ulong nameSize = 0, errSize = 0;

            Solver.GetDeviceName(deviceID, deviceName, ref nameSize, errMsg, ref errSize);

            errorMessage = errMsg.ToString();
            return deviceName.ToString();
        }

        public static string GetDevices(out string errorMessage)
        {
            errorMessage = string.Empty;
            var errMsg = new StringBuilder(1024);
            var deviceName = new StringBuilder(256);
            var devicesString = new StringBuilder();
            ulong nameSize = 0, errSize = 0;
            var cudaCount = 0;

            Solver.GetDeviceCount(ref cudaCount, errMsg, ref errSize);
            errorMessage = errMsg.ToString();

            if (!string.IsNullOrEmpty(errorMessage)) return string.Empty;

            for (int i = 0; i < cudaCount; i++)
            {
                errMsg.Clear();
                deviceName.Clear();

                Solver.GetDeviceName(i, deviceName, ref nameSize, errMsg, ref errSize);
                errorMessage = errMsg.ToString();

                if (!string.IsNullOrEmpty(errorMessage)) return string.Empty;

                devicesString.AppendLine(string.Format("{0}: {1}", i, deviceName.ToString()));
            }
            return devicesString.ToString();
        }

        #endregion Static

        private Timer m_hashPrintTimer;
        private int m_pauseOnFailedScan;
        private int m_failedScanCount;

        public readonly IntPtr m_instance;

        #region IMiner

        public NetworkInterface.INetworkInterface NetworkInterface { get; }

        public Device[] Devices { get; }

        public bool HasAssignedDevices
        {
            get
            {
                var isAssigned = false;

                if (m_instance != null && m_instance.ToInt64() != 0)
                    Solver.IsAssigned(m_instance, ref isAssigned);

                return isAssigned;
            }
        }

        public bool HasMonitoringAPI { get; private set; }

        public bool IsAnyInitialised
        {
            get
            {
                var isAnyInitialised = false;

                if (m_instance != null && m_instance.ToInt64() != 0)
                    Solver.IsAnyInitialised(m_instance, ref isAnyInitialised);

                return isAnyInitialised;
            }
        }

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

        public void StartMining(int networkUpdateInterval, int hashratePrintInterval)
        {
            try
            {
                NetworkInterface.UpdateMiningParameters();

                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += m_hashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();

                NetworkInterface.ResetEffectiveHashrate();
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

                NetworkInterface.ResetEffectiveHashrate();

                Solver.StopFinding(m_instance);
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
                Solver.GetHashRateByDeviceID(m_instance, (uint)deviceID, ref hashrate);

            return hashrate;
        }

        public ulong GetTotalHashrate()
        {
            var hashrate = 0ul;

            if (m_instance != null && m_instance.ToInt64() != 0)
                Solver.GetTotalHashRate(m_instance, ref hashrate);

            return hashrate;
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

        #endregion IMiner

        public CUDA(NetworkInterface.INetworkInterface networkInterface, Device[] cudaDevices, bool isSubmitStale, int pauseOnFailedScans)
        {
            try
            {
                Devices = cudaDevices;
                NetworkInterface = networkInterface;
                m_pauseOnFailedScan = pauseOnFailedScans;
                m_failedScanCount = 0;

                var hasNvAPI64 = false;
                Solver.FoundNvAPI64(ref hasNvAPI64);
                HasMonitoringAPI = hasNvAPI64;

                NetworkInterface.OnGetTotalHashrate += NetworkInterface_OnGetTotalHashrate;

                if (!HasMonitoringAPI) Program.Print("[WARN] NvAPI64 library not found.");

                m_instance = Solver.GetInstance();
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

                if (cudaDevices.All(d => d.DeviceID == -1))
                {
                    Program.Print("[INFO] CUDA device not set.");
                    return;
                }

                for (int i = 0; i < Devices.Length; i++)
                    if (Devices[i].DeviceID > -1)
                        Solver.AssignDevice(m_instance, Devices[i].DeviceID, ref Devices[i].Intensity);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        private void NetworkInterface_OnGetTotalHashrate(NetworkInterface.INetworkInterface sender, ref ulong totalHashrate)
        {
            try
            {
                var hashrate = 0ul;
                Solver.GetTotalHashRate(m_instance, ref hashrate);

                totalHashrate += hashrate;
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
            hashString.Append("CUDA [INFO] Hashrates:");

            foreach (var device in Devices)
            {
                if (device.DeviceID > -1)
                {
                    Solver.GetHashRateByDeviceID(m_instance, (uint)device.DeviceID, ref hashrate);
                    hashString.AppendFormat(" {0} MH/s", hashrate / 1000000.0f);
                }
            }
            Program.Print(hashString.ToString());
            
            if (HasMonitoringAPI)
            {
                var coreClock = 0;
                var temperature = 0;
                var tachometerRPM = 0;
                var coreClockString = new StringBuilder();
                var temperatureString = new StringBuilder();
                var fanTachometerRpmString = new StringBuilder();

                coreClockString.Append("CUDA [INFO] Core clocks:");
                foreach (var device in Devices)
                    if (device.DeviceID > -1)
                    {
                        Solver.GetDeviceCurrentCoreClock(m_instance, device.DeviceID, ref coreClock);
                        coreClockString.AppendFormat(" {0}MHz", coreClock);
                    }
                Program.Print(coreClockString.ToString());

                temperatureString.Append("CUDA [INFO] Temperatures:");
                foreach (var device in Devices)
                    if (device.DeviceID > -1)
                    {
                        Solver.GetDeviceCurrentTemperature(m_instance, device.DeviceID, ref temperature);
                        temperatureString.AppendFormat(" {0}C", temperature);
                    }
                Program.Print(temperatureString.ToString());

                fanTachometerRpmString.Append("CUDA [INFO] Fan tachometers:");
                foreach (var device in Devices)
                    if (device.DeviceID > -1)
                        if (device.DeviceID > -1)
                        {
                            Solver.GetDeviceCurrentFanTachometerRPM(m_instance, device.DeviceID, ref tachometerRPM);
                            fanTachometerRpmString.AppendFormat(" {0}RPM", tachometerRPM);
                        }
                Program.Print(fanTachometerRpmString.ToString());
            }

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, false);
        }

        private void m_instance_OnMessage(int deviceID, StringBuilder type, StringBuilder message)
        {
            var sFormat = new StringBuilder();
            if (deviceID > -1) sFormat.Append("CUDA ID: {0} ");
            else sFormat.Append("CUDA ");

            switch (type.ToString().ToUpperInvariant())
            {
                case "INFO":
                    sFormat.Append(deviceID > -1 ? "[INFO] {1}" : "[INFO] {0}");
                    break;

                case "WARN":
                    sFormat.Append(deviceID > -1 ? "[WARN] {1}" : "[WARN] {0}");
                    break;

                case "ERROR":
                    sFormat.Append(deviceID > -1 ? "[ERROR] {1}" : "[ERROR] {0}");
                    break;

                case "DEBUG":
                default:
#if DEBUG
                    sFormat.Append(deviceID > -1 ? "[DEBUG] {1}" : "[DEBUG] {0}");
                    break;
#else
                    return;
#endif
            }
            Program.Print(deviceID > -1
                ? string.Format(sFormat.ToString(), deviceID, message.ToString())
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

        private void NetworkInterface_OnGetMiningParameterStatusEvent(NetworkInterface.INetworkInterface sender, bool success, NetworkInterface.MiningParameters miningParameters)
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
            var difficulty = NetworkInterface.Difficulty.ToString("X64");

            NetworkInterface.SubmitSolution(digest.ToString(), address.ToString(), challenge.ToString(), difficulty, target.ToString(), solution.ToString(), this);
        }
    }
}
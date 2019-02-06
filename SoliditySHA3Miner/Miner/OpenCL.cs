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
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.Miner
{
    public class OpenCL : MinerBase
    {
        public bool UseLinuxQuery { get; private set; }

        public OpenCL(NetworkInterface.INetworkInterface networkInterface, Device.OpenCL[] intelDevices, Device.OpenCL[] amdDevices, bool isSubmitStale, int pauseOnFailedScans)
            : base(networkInterface, amdDevices.Union(intelDevices).ToArray(), isSubmitStale, pauseOnFailedScans)
        {
            try
            {
                NetworkInterface = networkInterface;
                m_pauseOnFailedScan = pauseOnFailedScans;
                m_failedScanCount = 0;

                var hasADL_API = false;
                Helper.OpenCL.Solver.FoundADL_API(ref hasADL_API);
                if (!hasADL_API) Program.Print("OpenCL [WARN] ADL library not found.");

                if (!hasADL_API && RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                    UseLinuxQuery = API.AmdLinuxQuery.QuerySuccess();

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && !UseLinuxQuery)
                    Program.Print("OpenCL [WARN] Unable to find AMD devices in Linux kernel.");

                HasMonitoringAPI = hasADL_API || UseLinuxQuery;

                if (!HasMonitoringAPI) Program.Print("OpenCL [WARN] GPU monitoring not available.");

                UnmanagedInstance = Helper.OpenCL.Solver.GetInstance();

                AssignDevices();
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("OpenCL [ERROR] {0}", ex.Message));
            }
        }

        #region IMiner

        public override void Dispose()
        {
            try
            {
                var disposeTask = Task.Factory.StartNew(() => base.Dispose());

                if (UnmanagedInstance != IntPtr.Zero)
                    Helper.OpenCL.Solver.DisposeInstance(UnmanagedInstance);

                if (!disposeTask.IsCompleted)
                    disposeTask.Wait();
            }
            catch { }
        }

        #endregion IMiner

        #region MinerBase abstracts

        protected override void HashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var errorMessage = new StringBuilder(1024);
            var hashString = new StringBuilder();
            hashString.Append("Hashrates:");

            foreach (var device in Devices.Where(d => d.AllowDevice))
                hashString.AppendFormat(" {0} MH/s", GetHashRateByDevice(device) / 1000000.0f);

            PrintMessage("OpenCL", string.Empty, -1, "Info", hashString.ToString());

            if (HasMonitoringAPI)
            {
                var coreClock = 0;
                var temperature = 0;
                var tachometerRPM = 0;
                var coreClockString = new StringBuilder();
                var temperatureString = new StringBuilder();
                var fanTachometerRpmString = new StringBuilder();

                coreClockString.Append("Core clocks:");
                foreach (var device in Devices)
                    if (device.AllowDevice)
                    {
                        errorMessage.Clear();

                        if (UseLinuxQuery)
                            coreClock = API.AmdLinuxQuery.GetDeviceCurrentCoreClock(device.PciBusID);
                        else
                            Helper.OpenCL.Solver.GetDeviceCurrentCoreClock(((Device.OpenCL)device).DeviceCL_Struct, ref coreClock, errorMessage);

                        coreClockString.AppendFormat(" {0}MHz", coreClock);
                    }
                PrintMessage("OpenCL", string.Empty, -1, "Info", coreClockString.ToString());

                temperatureString.Append("Temperatures:");
                foreach (var device in Devices)
                    if (device.AllowDevice)
                    {
                        errorMessage.Clear();

                        if (UseLinuxQuery)
                            temperature = API.AmdLinuxQuery.GetDeviceCurrentTemperature(device.PciBusID);
                        else
                            Helper.OpenCL.Solver.GetDeviceCurrentTemperature(((Device.OpenCL)device).DeviceCL_Struct, ref temperature, errorMessage);

                        temperatureString.AppendFormat(" {0}C", temperature);
                    }
                PrintMessage("OpenCL", string.Empty, -1, "Info", temperatureString.ToString());

                fanTachometerRpmString.Append("Fan tachometers:");
                foreach (var device in Devices)
                    if (device.AllowDevice)
                    {
                        errorMessage.Clear();

                        if (UseLinuxQuery)
                            tachometerRPM = API.AmdLinuxQuery.GetDeviceCurrentFanTachometerRPM(device.PciBusID);
                        else
                            Helper.OpenCL.Solver.GetDeviceCurrentFanTachometerRPM(((Device.OpenCL)device).DeviceCL_Struct, ref tachometerRPM, errorMessage);

                        fanTachometerRpmString.AppendFormat(" {0}RPM", tachometerRPM);
                    }
                PrintMessage("OpenCL", string.Empty, -1, "Info", fanTachometerRpmString.ToString());
            }

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, false);
        }

        protected override void AssignDevices()
        {
            if ((!Program.AllowIntel && !Program.AllowAMD) || (Devices.All(d => !d.AllowDevice)))
            {
                PrintMessage("OpenCL", string.Empty, -1, "Info", "Device not set.");
                return;
            }

            var deviceName = new StringBuilder(256);
            var errorMessage = new StringBuilder(1024);
            var isKingMaking = !string.IsNullOrWhiteSpace(Work.GetKingAddressString());
            var intelDevices = Devices.Where(d => d.Platform.IndexOf("Intel(R) OpenCL", StringComparison.OrdinalIgnoreCase) > -1).ToArray();
            var amdDevices = Devices.Where(d => d.Platform.IndexOf("AMD Accelerated Parallel Processing", StringComparison.OrdinalIgnoreCase) > -1).ToArray();

            if (Program.AllowIntel)
            {
                foreach (Device.OpenCL device in intelDevices.Where(d => d.AllowDevice))
                {
                    PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Assigning device...");
                    errorMessage.Clear();

                    device.DeviceCL_Struct.MaxSolutionCount = Device.DeviceBase.MAX_SOLUTION_COUNT;
                    device.DeviceCL_Struct.LocalWorkSize = Device.OpenCL.DEFAULT_LOCAL_WORK_SIZE_INTEL;
                    device.DeviceCL_Struct.Intensity = device.Intensity >= 1.0f
                                                     ? device.Intensity
                                                     : Device.OpenCL.DEFAULT_INTENSITY_INTEL;

                    device.Intensity = device.DeviceCL_Struct.Intensity;

                    device.Solutions = (ulong[])Array.CreateInstance(typeof(ulong), Device.DeviceBase.MAX_SOLUTION_COUNT);
                    var solutionsHandle = GCHandle.Alloc(device.Solutions, GCHandleType.Pinned);
                    device.DeviceCL_Struct.Solutions = solutionsHandle.AddrOfPinnedObject();
                    device.AddHandle(solutionsHandle);

                    Helper.OpenCL.Solver.InitializeDevice(UnmanagedInstance, ref device.DeviceCL_Struct, isKingMaking, errorMessage);
                    
                    if (errorMessage.Length > 0)
                        PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
                    else
                    {
                        device.IsInitialized = true;

                        device.Name = device.DeviceCL_Struct.NameToString();
                        device.PciBusID = (uint)device.DeviceCL_Struct.PciBusID;
                        device.IsAssigned = true;

                        PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", string.Format("Assigned device ({0})...", device.Name));
                        PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", string.Format("Intensity: {0}", device.Intensity));
                    }
                }
            }
            else
            {
                Devices.Where(d => d.Platform.IndexOf("Intel(R) OpenCL", StringComparison.OrdinalIgnoreCase) > -1).
                        AsParallel().
                        ForAll(d => d.AllowDevice = false);
            }

            if (Program.AllowAMD)
            {
                foreach (Device.OpenCL device in amdDevices.Where(d => d.AllowDevice))
                {
                    PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Assigning device...");
                    errorMessage.Clear();
                    deviceName.Clear();

                    device.DeviceCL_Struct.MaxSolutionCount = Device.DeviceBase.MAX_SOLUTION_COUNT;
                    device.DeviceCL_Struct.LocalWorkSize = Device.OpenCL.DEFAULT_LOCAL_WORK_SIZE;
                    device.DeviceCL_Struct.Intensity = device.Intensity >= 1.0f
                                                     ? device.Intensity
                                                     : isKingMaking
                                                     ? Device.OpenCL.DEFAULT_INTENSITY_KING
                                                     : Device.OpenCL.DEFAULT_INTENSITY;

                    device.Intensity = device.DeviceCL_Struct.Intensity;

                    device.Solutions = (ulong[])Array.CreateInstance(typeof(ulong), Device.DeviceBase.MAX_SOLUTION_COUNT);
                    var solutionsHandle = GCHandle.Alloc(device.Solutions, GCHandleType.Pinned);
                    device.DeviceCL_Struct.Solutions = solutionsHandle.AddrOfPinnedObject();
                    device.AddHandle(solutionsHandle);

                    Helper.OpenCL.Solver.InitializeDevice(UnmanagedInstance, ref device.DeviceCL_Struct, isKingMaking, errorMessage);
                    
                    if (errorMessage.Length > 0)
                        PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
                    else
                    {
                        device.IsInitialized = true;

                        device.Name = UseLinuxQuery
                                    ? API.AmdLinuxQuery.GetDeviceRealName((uint)device.DeviceCL_Struct.PciBusID, "Unknown AMD GPU")
                                    : device.DeviceCL_Struct.NameToString();

                        device.PciBusID = (uint)device.DeviceCL_Struct.PciBusID;
                        device.IsAssigned = true;

                        PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", string.Format("Assigned OpenCL device ({0})...", device.Name));
                        PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", string.Format("Intensity: {0}", device.Intensity));
                    }
                }
            }
            else
            {
                Devices.Where(d => d.Platform.IndexOf("AMD Accelerated Parallel Processing", StringComparison.OrdinalIgnoreCase) > -1).
                        AsParallel().
                        ForAll(d => d.AllowDevice = false);
            }
        }

        protected override void PushHigh64Target(Device.DeviceBase device)
        {
            var errorMessage = new StringBuilder(1024);
            Helper.OpenCL.Solver.PushHigh64Target(UnmanagedInstance, ((Device.OpenCL)device).DeviceCL_Struct, device.CommonPointers.High64Target, errorMessage);

            if (errorMessage.Length > 0)
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
        }

        protected override void PushTarget(Device.DeviceBase device)
        {
            var errorMessage = new StringBuilder(1024);
            Helper.OpenCL.Solver.PushTarget(UnmanagedInstance, ((Device.OpenCL)device).DeviceCL_Struct, device.CommonPointers.Target, errorMessage);

            if (errorMessage.Length > 0)
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
        }

        protected override void PushMidState(Device.DeviceBase device)
        {
            var errorMessage = new StringBuilder(1024);
            Helper.OpenCL.Solver.PushMidState(UnmanagedInstance, ((Device.OpenCL)device).DeviceCL_Struct, device.CommonPointers.MidState, errorMessage);

            if (errorMessage.Length > 0)
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
        }

        protected override void PushMessage(Device.DeviceBase device)
        {
            var errorMessage = new StringBuilder(1024);
            Helper.OpenCL.Solver.PushMessage(UnmanagedInstance, ((Device.OpenCL)device).DeviceCL_Struct, device.CommonPointers.Message, errorMessage);

            if (errorMessage.Length > 0)
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
        }

        protected override void StartFinding(Device.DeviceBase device, bool isKingMaking)
        {
            var deviceCL = ((Device.OpenCL)device).DeviceCL_Struct;
            try
            {
                if (!device.IsInitialized) return;

                while (!device.HasNewTarget || !device.HasNewChallenge)
                    Task.Delay(500).Wait();

                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Start mining...");

                PrintMessage(device.Type, device.Platform, device.DeviceID, "Debug",
                             string.Format("Threads: {0} Local work size: {1} Block size: {2}",
                                           deviceCL.GlobalWorkSize, deviceCL.LocalWorkSize, deviceCL.GlobalWorkSize / deviceCL.LocalWorkSize));

                var errorMessage = new StringBuilder(1024);
                var currentChallenge = (byte[])Array.CreateInstance(typeof(byte), UINT256_LENGTH);

                device.HashStartTime = DateTime.Now;
                device.HashCount = 0;
                device.IsMining = true;
                deviceCL.SolutionCount = 0;

                do
                {
                    while (device.IsPause)
                    {
                        Task.Delay(500).Wait();
                        device.HashStartTime = DateTime.Now;
                        device.HashCount = 0;
                    }

                    CheckInputs(device, isKingMaking, ref currentChallenge);

                    Work.IncrementPosition(ref deviceCL.WorkPosition, deviceCL.GlobalWorkSize);
                    device.HashCount += deviceCL.GlobalWorkSize;

                    Helper.OpenCL.Solver.Hash(UnmanagedInstance,
                                                ref deviceCL,
                                                errorMessage);
                    if (errorMessage.Length > 0)
                    {
                        PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
                        device.IsMining = false;
                    }

                    if (deviceCL.SolutionCount > 0)
                        lock (((Device.OpenCL)device).Solutions)
                            if (deviceCL.SolutionCount > 0)
                            {
                                var tempSolutionCount = deviceCL.SolutionCount;
                                var tempSolutions = ((Device.OpenCL)device).Solutions.Where(s => s != 0).ToArray();
                                
                                SubmitSolutions(tempSolutions, currentChallenge,
                                                device.Type, device.Platform, device.DeviceID,
                                                tempSolutionCount, isKingMaking);

                                deviceCL.SolutionCount = 0;
                            }
                } while (device.IsMining);

                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Stop mining...");

                device.HashCount = 0;

                Helper.OpenCL.Solver.ReleaseDeviceObjects(UnmanagedInstance, ref deviceCL, errorMessage);
                if (errorMessage.Length > 0)
                {
                    PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
                    errorMessage.Clear();
                }

                device.IsStopped = true;
                device.IsInitialized = false;
            }
            catch (Exception ex)
            {
                PrintMessage(device.Type, device.Platform, -1, "Error", ex.Message);
            }
            PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Mining stopped.");
        }

        #endregion
    }
}
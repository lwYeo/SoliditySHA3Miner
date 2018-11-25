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
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.Miner
{
    public class CUDA : MinerBase
    {
        public bool UseNvSMI { get; protected set; }

        public CUDA(NetworkInterface.INetworkInterface networkInterface, Device.CUDA[] cudaDevices, bool isSubmitStale, int pauseOnFailedScans)
            : base(networkInterface, cudaDevices, isSubmitStale, pauseOnFailedScans)
        {
            try
            {
                var hasNvAPI64 = false;
                Helper.CUDA.Solver.FoundNvAPI64(ref hasNvAPI64);

                if (!hasNvAPI64)
                    PrintMessage("CUDA", string.Empty, -1, "Warn", "NvAPI64 library not found.");

                var foundNvSMI = API.NvSMI.FoundNvSMI();

                if (!foundNvSMI)
                    PrintMessage("CUDA", string.Empty, -1, "Warn", "NvSMI library not found.");

                UseNvSMI = !hasNvAPI64 && foundNvSMI;

                HasMonitoringAPI = hasNvAPI64 | UseNvSMI;

                if (!HasMonitoringAPI)
                    PrintMessage("CUDA", string.Empty, -1, "Warn", "GPU monitoring not available.");

                UnmanagedInstance = Helper.CUDA.Solver.GetInstance();

                AssignDevices();
            }
            catch (Exception ex)
            {
                PrintMessage("CUDA", string.Empty, -1, "Error", ex.Message);
            }
        }

        #region IMiner

        public override void Dispose()
        {
            try
            {
                var disposeTask = Task.Factory.StartNew(() => base.Dispose());

                if (UnmanagedInstance != IntPtr.Zero)
                    Helper.CUDA.Solver.DisposeInstance(UnmanagedInstance);

                if (!disposeTask.IsCompleted)
                    disposeTask.Wait();
            }
            catch { }
        }

        #endregion IMiner

        #region MinerBase abstracts

        protected override void HashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var hashString = new StringBuilder();
            hashString.Append("Hashrates:");

            foreach (var device in Devices.Where(d => d.AllowDevice))
                hashString.AppendFormat(" {0} MH/s", GetHashRateByDevice(device) / 1000000.0f);

            PrintMessage("CUDA", string.Empty, -1, "Info", hashString.ToString());

            if (HasMonitoringAPI)
            {
                var coreClock = 0;
                var temperature = 0;
                var tachometerRPM = 0;
                var coreClockString = new StringBuilder();
                var temperatureString = new StringBuilder();
                var fanTachometerRpmString = new StringBuilder();

                coreClockString.Append("Core clocks:");
                foreach (Device.CUDA device in Devices)
                    if (device.AllowDevice)
                    {
                        if (UseNvSMI)
                            coreClock = API.NvSMI.GetDeviceCurrentCoreClock(device.PciBusID);
                        else
                            Helper.CUDA.Solver.GetDeviceCurrentCoreClock(device.DeviceCUDA_Struct, ref coreClock);
                        coreClockString.AppendFormat(" {0}MHz", coreClock);
                    }
                PrintMessage("CUDA", string.Empty, -1, "Info", coreClockString.ToString());

                temperatureString.Append("Temperatures:");
                foreach (Device.CUDA device in Devices)
                    if (device.AllowDevice)
                    {
                        if (UseNvSMI)
                            temperature = API.NvSMI.GetDeviceCurrentTemperature(device.PciBusID);
                        else
                            Helper.CUDA.Solver.GetDeviceCurrentTemperature(device.DeviceCUDA_Struct, ref temperature);
                        temperatureString.AppendFormat(" {0}C", temperature);
                    }
                PrintMessage("CUDA", string.Empty, -1, "Info", temperatureString.ToString());

                if (!UseNvSMI)
                {
                    fanTachometerRpmString.Append("Fan tachometers:");
                    foreach (Device.CUDA device in Devices)
                        if (device.AllowDevice)
                        {
                            Helper.CUDA.Solver.GetDeviceCurrentFanTachometerRPM(device.DeviceCUDA_Struct, ref tachometerRPM);
                            fanTachometerRpmString.AppendFormat(" {0}RPM", tachometerRPM);
                        }
                    PrintMessage("CUDA", string.Empty, -1, "Info", fanTachometerRpmString.ToString());
                }
            }

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, false);
        }

        protected override void AssignDevices()
        {
            if (!Program.AllowCUDA || Devices.All(d => !d.AllowDevice))
            {
                PrintMessage("CUDA", string.Empty, -1, "Info", "Device not set.");
                return;
            }

            var isKingMaking = !string.IsNullOrWhiteSpace(Work.GetKingAddressString());

            foreach (Device.CUDA device in Devices.Where(d => d.AllowDevice))
            {
                var errorMessage = new StringBuilder(1024);
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Assigning device...");

                device.DeviceCUDA_Struct.DeviceID = device.DeviceID;
                Helper.CUDA.Solver.GetDeviceProperties(UnmanagedInstance, ref device.DeviceCUDA_Struct, errorMessage);
                if (errorMessage.Length > 0)
                {
                    PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
                    return;
                }

                if (device.DeviceCUDA_Struct.ComputeMajor < 5)
                    device.Intensity = (device.Intensity < 1.000f) ? Device.CUDA.DEFAULT_INTENSITY : device.Intensity; // For older GPUs
                else
                {
                    float defaultIntensity = Device.CUDA.DEFAULT_INTENSITY;

                    if (isKingMaking)
                    {
                        if (new string[] { "2080", "2070", "1080" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 27.54f;

                        else if (new string[] { "2060", "1070 TI", "1070TI" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 27.46f;

                        else if (new string[] { "2050", "1070", "980" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 27.01f;

                        else if (new string[] { "1060", "970" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 26.01f;

                        else if (new string[] { "1050", "960" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 25.01f;
                    }
                    else
                    {
                        if (new string[] { "2080", "2070 TI", "2070TI", "1080 TI", "1080TI" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 27.00f;

                        else if (new string[] { "1080", "2070", "1070 TI", "1070TI" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 26.33f;

                        else if (new string[] { "2060", "1070", "980" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 26.00f;

                        else if (new string[] { "2050", "1060", "970" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 25.50f;

                        else if (new string[] { "1050", "960" }.Any(m => device.Name.IndexOf(m) > -1))
                            defaultIntensity = 25.00f;
                    }
                    device.Intensity = (device.Intensity < 1.000f) ? defaultIntensity : device.Intensity;
                }

                device.PciBusID = (uint)device.DeviceCUDA_Struct.PciBusID;
                device.ConputeVersion = (uint)((device.DeviceCUDA_Struct.ComputeMajor * 100) + (device.DeviceCUDA_Struct.ComputeMinor * 10));
                device.DeviceCUDA_Struct.MaxSolutionCount = Device.DeviceBase.MAX_SOLUTION_COUNT;
                device.DeviceCUDA_Struct.Intensity = device.Intensity;
                device.DeviceCUDA_Struct.Threads = device.Threads;
                device.DeviceCUDA_Struct.Block = device.Block;
                device.DeviceCUDA_Struct.Grid = device.Grid;
                device.IsAssigned = true;

                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", string.Format("Assigned device ({0})...", device.Name));
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", string.Format("Compute capability: {0}.{1}", device.DeviceCUDA_Struct.ComputeMajor, device.DeviceCUDA_Struct.ComputeMinor));
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", string.Format("Intensity: {0}", device.Intensity));

                if (!device.IsInitialized)
                {
                    PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Initializing device...");
                    errorMessage.Clear();

                    Helper.CUDA.Solver.InitializeDevice(UnmanagedInstance, ref device.DeviceCUDA_Struct, errorMessage);

                    if (errorMessage.Length > 0)
                        PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
                    else
                        device.IsInitialized = true;
                }
            }
        }

        protected override void PushHigh64Target(Device.DeviceBase device)
        {
            var errorMessage = new StringBuilder(1024);
            Helper.CUDA.Solver.PushHigh64Target(UnmanagedInstance, device.CommonPointers.High64Target, errorMessage);

            if (errorMessage.Length > 0)
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
        }

        protected override void PushTarget(Device.DeviceBase device)
        {
            var errorMessage = new StringBuilder(1024);
            Helper.CUDA.Solver.PushTarget(UnmanagedInstance, device.CommonPointers.Target, errorMessage);

            if (errorMessage.Length > 0)
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
        }

        protected override void PushMidState(Device.DeviceBase device)
        {
            var errorMessage = new StringBuilder(1024);
            Helper.CUDA.Solver.PushMidState(UnmanagedInstance, device.CommonPointers.MidState, errorMessage);

            if (errorMessage.Length > 0)
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
        }

        protected override void PushMessage(Device.DeviceBase device)
        {
            var errorMessage = new StringBuilder(1024);
            Helper.CUDA.Solver.PushMessage(UnmanagedInstance, device.CommonPointers.Message, errorMessage);

            if (errorMessage.Length > 0)
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Error", errorMessage.ToString());
        }

        protected override void StartFinding(Device.DeviceBase device, bool isKingMaking)
        {
            var deviceCUDA = (Device.CUDA)device;
            try
            {
                if (!deviceCUDA.IsInitialized) return;

                while (!deviceCUDA.HasNewTarget || !deviceCUDA.HasNewChallenge)
                    Task.Delay(500).Wait();

                PrintMessage(device.Type, device.Platform, deviceCUDA.DeviceID, "Info", "Start mining...");

                PrintMessage(device.Type, device.Platform, deviceCUDA.DeviceID, "Debug",
                             string.Format("Threads: {0} Grid size: {1} Block size: {2}",
                                           deviceCUDA.Threads, deviceCUDA.Grid.X, deviceCUDA.Block.X));

                var errorMessage = new StringBuilder(1024);
                var currentChallenge = (byte[])Array.CreateInstance(typeof(byte), UINT256_LENGTH);

                Helper.CUDA.Solver.SetDevice(UnmanagedInstance, deviceCUDA.DeviceID, errorMessage);
                if (errorMessage.Length > 0)
                {
                    PrintMessage(device.Type, device.Platform, deviceCUDA.DeviceID, "Error", errorMessage.ToString());
                    return;
                }

                deviceCUDA.HashStartTime = DateTime.Now;
                deviceCUDA.HashCount = 0;
                deviceCUDA.IsMining = true;

                unsafe
                {
                    ulong* solutions = (ulong*)deviceCUDA.DeviceCUDA_Struct.Solutions.ToPointer();
                    uint* solutionCount = (uint*)deviceCUDA.DeviceCUDA_Struct.SolutionCount.ToPointer();
                    *solutionCount = 0;
                    do
                    {
                        while (deviceCUDA.IsPause)
                        {
                            Task.Delay(500).Wait();
                            deviceCUDA.HashStartTime = DateTime.Now;
                            deviceCUDA.HashCount = 0;
                        }

                        CheckInputs(deviceCUDA, isKingMaking, ref currentChallenge);

                        Work.IncrementPosition(ref deviceCUDA.DeviceCUDA_Struct.WorkPosition, deviceCUDA.Threads);
                        deviceCUDA.HashCount += deviceCUDA.Threads;

                        if (isKingMaking)
                            Helper.CUDA.Solver.HashMessage(UnmanagedInstance, ref deviceCUDA.DeviceCUDA_Struct, errorMessage);
                        else
                            Helper.CUDA.Solver.HashMidState(UnmanagedInstance, ref deviceCUDA.DeviceCUDA_Struct, errorMessage);

                        if (errorMessage.Length > 0)
                        {
                            PrintMessage(device.Type, device.Platform, deviceCUDA.DeviceID, "Error", errorMessage.ToString());
                            deviceCUDA.IsMining = false;
                        }

                        if (*solutionCount > 0)
                        {
                            var solutionArray = (ulong[])Array.CreateInstance(typeof(ulong), *solutionCount);

                            for (var i = 0; i < *solutionCount; i++)
                                solutionArray[i] = solutions[i];

                            SubmitSolutions(solutionArray, currentChallenge, device.Type, device.Platform, deviceCUDA.DeviceID, *solutionCount, isKingMaking);

                            *solutionCount = 0;
                        }
                    } while (deviceCUDA.IsMining);
                }

                PrintMessage(device.Type, device.Platform, deviceCUDA.DeviceID, "Info", "Stop mining...");

                deviceCUDA.HashCount = 0;

                Helper.CUDA.Solver.ReleaseDeviceObjects(UnmanagedInstance, ref deviceCUDA.DeviceCUDA_Struct, errorMessage);
                if (errorMessage.Length > 0)
                {
                    PrintMessage(device.Type, device.Platform, deviceCUDA.DeviceID, "Error", errorMessage.ToString());
                    errorMessage.Clear();
                }

                Helper.CUDA.Solver.ResetDevice(UnmanagedInstance, deviceCUDA.DeviceID, errorMessage);
                if (errorMessage.Length > 0)
                {
                    PrintMessage(device.Type, device.Platform, deviceCUDA.DeviceID, "Error", errorMessage.ToString());
                    errorMessage.Clear();
                }

                deviceCUDA.IsStopped = true;
                deviceCUDA.IsInitialized = false;
            }
            catch (Exception ex)
            {
                PrintMessage(device.Type, device.Platform, -1, "Error", ex.Message);
            }
            PrintMessage(device.Type, device.Platform, deviceCUDA.DeviceID, "Info", "Mining stopped.");
        }

        #endregion
    }
}
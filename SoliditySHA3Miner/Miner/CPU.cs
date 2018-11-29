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
    public class CPU : MinerBase
    {
        public CPU(NetworkInterface.INetworkInterface networkInterface, Device.CPU devices, bool isSubmitStale, int pauseOnFailedScans)
            : base(networkInterface, new Device.DeviceBase[] { devices }, isSubmitStale, pauseOnFailedScans)
        {
            try
            {
                HasMonitoringAPI = false;

                UnmanagedInstance = Helper.CPU.Solver.GetInstance();

                AssignDevices();
            }
            catch (Exception ex)
            {
                PrintMessage("CPU", string.Empty, -1, "Error", ex.Message);
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

            PrintMessage("CPU", string.Empty, -1, "Info", hashString.ToString());
            
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, false);
        }

        protected override void AssignDevices()
        {
            if (!Program.AllowCPU || Devices.All(d => !d.AllowDevice))
            {
                PrintMessage("CPU", string.Empty, -1, "Info", "Affinity not set.");
                return;
            }

            var isKingMaking = !string.IsNullOrWhiteSpace(Work.GetKingAddressString());

            foreach (Device.CPU device in Devices.Where(d => d.AllowDevice))
            {
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Assigning device...");

                var cpuName = new StringBuilder(256);

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    Helper.CPU.Solver.GetCpuName(cpuName);
                    
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    if (System.IO.File.Exists("/proc/cpuinfo"))
                    {
                        var cpuInfo = System.IO.File.ReadAllLines("/proc/cpuinfo");
                        var cpuNameLine = cpuInfo.FirstOrDefault(i => i.StartsWith("model name", StringComparison.OrdinalIgnoreCase));

                        if (!string.IsNullOrWhiteSpace(cpuNameLine))
                            cpuName.Append(cpuNameLine.Split(':')[1].Trim());
                    }
                }

                device.Name = (cpuName.Length > 0)
                            ? cpuName.ToString().Trim()
                            : "Unknown CPU";

                var affinities = new StringBuilder();
                affinities.AppendJoin(',', device.Affinities);

                device.IsAssigned = true;

                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", string.Format("Assigned device ({0})...", device.Name));
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", string.Format("Affinities: {0}", affinities.ToString()));

                if (!device.IsInitialized)
                {
                    PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Initializing device...");

                    device.DeviceCPU_Struct.ProcessorCount = device.Affinities.Length;
                    device.Processor_Structs = (Structs.Processor[])Array.CreateInstance(typeof(Structs.Processor), device.Affinities.Length);
                    device.Solutions = (ulong[][])Array.CreateInstance(typeof(ulong[]), device.Affinities.Length);
                    
                    for (var i = 0; i < device.Affinities.Length; i++)
                    {
                        device.Processor_Structs[i].Affinity = device.Affinities[i];
                        device.Processor_Structs[i].WorkSize = (ulong)Math.Pow(2, 16);
                        device.Processor_Structs[i].MaxSolutionCount = Device.DeviceBase.MAX_SOLUTION_COUNT;

                        device.Solutions[i] = (ulong[])Array.CreateInstance(typeof(ulong), Device.DeviceBase.MAX_SOLUTION_COUNT);
                        var solutionsHandle = GCHandle.Alloc(device.Solutions[i], GCHandleType.Pinned);
                        device.Processor_Structs[i].Solutions = solutionsHandle.AddrOfPinnedObject();
                        device.AddHandle(solutionsHandle);
                    }

                    var processorsHandle = GCHandle.Alloc(device.Processor_Structs, GCHandleType.Pinned);
                    device.DeviceCPU_Struct.Processors = processorsHandle.AddrOfPinnedObject();
                    device.AddHandle(processorsHandle);

                    device.SolutionTemplate = Work.SolutionTemplate.ToArray();
                    var solutionTemplateHandle = GCHandle.Alloc(device.SolutionTemplate, GCHandleType.Pinned);
                    device.DeviceCPU_Struct.SolutionTemplate = solutionTemplateHandle.AddrOfPinnedObject();
                    device.AddHandle(solutionTemplateHandle);

                    device.DeviceCPU_Struct.Message = device.CommonPointers.Message;
                    device.DeviceCPU_Struct.MidState = device.CommonPointers.MidState;
                    device.DeviceCPU_Struct.High64Target = device.CommonPointers.High64Target;
                    device.DeviceCPU_Struct.Target = device.CommonPointers.Target;

                    device.IsInitialized = true;
                }
            }
        }

        protected override void PushHigh64Target(Device.DeviceBase device)
        {
            // Do nothing
        }

        protected override void PushTarget(Device.DeviceBase device)
        {
            // Do nothing
        }

        protected override void PushMidState(Device.DeviceBase device)
        {
            // Do nothing
        }

        protected override void PushMessage(Device.DeviceBase device)
        {
            // Do nothing
        }

        protected override void StartFinding(Device.DeviceBase device, bool isKingMaking)
        {
            var deviceCPU = (Device.CPU)device;

            if (!deviceCPU.IsInitialized) return;

            while (!deviceCPU.HasNewTarget || !deviceCPU.HasNewChallenge)
                Task.Delay(500).Wait();

            PrintMessage(device.Type, device.Platform, deviceCPU.DeviceID, "Info", "Start mining...");

            foreach (var processor in deviceCPU.Processor_Structs)
                Task.Factory.StartNew(() => StartThreadFinding(deviceCPU, processor, isKingMaking));
        }

        #endregion

        private void StartThreadFinding(Device.CPU device, Structs.Processor processor, bool isKingMaking)
        {
            try
            {
                var errorMessage = new StringBuilder(1024);
                var currentChallenge = (byte[])Array.CreateInstance(typeof(byte), UINT256_LENGTH);

                DateTime loopStartTime;
                int loopTimeElapsed;
                double timeAccuracy;
                var loopTimeTarget = (int)(m_hashPrintTimer.Interval * 0.1);

                var processorIndex = -1;
                for (var i = 0; i < device.Processor_Structs.Length; i++)
                    if (device.Processor_Structs[i].Affinity == processor.Affinity)
                        processorIndex = i;

                Helper.CPU.Solver.SetThreadAffinity(UnmanagedInstance, processor.Affinity, errorMessage);
                if (errorMessage.Length > 0)
                {
                    PrintMessage(device.Type, device.Platform, processorIndex, "Error", errorMessage.ToString());
                    return;
                }

                device.HashStartTime = DateTime.Now;
                device.HashCount = 0;
                device.IsMining = true;
                do
                {
                    while (device.IsPause)
                    {
                        Task.Delay(500).Wait();
                        device.HashStartTime = DateTime.Now;
                        device.HashCount = 0;
                    }

                    CheckInputs(device, isKingMaking, ref currentChallenge);

                    Work.IncrementPosition(ref processor.WorkPosition, processor.WorkSize);

                    lock (device)
                        device.HashCount += processor.WorkSize;

                    loopStartTime = DateTime.Now;

                    if (isKingMaking)
                        Helper.CPU.Solver.HashMessage(UnmanagedInstance, ref device.DeviceCPU_Struct, ref processor);
                    else
                        Helper.CPU.Solver.HashMidState(UnmanagedInstance, ref device.DeviceCPU_Struct, ref processor);

                    loopTimeElapsed = (int)(DateTime.Now - loopStartTime).TotalMilliseconds;

                    timeAccuracy = (float)loopTimeTarget / loopTimeElapsed;

                    if (timeAccuracy > 1.2f || timeAccuracy < 0.8f)
                        processor.WorkSize = (ulong)(timeAccuracy * processor.WorkSize);

                    if (processor.SolutionCount > 0)
                    {
                        SubmitSolutions(device.Solutions[processorIndex].ToArray(),
                                        currentChallenge,
                                        device.Type,
                                        device.Platform,
                                        device.DeviceID,
                                        processor.SolutionCount,
                                        isKingMaking);

                        processor.SolutionCount = 0;
                    }
                } while (device.IsMining);

                if (processor.Affinity == device.Processor_Structs.First().Affinity)
                    PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Stop mining...");

                device.HashCount = 0;
                device.IsStopped = true;
                device.IsInitialized = false;
            }
            catch (Exception ex)
            {
                PrintMessage(device.Type, device.Platform, -1, "Error", ex.Message);
            }
            if (processor.Affinity == device.Processor_Structs.First().Affinity)
                PrintMessage(device.Type, device.Platform, device.DeviceID, "Info", "Mining stopped.");
        }
    }
}
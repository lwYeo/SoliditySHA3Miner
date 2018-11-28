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

using SoliditySHA3Miner.Miner.Helper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace SoliditySHA3Miner.API
{
    public class Json : IDisposable
    {
        public bool IsSupported { get; }

        private Miner.IMiner[] m_miners;
        private HttpListener m_Listener;
        private bool m_isOngoing;

        public Json(params Miner.IMiner[] miners)
        {
            IsSupported = HttpListener.IsSupported;
            if (!IsSupported)
            {
                Program.Print("[ERROR] Obsolete OS detected, JSON-API will not start.");
                return;
            }
            m_miners = miners;
        }

        public void Start(string apiBind)
        {
            m_isOngoing = false;

            var httpBind = apiBind.ToString();
            if (string.IsNullOrWhiteSpace(httpBind))
            {
                Program.Print("[INFO] minerJsonAPI is null or empty, using default...");
                httpBind = Config.Defaults.JsonAPIPath;
            }
            else if (apiBind == "0")
            {
                Program.Print("[INFO] JSON-API is disabled.");
                return;
            }

            if (!httpBind.StartsWith("http://") || httpBind.StartsWith("https://")) httpBind = "http://" + httpBind;
            if (!httpBind.EndsWith("/")) httpBind += "/";

            if (!int.TryParse(httpBind.Split(':')[2].TrimEnd('/'), out int port))
            {
                Program.Print("[ERROR] Invalid port provided for JSON-API.");
                return;
            }

            var tempIPAddress = httpBind.Split(new string[] { "//" }, StringSplitOptions.None)[1].Split(':')[0];
            if (!IPAddress.TryParse(tempIPAddress, out IPAddress ipAddress))
            {
                Program.Print("[ERROR] Invalid IP address provided for JSON-API.");
                return;
            }

            using (var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
            {
                try { socket.Bind(new IPEndPoint(ipAddress, port)); }
                catch (Exception)
                {
                    Program.Print("[ERROR] JSON-API failed to bind to: " + (string.IsNullOrEmpty(apiBind) ? Config.Defaults.JsonAPIPath : apiBind));
                    return;
                }
            };

            try
            {
                m_Listener = new HttpListener();
                m_Listener.Prefixes.Add(httpBind);

                Process(m_Listener);
            }
            catch (Exception ex)
            {
                Program.Print("[ERROR] An error has occured while starting JSON-API: " + ex.Message);
                return;
            }
        }

        public void Stop()
        {
            if (!m_isOngoing) return;
            try
            {
                m_isOngoing = false;
                Program.Print("[INFO] JSON-API service stopping...");
                m_Listener.Stop();
            }
            catch (Exception ex)
            {
                Program.Print("[ERROR] An error has occured while stopping JSON-API: " + ex.Message);
            }
        }

        private async void Process(HttpListener listener)
        {
            listener.Start();
            Program.Print(string.Format("[INFO] JSON-API service started at {0}...", listener.Prefixes.ElementAt(0)));

            m_isOngoing = true;
            while (m_isOngoing)
            {
                HttpListenerContext context = null;

                try { context = await listener.GetContextAsync(); }
                catch (HttpListenerException ex)
                {
                    if (ex.ErrorCode == 995)
                        break;

                    Program.Print(string.Format("[ERROR] {0}: Error code: {1}, Message: {2}",
                                                ex.GetType().Name, ex.ErrorCode, ex.Message));
                    await Task.Delay(1000);
                    continue;
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Program.Print(string.Format("[ERROR] {0}", ex.Message));
                    await Task.Delay(1000);
                    continue;
                }

                HttpListenerResponse response = context.Response;
                if (response != null)
                {
                    response.AppendHeader("Pragma", "no-cache");
                    response.AppendHeader("Expires", "0");
                    response.ContentType = "application/json";
                    response.StatusCode = (int)HttpStatusCode.OK;

                    ProcessApiDataResponse(response);
                }
            }
        }

        private void ProcessApiDataResponse(HttpListenerResponse response)
        {
            Task.Factory.StartNew(() =>
            {
                try
                {
                    double divisor = 1;
                    var api = new JsonAPI();

                    PopulateCommonApiData(ref api, ref divisor);

                    foreach (var miner in m_miners)
                    {
                        foreach (var device in miner.Devices.Where(d => d.AllowDevice))
                        {
                            if (miner.HasMonitoringAPI)
                            {
                                switch (device.Type)
                                {
                                    case "CUDA":
                                        JsonAPI.CUDA_Miner cudaMiner = null;
                                        PopulateCudaApiData((Miner.CUDA)miner, (Miner.Device.CUDA)device, divisor, ref cudaMiner);
                                        if (cudaMiner != null) api.Miners.Add(cudaMiner);
                                        break;

                                    case "OpenCL":
                                        JsonAPI.AMD_Miner amdMiner = null;
                                        PopulateAmdApiData((Miner.OpenCL)miner, (Miner.Device.OpenCL)device, divisor, ref amdMiner);
                                        if (amdMiner != null) api.Miners.Add(amdMiner);
                                        break;
                                }
                            }
                            else
                            {
                                switch (device.Type)
                                {
                                    case "OpenCL":
                                        JsonAPI.OpenCLMiner openClMiner = null;
                                        PopulateOpenCLApiData((Miner.OpenCL)miner, (Miner.Device.OpenCL)device, divisor, ref openClMiner);
                                        if (openClMiner != null) api.Miners.Add(openClMiner);
                                        break;

                                    case "CPU":
                                    default:
                                        JsonAPI.Miner cpuMiner = null;
                                        PopulateCpuApiData((Miner.CPU)miner, (Miner.Device.CPU)device, divisor, ref cpuMiner);
                                        if (cpuMiner != null) api.Miners.Add(cpuMiner);
                                        break;
                                }
                            }
                        }
                    }
                    api.Miners.Sort((x, y) => x.PciBusID.CompareTo(y.PciBusID));

                    byte[] buffer = Encoding.UTF8.GetBytes(Utils.Json.SerializeFromObject(api, Utils.Json.BaseClassFirstSettings));

                    using (var output = response.OutputStream)
                    {
                        if (buffer != null)
                        {
                            output.Write(buffer, 0, buffer.Length);
                            output.Flush();
                        }
                    }
                }
                catch (Exception ex)
                {
                    response.StatusCode = (int)HttpStatusCode.InternalServerError;

                    var errorMessage = string.Empty;
                    var currentEx = ex;

                    while (currentEx != null)
                    {
                        if (!string.IsNullOrEmpty(errorMessage)) errorMessage += " ";
                        errorMessage += currentEx.Message;
                        currentEx = currentEx.InnerException;
                    }
                    Program.Print(string.Format("[ERROR] {0}", errorMessage));
                }
                finally
                {
                    try
                    {
                        if (response != null) response.Close();
                    }
                    catch (Exception ex)
                    {
                        var errorMessage = string.Empty;
                        var currentEx = ex;

                        while (currentEx != null)
                        {
                            if (!string.IsNullOrEmpty(errorMessage)) errorMessage += " ";
                            errorMessage += currentEx.Message;
                            currentEx = currentEx.InnerException;
                        }
                        Program.Print(string.Format("[ERROR] {0}", errorMessage));
                    }
                }
            },
            TaskCreationOptions.LongRunning);
        }

        private void GetHashRateUnit(ulong hashrate, ref double divisor, ref string unit)
        {
            var sHashrate = hashrate.ToString();
            if (sHashrate.Length > 12 + 1)
            {
                divisor = 1000000000000;
                unit = "TH/s";
            }
            else if (sHashrate.Length > 9 + 1)
            {
                divisor = 1000000000;
                unit = "GH/s";
            }
            else if (sHashrate.Length > 6 + 1)
            {
                divisor = 1000000;
                unit = "MH/s";
            }
            else if (sHashrate.Length > 3 + 1)
            {
                divisor = 1000;
                unit = "KH/s";
            }
        }

        private void PopulateCommonApiData(ref JsonAPI api, ref double divisor)
        {
            try
            {
                var networkInterface = m_miners.Select(m => m.NetworkInterface).FirstOrDefault(m => m != null);

                var totalHashRate = 0ul;
                var hashRateUnit = string.Empty;

                foreach (var miner in m_miners)
                    totalHashRate += miner.GetTotalHashrate();

                if (totalHashRate > 0)
                    GetHashRateUnit(totalHashRate, ref divisor, ref hashRateUnit);
                else
                    GetHashRateUnit((networkInterface?.GetEffectiveHashrate() ?? 0), ref divisor, ref hashRateUnit);

                var timeLeftToSolveBlock = networkInterface?.GetTimeLeftToSolveBlock(totalHashRate) ?? TimeSpan.Zero;

                if (timeLeftToSolveBlock != TimeSpan.Zero)
                    api.EstimateTimeLeftToSolveBlock = (long)timeLeftToSolveBlock.TotalSeconds;

                api.EffectiveHashRate = (float)((networkInterface?.GetEffectiveHashrate() ?? 0) / divisor);

                api.TotalHashRate = (float)(totalHashRate / divisor);

                api.HashRateUnit = hashRateUnit;

                api.MinerAddress = networkInterface.MinerAddress ?? string.Empty;

                api.MiningURL = networkInterface.SubmitURL ?? string.Empty;

                if (networkInterface.CurrentChallenge != null)
                    api.CurrentChallenge = Utils.Numerics.Byte32ArrayToHexString(networkInterface.CurrentChallenge);

                api.CurrentDifficulty = networkInterface.Difficulty.Value;

                api.LastSubmitLatencyMS = networkInterface?.LastSubmitLatency ?? -1;

                api.LatencyMS = networkInterface?.Latency ?? -1;

                api.Uptime = (long)(DateTime.Now - Program.LaunchTime).TotalSeconds;

                api.RejectedShares = m_miners.Select(m => m.NetworkInterface).Distinct().Sum(i => (long)(i.RejectedShares));

                api.AcceptedShares = m_miners.Select(m => m.NetworkInterface).Distinct().Sum(i => (long)(i.SubmittedShares)) - api.RejectedShares;
            }
            catch (Exception ex)
            {
                var errorMessage = string.Empty;
                var currentEx = ex;

                while (currentEx != null)
                {
                    if (!string.IsNullOrEmpty(errorMessage)) errorMessage += " ";
                    errorMessage += currentEx.Message;
                    currentEx = currentEx.InnerException;
                }
                Program.Print(string.Format("[ERROR] {0}", errorMessage));
            }
        }

        private void PopulateCudaApiData(Miner.CUDA miner, Miner.Device.CUDA device, double divisor, ref JsonAPI.CUDA_Miner cudaMiner)
        {
            try
            {
                var instancePtr = miner.UnmanagedInstance;

                cudaMiner = new JsonAPI.CUDA_Miner()
                {
                    Type = device.Type,
                    DeviceID = device.DeviceID,
                    PciBusID = device.PciBusID,
                    ModelName = device.Name,
                    HashRate = (float)(miner.GetHashRateByDevice(device) / divisor),
                    HasMonitoringAPI = miner.HasMonitoringAPI,
                    SettingIntensity = device.Intensity
                };

                if (miner.UseNvSMI)
                {
                    cudaMiner.SettingMaxCoreClockMHz = -1;

                    cudaMiner.SettingMaxMemoryClockMHz = -1;

                    cudaMiner.SettingPowerLimitPercent = Miner.API.NvSMI.GetDeviceSettingPowerLimit(device.PciBusID);

                    cudaMiner.SettingThermalLimitC = Miner.API.NvSMI.GetDeviceSettingThermalLimit(device.PciBusID);

                    cudaMiner.SettingFanLevelPercent = Miner.API.NvSMI.GetDeviceSettingFanLevelPercent(device.PciBusID);

                    cudaMiner.CurrentFanTachometerRPM = -1;

                    cudaMiner.CurrentTemperatureC = Miner.API.NvSMI.GetDeviceCurrentTemperature(device.PciBusID);

                    cudaMiner.CurrentCoreClockMHz = Miner.API.NvSMI.GetDeviceCurrentCoreClock(device.PciBusID);

                    cudaMiner.CurrentMemoryClockMHz = Miner.API.NvSMI.GetDeviceCurrentMemoryClock(device.PciBusID);

                    cudaMiner.CurrentUtilizationPercent = Miner.API.NvSMI.GetDeviceCurrentUtilizationPercent(device.PciBusID);

                    cudaMiner.CurrentPState = Miner.API.NvSMI.GetDeviceCurrentPstate(device.PciBusID);

                    cudaMiner.CurrentThrottleReasons = Miner.API.NvSMI.GetDeviceCurrentThrottleReasons(device.PciBusID);
                }
                else
                {
                    var tempValue = 0;
                    var tempStr = new StringBuilder(1024);
                    var cudaDevice = device.DeviceCUDA_Struct;

                    CUDA.Solver.GetDeviceSettingMaxCoreClock(cudaDevice, ref tempValue);
                    cudaMiner.SettingMaxCoreClockMHz = tempValue;

                    CUDA.Solver.GetDeviceSettingMaxMemoryClock(cudaDevice, ref tempValue);
                    cudaMiner.SettingMaxMemoryClockMHz = tempValue;

                    CUDA.Solver.GetDeviceSettingPowerLimit(cudaDevice, ref tempValue);
                    cudaMiner.SettingPowerLimitPercent = tempValue;

                    CUDA.Solver.GetDeviceSettingThermalLimit(cudaDevice, ref tempValue);
                    cudaMiner.SettingThermalLimitC = tempValue;

                    CUDA.Solver.GetDeviceSettingFanLevelPercent(cudaDevice, ref tempValue);
                    cudaMiner.SettingFanLevelPercent = tempValue;

                    CUDA.Solver.GetDeviceCurrentFanTachometerRPM(cudaDevice, ref tempValue);
                    cudaMiner.CurrentFanTachometerRPM = tempValue;

                    CUDA.Solver.GetDeviceCurrentTemperature(cudaDevice, ref tempValue);
                    cudaMiner.CurrentTemperatureC = tempValue;

                    CUDA.Solver.GetDeviceCurrentCoreClock(cudaDevice, ref tempValue);
                    cudaMiner.CurrentCoreClockMHz = tempValue;

                    CUDA.Solver.GetDeviceCurrentMemoryClock(cudaDevice, ref tempValue);
                    cudaMiner.CurrentMemoryClockMHz = tempValue;

                    CUDA.Solver.GetDeviceCurrentUtilizationPercent(cudaDevice, ref tempValue);
                    cudaMiner.CurrentUtilizationPercent = tempValue;

                    CUDA.Solver.GetDeviceCurrentPstate(cudaDevice, ref tempValue);
                    cudaMiner.CurrentPState = tempValue;

                    CUDA.Solver.GetDeviceCurrentThrottleReasons(cudaDevice, tempStr);
                    cudaMiner.CurrentThrottleReasons = tempStr.ToString();
                }
            }
            catch (Exception ex)
            {
                var errorMessage = string.Empty;
                var currentEx = ex;

                while (currentEx != null)
                {
                    if (!string.IsNullOrEmpty(errorMessage)) errorMessage += " ";
                    errorMessage += currentEx.Message;
                    currentEx = currentEx.InnerException;
                }
                Program.Print(string.Format("[ERROR] {0}", errorMessage));
            }
        }

        private void PopulateAmdApiData(Miner.OpenCL miner, Miner.Device.OpenCL device, double divisor, ref JsonAPI.AMD_Miner amdMiner)
        {
            try
            {
                var tempValue = 0;
                var errorMessage = new StringBuilder(1024);
                var instancePtr = miner.UnmanagedInstance;

                amdMiner = new JsonAPI.AMD_Miner()
                {
                    Type = device.Type,
                    DeviceID = device.DeviceID,
                    PciBusID = device.PciBusID,
                    ModelName = device.Name,
                    HashRate = (float)(miner.GetHashRateByDevice(device) / divisor),
                    HasMonitoringAPI = miner.HasMonitoringAPI,
                    Platform = device.Platform,
                    SettingIntensity = device.Intensity
                };

                if (miner.UseLinuxQuery)
                {
                    amdMiner.SettingMaxCoreClockMHz = -1;

                    amdMiner.SettingMaxMemoryClockMHz = -1;

                    amdMiner.SettingPowerLimitPercent = -1;

                    amdMiner.SettingThermalLimitC = int.MinValue;

                    amdMiner.SettingFanLevelPercent = Miner.API.AmdLinuxQuery.GetDeviceSettingFanLevelPercent(device.PciBusID);

                    amdMiner.CurrentFanTachometerRPM = Miner.API.AmdLinuxQuery.GetDeviceCurrentFanTachometerRPM(device.PciBusID);

                    amdMiner.CurrentTemperatureC = Miner.API.AmdLinuxQuery.GetDeviceCurrentTemperature(device.PciBusID);

                    amdMiner.CurrentCoreClockMHz = Miner.API.AmdLinuxQuery.GetDeviceCurrentCoreClock(device.PciBusID);

                    amdMiner.CurrentMemoryClockMHz = Miner.API.AmdLinuxQuery.GetDeviceCurrentCoreClock(device.PciBusID);

                    amdMiner.CurrentUtilizationPercent = Miner.API.AmdLinuxQuery.GetDeviceCurrentUtilizationPercent(device.PciBusID);
                }
                else
                {
                    var deviceCL = device.DeviceCL_Struct;

                    OpenCL.Solver.GetDeviceSettingMaxCoreClock(deviceCL, ref tempValue, errorMessage);
                    amdMiner.SettingMaxCoreClockMHz = tempValue;

                    OpenCL.Solver.GetDeviceSettingMaxMemoryClock(deviceCL, ref tempValue, errorMessage);
                    amdMiner.SettingMaxMemoryClockMHz = tempValue;

                    OpenCL.Solver.GetDeviceSettingPowerLimit(deviceCL, ref tempValue, errorMessage);
                    amdMiner.SettingPowerLimitPercent = tempValue;

                    OpenCL.Solver.GetDeviceSettingThermalLimit(deviceCL, ref tempValue, errorMessage);
                    amdMiner.SettingThermalLimitC = tempValue;

                    OpenCL.Solver.GetDeviceSettingFanLevelPercent(deviceCL, ref tempValue, errorMessage);
                    amdMiner.SettingFanLevelPercent = tempValue;

                    OpenCL.Solver.GetDeviceCurrentFanTachometerRPM(deviceCL, ref tempValue, errorMessage);
                    amdMiner.CurrentFanTachometerRPM = tempValue;

                    OpenCL.Solver.GetDeviceCurrentTemperature(deviceCL, ref tempValue, errorMessage);
                    amdMiner.CurrentTemperatureC = tempValue;

                    OpenCL.Solver.GetDeviceCurrentCoreClock(deviceCL, ref tempValue, errorMessage);
                    amdMiner.CurrentCoreClockMHz = tempValue;

                    OpenCL.Solver.GetDeviceCurrentMemoryClock(deviceCL, ref tempValue, errorMessage);
                    amdMiner.CurrentMemoryClockMHz = tempValue;

                    OpenCL.Solver.GetDeviceCurrentUtilizationPercent(deviceCL, ref tempValue, errorMessage);
                    amdMiner.CurrentUtilizationPercent = tempValue;
                }
            }
            catch (Exception ex)
            {
                var errorMessage = string.Empty;
                var currentEx = ex;

                while (currentEx != null)
                {
                    if (!string.IsNullOrEmpty(errorMessage)) errorMessage += " ";
                    errorMessage += currentEx.Message;
                    currentEx = currentEx.InnerException;
                }
                Program.Print(string.Format("[ERROR] {0}", errorMessage));
            }
        }

        private void PopulateOpenCLApiData(Miner.OpenCL miner, Miner.Device.OpenCL device, double divisor, ref JsonAPI.OpenCLMiner openCLMiner)
        {
            try
            {
                openCLMiner = new JsonAPI.OpenCLMiner()
                {
                    Type = device.Type,
                    DeviceID = device.DeviceID,
                    ModelName = device.Name,
                    HashRate = (float)(miner.GetHashRateByDevice(device) / divisor),
                    HasMonitoringAPI = miner.HasMonitoringAPI,

                    Platform = device.Platform,
                    SettingIntensity = device.Intensity
                };
            }
            catch (Exception ex)
            {
                var errorMessage = string.Empty;
                var currentEx = ex;

                while (currentEx != null)
                {
                    if (!string.IsNullOrEmpty(errorMessage)) errorMessage += " ";
                    errorMessage += currentEx.Message;
                    currentEx = currentEx.InnerException;
                }
                Program.Print(string.Format("[ERROR] {0}", errorMessage));
            }
        }

        private void PopulateCpuApiData(Miner.CPU miner, Miner.Device.CPU device, double divisor, ref JsonAPI.Miner cpuMiner)
        {
            try
            {
                cpuMiner = new JsonAPI.Miner()
                {
                    Type = device.Type,
                    DeviceID = device.DeviceID,
                    ModelName = device.Name,
                    HashRate = (float)(miner.GetHashRateByDevice(device) / divisor),
                    HasMonitoringAPI = miner.HasMonitoringAPI
                };
            }
            catch (Exception ex)
            {
                var errorMessage = string.Empty;
                var currentEx = ex;

                while (currentEx != null)
                {
                    if (!string.IsNullOrEmpty(errorMessage)) errorMessage += " ";
                    errorMessage += currentEx.Message;
                    currentEx = currentEx.InnerException;
                }
                Program.Print(string.Format("[ERROR] {0}", errorMessage));
            }
        }

        public void Dispose()
        {
            m_isOngoing = false;
            if (m_Listener != null) m_Listener.Close();
        }

        public class JsonAPI
        {
            public DateTime SystemDateTime => DateTime.Now;
            public string MinerAddress { get; set; }
            public string MiningURL { get; set; }
            public string CurrentChallenge { get; set; }
            public BigInteger CurrentDifficulty { get; set; }
            public long EstimateTimeLeftToSolveBlock { get; set; }
            public float EffectiveHashRate { get; set; }
            public float TotalHashRate { get; set; }
            public string HashRateUnit { get; set; }
            public int LastSubmitLatencyMS { get; set; }
            public int LatencyMS { get; set; }
            public long Uptime { get; set; }
            public long AcceptedShares { get; set; }
            public long RejectedShares { get; set; }
            public List<Miner> Miners { get; set; }

            public JsonAPI() => Miners = new List<Miner>();

            public class Miner
            {
                public string Type { get; set; }
                public int DeviceID { get; set; }
                public uint PciBusID { get; set; }
                public string ModelName { get; set; }
                public float HashRate { get; set; }
                public bool HasMonitoringAPI { get; set; }
            }

            public class CUDA_Miner : Miner
            {
                public float SettingIntensity { get; set; }
                public int SettingMaxCoreClockMHz { get; set; }
                public int SettingMaxMemoryClockMHz { get; set; }
                public int SettingPowerLimitPercent { get; set; }
                public int SettingThermalLimitC { get; set; }
                public int SettingFanLevelPercent { get; set; }

                public int CurrentFanTachometerRPM { get; set; }
                public int CurrentTemperatureC { get; set; }
                public int CurrentCoreClockMHz { get; set; }
                public int CurrentMemoryClockMHz { get; set; }
                public int CurrentUtilizationPercent { get; set; }
                public int CurrentPState { get; set; }
                public string CurrentThrottleReasons { get; set; }
            }

            public class OpenCLMiner : Miner
            {
                public string Platform { get; set; }
                public float SettingIntensity { get; set; }
            }

            public class AMD_Miner : OpenCLMiner
            {
                public int SettingMaxCoreClockMHz { get; set; }
                public int SettingMaxMemoryClockMHz { get; set; }
                public int SettingPowerLimitPercent { get; set; }
                public int SettingThermalLimitC { get; set; }
                public int SettingFanLevelPercent { get; set; }

                public int CurrentFanTachometerRPM { get; set; }
                public int CurrentTemperatureC { get; set; }
                public int CurrentCoreClockMHz { get; set; }
                public int CurrentMemoryClockMHz { get; set; }
                public int CurrentUtilizationPercent { get; set; }
            }
        }
    }
}
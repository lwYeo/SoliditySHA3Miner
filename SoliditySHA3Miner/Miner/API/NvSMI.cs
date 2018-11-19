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
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Xml.Linq;

namespace SoliditySHA3Miner.Miner.API
{
    public static class NvSMI
    {
        public static string NvSMI_PATH = string.Empty;

        public static bool FoundNvSMI()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (File.Exists(@"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"))
                {
                    NvSMI_PATH = @"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe";
                    return true;
                }
            }
            else
            {
                var values = Environment.GetEnvironmentVariable("PATH");
                foreach (var path in values.Split(':'))
                {
                    var fullPath = Path.Combine(path, "nvidia-smi");
                    if (File.Exists(fullPath))
                    {
                        NvSMI_PATH = fullPath;
                        return true;
                    }
                }
            }
            return false;
        }

        private static string GetProcessOutput()
        {
            var smiProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = NvSMI_PATH,
                    Arguments = "-q -x",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };
            smiProcess.Start();
            smiProcess.WaitForExit();

            var output = smiProcess.StandardOutput.ReadToEnd();
            smiProcess.Dispose();
            return output;
        }

        private static XElement GetDevice(uint pciBusID)
        {
            var smiOutput = GetProcessOutput();
            if (string.IsNullOrWhiteSpace(smiOutput)) return null;

            var smiXML = XDocument.Parse(smiOutput);
            var smiDevice = smiXML.Descendants("pci_bus").
                                    Where(d => Convert.ToUInt32(d.Value, 16) == pciBusID).
                                    Select(d => d.Parent.Parent).
                                    FirstOrDefault();
            return smiDevice;
        }

        // The following methods are not available/inaccurate in NvSMI
        //GetDeviceSettingMaxCoreClock
        //GetDeviceSettingMaxMemoryClock
        //GetDeviceCurrentFanTachometerRPM

        public static int GetDeviceSettingPowerLimit(uint pciBusID)
        {
            var smiDevice = GetDevice(pciBusID);
            if (smiDevice == null) return -1;

            var powLimitStr = smiDevice.Descendants("power_limit").FirstOrDefault()?.Value;
            if (string.IsNullOrWhiteSpace(powLimitStr)) return -1;

            var defPowLimitStr = smiDevice.Descendants("default_power_limit").FirstOrDefault()?.Value;
            if (string.IsNullOrWhiteSpace(defPowLimitStr)) return -1;

            if (float.TryParse(powLimitStr.Replace(" W", string.Empty), NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out float powLimit))
            {
                if (float.TryParse(defPowLimitStr.Replace(" W", string.Empty), NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out float defPowLimit))
                {
                    return (int)(powLimit / defPowLimit * 100);
                }
            }
            
            return -1;
        }

        public static int GetDeviceSettingThermalLimit(uint pciBusID)
        {
            var smiDevice = GetDevice(pciBusID);
            if (smiDevice == null) return -1;

            var temperature = smiDevice.Descendants("gpu_temp_slow_threshold").FirstOrDefault()?.Value;
            if (string.IsNullOrWhiteSpace(temperature)) return -1;

            if (int.TryParse(temperature.Replace(" C", string.Empty).Replace(" F", string.Empty), NumberStyles.Number, CultureInfo.InvariantCulture, out int tempValue))
                return tempValue;
            else
                return int.MinValue;
        }
        
        public static int GetDeviceSettingFanLevelPercent(uint pciBusID)
        {
            var smiDevice = GetDevice(pciBusID);
            if (smiDevice == null) return -1;

            var levelStr = smiDevice.Descendants("fan_speed").FirstOrDefault()?.Value;
            if (string.IsNullOrWhiteSpace(levelStr)) return -1;

            if (int.TryParse(levelStr.Replace(" %", string.Empty), NumberStyles.Number, CultureInfo.InvariantCulture, out int fanLevel))
                return fanLevel;
            else
                return -1;
        }

        public static int GetDeviceCurrentTemperature(uint pciBusID)
        {
            var smiDevice = GetDevice(pciBusID);
            if (smiDevice == null) return -1;

            var temperature = smiDevice.Descendants("gpu_temp").FirstOrDefault()?.Value;
            if (string.IsNullOrWhiteSpace(temperature)) return -1;

            if (int.TryParse(temperature.Replace(" C", string.Empty).Replace(" F", string.Empty), NumberStyles.Number, CultureInfo.InvariantCulture, out int tempValue))
                return tempValue;
            else
                return int.MinValue;
        }

        public static int GetDeviceCurrentCoreClock(uint pciBusID)
        {
            var smiDevice = GetDevice(pciBusID);
            if (smiDevice == null) return -1;

            var clockStr = smiDevice.Descendants("graphics_clock").FirstOrDefault()?.Value;
            if (string.IsNullOrWhiteSpace(clockStr)) return -1;

            if (int.TryParse(clockStr.Replace(" GHz", string.Empty).Replace(" MHz", string.Empty), NumberStyles.Number, CultureInfo.InvariantCulture, out int coreClock))
                return coreClock;
            else
                return -1;
        }

        public static int GetDeviceCurrentMemoryClock(uint pciBusID)
        {
            var smiDevice = GetDevice(pciBusID);
            if (smiDevice == null) return -1;

            var clockStr = smiDevice.Descendants("mem_clock").FirstOrDefault()?.Value;
            if (string.IsNullOrWhiteSpace(clockStr)) return -1;

            if (int.TryParse(clockStr.Replace(" GHz", string.Empty).Replace(" MHz", string.Empty), NumberStyles.Number, CultureInfo.InvariantCulture, out int memClock))
                return memClock;
            else
                return -1;
        }

        public static int GetDeviceCurrentUtilizationPercent(uint pciBusID)
        {
            var smiDevice = GetDevice(pciBusID);
            if (smiDevice == null) return -1;

            var utilStr = smiDevice.Descendants("gpu_util").FirstOrDefault()?.Value;
            if (string.IsNullOrWhiteSpace(utilStr)) return -1;

            if (int.TryParse(utilStr.Replace(" %", string.Empty), NumberStyles.Number, CultureInfo.InvariantCulture, out int utilization))
                return utilization;
            else
                return -1;
        }

        public static int GetDeviceCurrentPstate(uint pciBusID)
        {
            var smiDevice = GetDevice(pciBusID);
            if (smiDevice == null) return -1; 

            var pStateStr = smiDevice.Descendants("power_state").FirstOrDefault()?.Value;
            
            if (string.IsNullOrWhiteSpace(pStateStr)) return -1;

            if (int.TryParse(string.Concat(pStateStr.Skip(1).Take(1)), NumberStyles.Number, CultureInfo.InvariantCulture, out int powerState))
                return powerState;
            else
                return -1;
        }

        public static string GetDeviceCurrentThrottleReasons(uint pciBusID)
        {
            var smiDevice = GetDevice(pciBusID);
            if (smiDevice == null) return string.Empty;

            var reasons = string.Empty;

            var gpuIdle = smiDevice.Descendants("clocks_throttle_reason_gpu_idle").FirstOrDefault()?.Value;
            var appClockSetting = smiDevice.Descendants("clocks_throttle_reason_applications_clocks_setting").FirstOrDefault()?.Value;
            var hwPowerBrakeSlowdown = smiDevice.Descendants("clocks_throttle_reason_hw_power_brake_slowdown").FirstOrDefault()?.Value;
            var swPowerCap = smiDevice.Descendants("clocks_throttle_reason_sw_power_cap").FirstOrDefault()?.Value;
            var hwThermalSlowdown = smiDevice.Descendants("clocks_throttle_reason_hw_thermal_slowdown").FirstOrDefault()?.Value;
            var swThermalSlowdown = smiDevice.Descendants("clocks_throttle_reason_sw_thermal_slowdown").FirstOrDefault()?.Value;
            var sync_boost = smiDevice.Descendants("clocks_throttle_reason_sync_boost").FirstOrDefault()?.Value;

            if ((gpuIdle ?? string.Empty) == "Active")
                reasons += string.IsNullOrEmpty(reasons) ? "Idle" : ", Idle";

            if ((appClockSetting ?? string.Empty) == "Active")
                reasons += string.IsNullOrEmpty(reasons) ? "Applications Clocks Setting" : ", Applications Clocks Setting";

            if ((hwPowerBrakeSlowdown ?? string.Empty) == "Active")
                reasons += string.IsNullOrEmpty(reasons) ? "HW Power Brake Slowdown" : ", HW Power Brake Slowdown";

            if ((swPowerCap ?? string.Empty) == "Active")
                reasons += string.IsNullOrEmpty(reasons) ? "SW Power Cap" : ", SW Power Cap";

            if ((hwThermalSlowdown ?? string.Empty) == "Active")
                reasons += string.IsNullOrEmpty(reasons) ? "HW Thermal Slowdown" : ", HW Thermal Slowdown";

            if ((swThermalSlowdown ?? string.Empty) == "Active")
                reasons += string.IsNullOrEmpty(reasons) ? "SW Thermal Slowdown" : ", SW Thermal Slowdown";

            if ((sync_boost ?? string.Empty) == "Active")
                reasons += string.IsNullOrEmpty(reasons) ? "Sync Boost" : ", Sync Boost";

            return reasons;
        }
    }
}

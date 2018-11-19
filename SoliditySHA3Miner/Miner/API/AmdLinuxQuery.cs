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
using System.Text.RegularExpressions;

namespace SoliditySHA3Miner.Miner.API
{
    public static class AmdLinuxQuery
    {
        private static object m_queryLock = new object();
        private static Regex m_deviceNameQueryRegex = new Regex(@"\[([^\[\]\/]+)\]");

        private static string GetLsPciQuery()
        {
            lock(m_queryLock)
            {
                var queryProcess = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "lspci",
                        Arguments = "-k",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };
                queryProcess.Start();
                queryProcess.WaitForExit();

                var query = queryProcess.StandardOutput.ReadToEnd();
                queryProcess.Dispose();

                return query ?? string.Empty;
            }
        }

        private static DirectoryInfo GetQueryDir(uint pciBusID, out int deviceEnum)
        {
            lock(m_queryLock)
            {
                deviceEnum = -1;
                var queryDir = new DirectoryInfo(@"/sys/kernel/debug/dri/");
                if (!queryDir.Exists) return null;

                queryDir = queryDir.GetDirectories().FirstOrDefault(d =>
                {
                    var devFullInfo = d.GetFiles("name").Select(sd => File.ReadAllText(sd.FullName).Replace(Environment.NewLine, " ")).FirstOrDefault();
                    if (string.IsNullOrWhiteSpace(devFullInfo)) return false;

                    var devInfo = devFullInfo.Split(' ').Where(i => i.StartsWith("dev=")).FirstOrDefault();
                    if (string.IsNullOrWhiteSpace(devInfo) || devInfo.Count(i => i.Equals(':')) < 2) return false;

                    var devPciBusID = devInfo.Split(':')[1];
                    
                    if (!UInt32.TryParse(devPciBusID, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out uint value))
                        return false;

                    if (!value.Equals(pciBusID)) return false;

                    return d.GetFiles("amdgpu_pm_info").Any();
                });

                if (queryDir != null)
                    if (int.TryParse(queryDir.Name, NumberStyles.None, CultureInfo.InvariantCulture, out int devEnum))
                        deviceEnum = devEnum;

                return queryDir;                
            }
        }

        private static FileInfo QueryAmdgpuPmInfo(uint pciBusID, out int deviceEnum)
        {
            var queryDir = GetQueryDir(pciBusID, out int devEnum);
            deviceEnum = devEnum;
            
            if (queryDir == null) return null;

            lock(m_queryLock)
            {
                return queryDir.GetFiles("amdgpu_pm_info").FirstOrDefault();   
            }
        }

        public static bool QuerySuccess()
        {
            var query = GetLsPciQuery();
            if (string.IsNullOrWhiteSpace(query)) return false;

            lock(m_queryLock)
            {
                var queryDir = new DirectoryInfo(@"/sys/kernel/debug/dri/");
                if (!queryDir.Exists) return false;

                return queryDir.GetDirectories().Where(d => d.GetFiles("amdgpu_pm_info").Any()).Any();
            }
        }

        public static string GetDeviceRealName(uint pciBusID, string defaultName)
        {
            var query = GetLsPciQuery();
            if (string.IsNullOrWhiteSpace(query)) return defaultName;

            var outputArr = query.Split(new string[] { Environment.NewLine }, StringSplitOptions.None).
                                  Where(l => !string.IsNullOrEmpty(l) && !l.StartsWith(" ") 
                                              && l.Contains("VGA") && l.Contains("Advanced Micro Devices") && l.Contains("[AMD/ATI]")).
                                  ToArray();

            var foundDeviceLine = outputArr.FirstOrDefault(l =>
            {
                if (UInt32.TryParse(l.Split(':')[0], NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out uint value))
                    return value.Equals(pciBusID);
                else
                    return false;
            });
            
            var match = m_deviceNameQueryRegex.Match(foundDeviceLine ?? string.Empty);

            return match.Success
                    ? match.Groups[1].Value
                    : defaultName;
        }

        public static int GetDeviceCurrentCoreClock(uint pciBusID)
        {
            var queryFile = QueryAmdgpuPmInfo(pciBusID, out int deviceEnum);
            if (queryFile == null) return -1;
            
            var query = File.ReadAllLines(queryFile.FullName).FirstOrDefault(l => l.TrimEnd().EndsWith("(SCLK)"));
            if (string.IsNullOrWhiteSpace(query)) return -1;

            var value = (query.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries)[0] ?? string.Empty).Trim();

            if (int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int iValue))
                return iValue;
            else
                return -1;
        }

        public static int GetDeviceCurrentMemoryClock(uint pciBusID)
        {
            var queryFile = QueryAmdgpuPmInfo(pciBusID, out int deviceEnum);
            if (queryFile == null) return -1;
            
            var query = File.ReadAllLines(queryFile.FullName).FirstOrDefault(l => l.TrimEnd().EndsWith("(MCLK)"));
            if (string.IsNullOrWhiteSpace(query)) return -1;

            var value = (query.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries)[0] ?? string.Empty).Trim();

            if (int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int iValue))
                return iValue;
            else
                return -1;
        }

        public static int GetDeviceCurrentTemperature(uint pciBusID)
        {
            var queryFile = QueryAmdgpuPmInfo(pciBusID, out int deviceEnum);
            if (queryFile == null) return int.MinValue;
            
            var query = File.ReadAllLines(queryFile.FullName).FirstOrDefault(l => l.TrimStart().StartsWith("GPU Temperature:"));
            if (string.IsNullOrWhiteSpace(query)) return int.MinValue;

            var value = (query.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries)[2] ?? string.Empty).Trim();

            if (int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int iValue))
                return iValue;
            else
                return int.MinValue;
        }

        public static int GetDeviceCurrentUtilizationPercent(uint pciBusID)
        {
            var queryFile = QueryAmdgpuPmInfo(pciBusID, out int deviceEnum);
            if (queryFile == null) return -1;
            
            var query = File.ReadAllLines(queryFile.FullName).FirstOrDefault(l => l.TrimStart().StartsWith("GPU Load:"));
            if (string.IsNullOrWhiteSpace(query)) return -1;

            var value = (query.Split(new string[] { " " }, StringSplitOptions.RemoveEmptyEntries)[2] ?? string.Empty).Trim();

            if (int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int iValue))
                return iValue;
            else
                return -1;
        }

        public static int GetDeviceCurrentFanTachometerRPM(uint pciBusID)
        {
            var queryDir = GetQueryDir(pciBusID, out int deviceEnum);
            if (queryDir == null) return -1;

            lock(m_queryLock)
            {
                queryDir = new DirectoryInfo(@"/sys/class/drm/card" + deviceEnum.ToString() + "/device/hwmon");
                if (!queryDir.Exists) return -1;
                
                var queryFile = queryDir.GetFiles("fan1_input").FirstOrDefault()
                            ?? queryDir.GetDirectories().SelectMany(d => d.GetFiles("fan1_input")).FirstOrDefault();

                if (queryFile == null) return -1;

                var query = (File.ReadAllText(queryFile.FullName) ?? string.Empty).Trim();

                if (int.TryParse(query, NumberStyles.None, CultureInfo.InvariantCulture, out int value))
                    return value;
                else
                    return -1;
            }
         }

        public static int GetDeviceSettingFanLevelPercent(uint pciBusID)
        {
            var queryDir = GetQueryDir(pciBusID, out int deviceEnum);
            if (queryDir == null) return -1;

            lock(m_queryLock)
            {
                queryDir = new DirectoryInfo(@"/sys/class/drm/card" + deviceEnum.ToString() + "/device/hwmon");
                if (!queryDir.Exists) return -1;
                
                var queryFile = queryDir.GetFiles("pwm1_max").FirstOrDefault()
                            ?? queryDir.GetDirectories().SelectMany(d => d.GetFiles("pwm1_max")).FirstOrDefault();

                if (queryFile == null) return -1;

                var query = (File.ReadAllText(queryFile.FullName) ?? string.Empty).Trim();

                if (!int.TryParse(query, NumberStyles.None, CultureInfo.InvariantCulture, out int maxPWM))
                    return -1;

                if (maxPWM < 1) return -1;

                queryFile = queryDir.GetFiles("pwm1").FirstOrDefault()
                        ?? queryDir.GetDirectories().SelectMany(d => d.GetFiles("pwm1")).FirstOrDefault();

                if (queryFile == null) return -1;

                query = (File.ReadAllText(queryFile.FullName) ?? string.Empty).Trim();

                if (!int.TryParse(query, NumberStyles.None, CultureInfo.InvariantCulture, out int pwm))
                    return -1;
                    
                return (int)(100 * pwm / maxPWM);
            }
        }
    }
}

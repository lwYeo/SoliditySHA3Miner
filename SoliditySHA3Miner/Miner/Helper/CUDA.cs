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
using System.Runtime.InteropServices;
using System.Text;

namespace SoliditySHA3Miner.Miner.Helper
{
    public static class CUDA
    {
        #region P/Invoke interface

        public static class Solver
        {
            public const string SOLVER_NAME = "CudaSoliditySHA3Solver";

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void FoundNvAPI64(ref bool hasNvAPI64);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCount(ref int deviceCount, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceName(int deviceID, StringBuilder deviceName, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern IntPtr GetInstance();

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void DisposeInstance(IntPtr instance);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceProperties(IntPtr instance, ref Structs.DeviceCUDA device, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void InitializeDevice(IntPtr instance, ref Structs.DeviceCUDA device, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void SetDevice(IntPtr instance, int deviceID, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void ResetDevice(IntPtr instance, int deviceID, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void ReleaseDeviceObjects(IntPtr instance, ref Structs.DeviceCUDA device, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PushHigh64Target(IntPtr instance, IntPtr high64Target, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PushMidState(IntPtr instance, IntPtr midState, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PushTarget(IntPtr instance, IntPtr target, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PushMessage(IntPtr instance, IntPtr message, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void HashMidState(IntPtr instance, ref Structs.DeviceCUDA device, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void HashMessage(IntPtr instance, ref Structs.DeviceCUDA device, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingMaxCoreClock(Structs.DeviceCUDA device, ref int coreClock);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingMaxMemoryClock(Structs.DeviceCUDA device, ref int memoryClock);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingPowerLimit(Structs.DeviceCUDA device, ref int powerLimit);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingThermalLimit(Structs.DeviceCUDA device, ref int thermalLimit);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingFanLevelPercent(Structs.DeviceCUDA device, ref int fanLevel);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentFanTachometerRPM(Structs.DeviceCUDA device, ref int tachometerRPM);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentTemperature(Structs.DeviceCUDA device, ref int temperature);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentCoreClock(Structs.DeviceCUDA device, ref int coreClock);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentMemoryClock(Structs.DeviceCUDA device, ref int memoryClock);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentUtilizationPercent(Structs.DeviceCUDA device, ref int utilization);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentPstate(Structs.DeviceCUDA device, ref int pState);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentThrottleReasons(Structs.DeviceCUDA device, StringBuilder reasons);
        }

        #endregion
    }
}
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
    public static class OpenCL
    {
        #region P/Invoke interface

        public static class Solver
        {
            public const string SOLVER_NAME = "OpenCLSoliditySHA3Solver";

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void FoundADL_API(ref bool hasADL_API);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PreInitialize(StringBuilder sha3Kernel, StringBuilder sha3KingKernel, ulong sha3KernelSize, ulong sha3KingKernelSize);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetPlatforms(ref IntPtr platforms, uint maxPlatforms, ref uint platformCount, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDevicesByPlatform(Structs.DeviceCL.Platform platform, uint maxDeviceCount, ref uint deviceCount, ref IntPtr devices, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern IntPtr GetInstance();

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void DisposeInstance(IntPtr instance);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void InitializeDevice(IntPtr instance, ref Structs.DeviceCL device, bool isKingMaing, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingMaxCoreClock(Structs.DeviceCL device, ref int coreClock, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingMaxMemoryClock(Structs.DeviceCL device, ref int memoryClock, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingPowerLimit(Structs.DeviceCL device, ref int powerLimit, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingThermalLimit(Structs.DeviceCL device, ref int thermalLimit, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceSettingFanLevelPercent(Structs.DeviceCL device, ref int fanLevel, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentFanTachometerRPM(Structs.DeviceCL device, ref int tachometerRPM, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentTemperature(Structs.DeviceCL device, ref int temperature, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentCoreClock(Structs.DeviceCL device, ref int coreClock, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentMemoryClock(Structs.DeviceCL device, ref int memoryClock, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void GetDeviceCurrentUtilizationPercent(Structs.DeviceCL device, ref int utilization, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PushHigh64Target(IntPtr instance, Structs.DeviceCL device, IntPtr high64Target, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PushTarget(IntPtr instance, Structs.DeviceCL device, IntPtr target, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PushMidState(IntPtr instance, Structs.DeviceCL device, IntPtr midState, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void PushMessage(IntPtr instance, Structs.DeviceCL device, IntPtr message, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void Hash(IntPtr instance, ref Structs.DeviceCL device, StringBuilder errorMessage);

            [DllImport(SOLVER_NAME, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
            public static extern void ReleaseDeviceObjects(IntPtr instance, ref Structs.DeviceCL device, StringBuilder errorMessage);
        }

        #endregion
    }
}

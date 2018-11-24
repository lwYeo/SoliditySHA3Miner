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

namespace SoliditySHA3Miner.Miner
{
    public interface IMiner : IDisposable
    {
        NetworkInterface.INetworkInterface NetworkInterface { get; }
        Device.DeviceBase[] Devices { get; }

        bool HasAssignedDevices { get; }
        bool HasMonitoringAPI { get; }
        bool IsAnyInitialised { get; }
        bool IsMining { get; }
        bool IsPause { get; }
        bool IsStopped { get; }

        void StartMining(int networkUpdateInterval, int hashratePrintInterval);

        void StopMining();

        ulong GetTotalHashrate();

        ulong GetHashRateByDevice(Device.DeviceBase device);
    }
}
using System;

namespace SoliditySHA3Miner.Miner
{
    public interface IMiner : IDisposable
    {
        NetworkInterface.INetworkInterface NetworkInterface { get; }
        bool HasAssignedDevices { get; }
        Device[] Devices { get; }
        bool IsAnyInitialised { get; }
        bool IsMining { get; }

        void StartMining(int networkUpdateInterval, int hashratePrintInterval);
        void StopMining();

        ulong GetTotalHashrate();
        ulong GetHashrateByDevice(string platformName, int deviceID);
        long GetDifficulty();
    }

    public struct Device
    {
        public string Type;
        public string Platform;
        public int DeviceID;
        public string Name;
        public float Intensity;
    }
}

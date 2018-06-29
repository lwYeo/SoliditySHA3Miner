using System;

namespace SoliditySHA3Miner.Miner
{
    public interface IMiner : IDisposable
    {
        NetworkInterface.INetworkInterface NetworkInterface { get; }
        bool HasAssignedDevices { get; }
        Device[] Devices { get; }
        bool IsMining { get; }

        void StartMining(int networkUpdateInterval, int hashratePrintInterval);
        void StopMining();

        ulong GetTotalHashrate();
        ulong GetHashrateByDeviceID(int deviceID);
        long GetDifficulty();
    }

    public struct Device
    {
        public string Type;
        public int DeviceID;
        public string Name;
        public float Intensity;
    }
}

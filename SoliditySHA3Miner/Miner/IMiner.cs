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
        bool IsPaused { get; }

        void StartMining(int networkUpdateInterval, int hashratePrintInterval);
        void StopMining();

        ulong GetTotalHashrate();
        ulong GetHashrateByDevice(string platformName, int deviceID);
        long GetDifficulty();
    }

    public static class Work
    {
        private static ulong m_Position = 0;
        private static readonly object m_positionLock = new object();

        public static void GetPosition(ref ulong workPosition)
        {
            lock (m_positionLock) { workPosition = m_Position; }
        }

        public static void ResetPosition(ref ulong lastPosition)
        {
            lock (m_positionLock)
            {
                lastPosition = m_Position;
                m_Position = 0u;
            }
        }

        public static void IncrementPosition(ref ulong lastPosition, ulong increment)
        {
            lock (m_positionLock)
            {
                lastPosition = m_Position;
                m_Position += increment;
            }
        }
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

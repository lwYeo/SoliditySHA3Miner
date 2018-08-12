using System;
using System.Runtime.InteropServices;

namespace SoliditySHA3Miner.Miner
{
    public interface IMiner : IDisposable
    {
        NetworkInterface.INetworkInterface NetworkInterface { get; }
        bool HasAssignedDevices { get; }
        bool HasMonitoringAPI { get; }
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

        public static byte[] SolutionTemplate { get; set; }

        public static unsafe void GetSolutionTemplate(ref byte* solutionTemplate)
        {
            if (SolutionTemplate == null) return;

            fixed (byte *tempSolutionTemplate = SolutionTemplate)
            {
                Buffer.MemoryCopy(tempSolutionTemplate, solutionTemplate, 32L, 32L);
            }
        }

        public static void SetSolutionTemplate(string solutionTemplate) => SolutionTemplate = Utils.Numerics.HexStringToByte32Array(solutionTemplate);

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

using Nethereum.Hex.HexConvertors.Extensions;
using Nethereum.Hex.HexTypes;
using System;
using System.Linq;
using System.Runtime.CompilerServices;

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
    }

    public static class Work
    {
        private static ulong m_Position = 0;

        private static readonly object m_positionLock = new object();

        public static byte[] KingAddress { get; set; }

        public static byte[] SolutionTemplate { get; set; }

        public static string GetKingAddressString()
        {
            if (KingAddress == null) return string.Empty;

            return HexByteConvertorExtensions.ToHex(KingAddress.ToArray(), prefix: true);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void GetKingAddress(byte* kingAddress)
        {
            if (KingAddress != null)
            {
                fixed (byte* tempKingAddress = KingAddress)
                {
                    Buffer.MemoryCopy(tempKingAddress, kingAddress, 20L, 20L);
                }
            }
            else
                for (int i = 0; i < 20; ++i)
                    kingAddress[i] = 0;
        }

        public static void SetKingAddress(string kingAddress)
        {
            if (string.IsNullOrWhiteSpace(kingAddress))
                KingAddress = null;
            else
                KingAddress = new HexBigInteger(kingAddress).ToHexByteArray().ToArray();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void GetSolutionTemplate(byte* solutionTemplate)
        {
            if (SolutionTemplate == null) return;

            fixed (byte* tempSolutionTemplate = SolutionTemplate)
            {
                Buffer.MemoryCopy(tempSolutionTemplate, solutionTemplate, 32L, 32L);
            }
        }

        public static void SetSolutionTemplate(string solutionTemplate) => SolutionTemplate = new HexBigInteger(solutionTemplate).ToHexByteArray();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void GetPosition(ref ulong workPosition)
        {
            lock (m_positionLock) { workPosition = m_Position; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ResetPosition(ref ulong lastPosition)
        {
            lock (m_positionLock)
            {
                lastPosition = m_Position;
                m_Position = 0u;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
        public uint PciBusID;
        public string Name;
        public float Intensity;
    }
}
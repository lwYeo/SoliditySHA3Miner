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

using Nethereum.Hex.HexTypes;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.Miner
{
    public abstract class MinerBase : IMiner
    {
        public const int UINT32_LENGTH = 4;
        public const int UINT64_LENGTH = 8;
        public const int SPONGE_LENGTH = 200;
        public const int ADDRESS_LENGTH = 20;
        public const int UINT256_LENGTH = 32;
        public const int MESSAGE_LENGTH = UINT256_LENGTH + ADDRESS_LENGTH + UINT256_LENGTH;

        private static readonly object m_submissionQueueLock = new object();

        protected Timer m_hashPrintTimer;
        protected int m_pauseOnFailedScan;
        protected int m_failedScanCount;
        protected bool m_isCurrentChallengeStopSolving;
        protected bool m_isSubmitStale;        
        protected string m_AddressString;

        protected HexBigInteger m_Target;
        protected ulong m_High64Target;

        protected byte[] m_ChallengeBytes;
        protected byte[] m_AddressBytes;
        protected byte[] m_SolutionTemplateBytes;
        protected byte[] m_MidStateBytes;

        public IntPtr UnmanagedInstance { get; protected set; }
        
        #region IMiner

        public NetworkInterface.INetworkInterface NetworkInterface { get; protected set; }

        public Device.DeviceBase[] Devices { get; }

        public bool HasAssignedDevices => Devices?.Any(d => d.IsAssigned) ?? false;

        public bool HasMonitoringAPI { get; protected set; }

        public bool IsAnyInitialised => Devices?.Any(d => d.IsInitialized) ?? false;

        public bool IsMining => Devices?.Any(d => d.IsMining) ?? false;

        public bool IsPause => Devices?.Any(d => d.IsPause) ?? false;

        public bool IsStopped => Devices?.Any(d => d.IsStopped) ?? true;

        public void StartMining(int networkUpdateInterval, int hashratePrintInterval)
        {
            try
            {
                NetworkInterface.ResetEffectiveHashrate();
                NetworkInterface.UpdateMiningParameters();

                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += HashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();

                var isKingMaking = !string.IsNullOrWhiteSpace(Work.GetKingAddressString());
                StartFindingAll(isKingMaking);
            }
            catch (Exception ex)
            {
                PrintMessage(string.Empty, string.Empty, -1, "Error", ex.Message);
                StopMining();
            }
        }

        public void StopMining()
        {
            try
            {
                if (m_hashPrintTimer != null)
                    m_hashPrintTimer.Stop();

                NetworkInterface.ResetEffectiveHashrate();

                foreach (var device in Devices)
                    device.IsMining =  false;
            }
            catch (Exception ex)
            {
                PrintMessage(string.Empty, string.Empty, -1, "Error", ex.Message);
            }
        }

        public ulong GetHashRateByDevice(Device.DeviceBase device)
        {
            var hashRate = 0ul;

            if (!IsPause)
                hashRate = (ulong)(device.HashCount / (DateTime.Now - device.HashStartTime).TotalSeconds);

            return hashRate;
        }

        protected void NetworkInterface_OnGetTotalHashrate(NetworkInterface.INetworkInterface sender, ref ulong totalHashrate)
        {
            totalHashrate += GetTotalHashrate();
        }

        public ulong GetTotalHashrate()
        {
            var totalHashrate = 0ul;
            try
            {
                foreach (var device in Devices)
                    totalHashrate += GetHashRateByDevice(device);
            }
            catch (Exception ex)
            {
                PrintMessage(string.Empty, string.Empty, -1, "Error", ex.Message);
            }
            return totalHashrate;
        }

        public virtual void Dispose()
        {
            NetworkInterface.OnGetTotalHashrate -= NetworkInterface_OnGetTotalHashrate;
            NetworkInterface.OnGetMiningParameterStatus -= NetworkInterface_OnGetMiningParameterStatus;
            NetworkInterface.OnNewChallenge -= NetworkInterface_OnNewChallenge;
            NetworkInterface.OnNewTarget -= NetworkInterface_OnNewTarget;
            NetworkInterface.OnStopSolvingCurrentChallenge -= NetworkInterface_OnStopSolvingCurrentChallenge;

            if (m_hashPrintTimer != null)
            {
                try
                {
                    m_hashPrintTimer.Elapsed -= HashPrintTimer_Elapsed;
                    m_hashPrintTimer.Dispose();
                }
                catch { }
                m_hashPrintTimer = null;
            }

            try
            {
                if (Devices != null)
                    Devices.AsParallel().
                            ForAll(d => d.Dispose());
            }
            catch { }

            try { NetworkInterface.Dispose(); }
            catch { }
        }

        #endregion IMiner
        
        protected abstract void HashPrintTimer_Elapsed(object sender, ElapsedEventArgs e);
        protected abstract void AssignDevices();
        protected abstract void PushHigh64Target(Device.DeviceBase device);
        protected abstract void PushTarget(Device.DeviceBase device);
        protected abstract void PushMidState(Device.DeviceBase device);
        protected abstract void PushMessage(Device.DeviceBase device);
        protected abstract void StartFinding(Device.DeviceBase device, bool isKingMaking);

        public MinerBase(NetworkInterface.INetworkInterface networkInterface, Device.DeviceBase[] devices, bool isSubmitStale, int pauseOnFailedScans)
        {
            m_failedScanCount = 0;
            m_pauseOnFailedScan = pauseOnFailedScans;
            m_isSubmitStale = isSubmitStale;
            NetworkInterface = networkInterface;
            Devices = devices;

            m_ChallengeBytes = (byte[])Array.CreateInstance(typeof(byte), UINT256_LENGTH);
            m_AddressBytes = (byte[])Array.CreateInstance(typeof(byte), ADDRESS_LENGTH);

            NetworkInterface.OnGetTotalHashrate += NetworkInterface_OnGetTotalHashrate;
            NetworkInterface.OnGetMiningParameterStatus += NetworkInterface_OnGetMiningParameterStatus;
            NetworkInterface.OnNewChallenge += NetworkInterface_OnNewChallenge;
            NetworkInterface.OnNewTarget += NetworkInterface_OnNewTarget;
            NetworkInterface.OnStopSolvingCurrentChallenge += NetworkInterface_OnStopSolvingCurrentChallenge;
        }

        protected void PrintMessage(string platformType, string platform, int deviceEnum, string type, string message)
        {
            var sFormat = new StringBuilder();
            if (!string.IsNullOrWhiteSpace(platformType)) sFormat.Append(platformType + " ");

            if (!string.IsNullOrWhiteSpace(platform))
            {
                if (sFormat.Length > 0)
                    sFormat.AppendFormat("({0}) ", platform);
                else
                    sFormat.Append(platform + " ");
            }
            if (deviceEnum > -1) sFormat.Append("ID: {0} ");

            switch (type.ToUpperInvariant())
            {
                case "INFO":
                    sFormat.Append(deviceEnum > -1 ? "[INFO] {1}" : "[INFO] {0}");
                    break;

                case "WARN":
                    sFormat.Append(deviceEnum > -1 ? "[WARN] {1}" : "[WARN] {0}");
                    break;

                case "ERROR":
                    sFormat.Append(deviceEnum > -1 ? "[ERROR] {1}" : "[ERROR] {0}");
                    break;

                case "DEBUG":
                default:
#if DEBUG
                    sFormat.Append(deviceEnum > -1 ? "[DEBUG] {1}" : "[DEBUG] {0}");
                    break;
#else
                    return;
#endif
            }
            Program.Print(deviceEnum > -1
                ? string.Format(sFormat.ToString(), deviceEnum, message)
                : string.Format(sFormat.ToString(), message));
        }

        private void NetworkInterface_OnGetMiningParameterStatus(NetworkInterface.INetworkInterface sender, bool success)
        {
            try
            {
                if (UnmanagedInstance != null && UnmanagedInstance.ToInt64() != 0)
                {
                    if (success)
                    {
                        var isPause = Devices.All(d => d.IsPause);

                        if (m_isCurrentChallengeStopSolving) { isPause = true; }
                        else if (isPause)
                        {
                            if (m_failedScanCount > m_pauseOnFailedScan)
                                m_failedScanCount = 0;

                            isPause = false;
                        }
                        foreach (var device in Devices)
                            device.IsPause = isPause;
                    }
                    else
                    {
                        m_failedScanCount++;

                        var isMining = Devices.Any(d => d.IsMining);

                        if (m_failedScanCount > m_pauseOnFailedScan && IsMining)
                            foreach (var device in Devices)
                                device.IsPause = true;
                    }
                }
            }
            catch (Exception ex)
            {
                PrintMessage(string.Empty, string.Empty, -1, "Error", ex.Message);
            }
        }

        private void NetworkInterface_OnNewChallenge(NetworkInterface.INetworkInterface sender, byte[] challenge, string address)
        {
            try
            {
                if (UnmanagedInstance != null && UnmanagedInstance.ToInt64() != 0)
                {
                    for (var i = 0; i < challenge.Length; i++)
                        m_ChallengeBytes[i] = challenge[i];

                    m_AddressString = address;
                    // some pools provide invalid checksum address
                    Utils.Numerics.AddressStringToByte20Array(address, ref m_AddressBytes, isChecksum:false);
                    
                    m_SolutionTemplateBytes = Work.SolutionTemplate;
                    m_MidStateBytes = Helper.CPU.GetMidState(m_ChallengeBytes, m_AddressBytes, m_SolutionTemplateBytes);

                    foreach (var device in Devices)
                    {
                        Array.ConstrainedCopy(m_ChallengeBytes, 0, device.Message, 0, UINT256_LENGTH);
                        Array.ConstrainedCopy(m_AddressBytes, 0, device.Message, UINT256_LENGTH, ADDRESS_LENGTH);
                        Array.ConstrainedCopy(m_SolutionTemplateBytes, 0, device.Message, UINT256_LENGTH + ADDRESS_LENGTH, UINT256_LENGTH);

                        Array.Copy(m_ChallengeBytes, device.Challenge, UINT256_LENGTH);
                        Array.Copy(m_MidStateBytes, device.MidState, SPONGE_LENGTH);                        
                        device.HasNewChallenge = true;
                    }

                    if (m_isCurrentChallengeStopSolving)
                    {
                        foreach (var device in Devices)
                            device.IsPause = false;

                        m_isCurrentChallengeStopSolving = false;
                    }
                }
            }
            catch (Exception ex)
            {
                PrintMessage(string.Empty, string.Empty, -1, "Error", ex.Message);
            }
        }

        private void NetworkInterface_OnNewTarget(NetworkInterface.INetworkInterface sender, HexBigInteger target)
        {
            try
            {
                var targetBytes = Utils.Numerics.FilterByte32Array(target.Value.ToByteArray(isUnsigned: true, isBigEndian:true));
                var high64Bytes = targetBytes.Take(UINT64_LENGTH).Reverse().ToArray();

                m_Target = target;
                m_High64Target = BitConverter.ToUInt64(high64Bytes);

                foreach (var device in Devices)
                {
                    Array.Copy(targetBytes, device.Target, UINT256_LENGTH);
                    device.High64Target[0] = m_High64Target;
                    device.HasNewTarget = true;
                }
            }
            catch (Exception ex)
            {
                PrintMessage(string.Empty, string.Empty, -1, "Error", ex.Message);
            }
        }

        private void NetworkInterface_OnStopSolvingCurrentChallenge(NetworkInterface.INetworkInterface sender, bool stopSolving = true)
        {
            if (stopSolving)
            {
                if (m_isCurrentChallengeStopSolving) return;

                m_isCurrentChallengeStopSolving = true;

                foreach (var device in Devices)
                    device.IsPause = true;

                PrintMessage(string.Empty, string.Empty, -1, "Info", "Mining temporary paused until new challenge receive...");
            }
            else if (m_isCurrentChallengeStopSolving)
            {
                PrintMessage(string.Empty, string.Empty, -1, "Info", "Resume mining...");

                m_isCurrentChallengeStopSolving = false;

                foreach (var device in Devices)
                    device.IsPause = false;
            }
        }

        private void StartFindingAll(bool isKingMaking)
        {
            foreach (var device in Devices)
                Task.Factory.StartNew(() => StartFinding(device, isKingMaking));
        }

        protected void CheckInputs(Device.DeviceBase device, bool isKingMaking, ref byte[] currentChallenge)
        {
            if (device.HasNewTarget || device.HasNewChallenge)
            {
                if (device.HasNewTarget)
                {
                    if (isKingMaking) PushTarget(device);
                    else PushHigh64Target(device);

                    device.HasNewTarget = false;
                }

                if (device.HasNewChallenge)
                {
                    if (isKingMaking) PushMessage(device);
                    else PushMidState(device);

                    Array.Copy(device.Challenge, currentChallenge, UINT256_LENGTH);
                    device.HasNewChallenge = false;
                }

                device.HashStartTime = DateTime.Now;
                device.HashCount = 0;
            }
        }

        protected void SubmitSolutions(ulong[] solutions, byte[] challenge, string platformType, string platform, int deviceID, uint solutionCount, bool isKingMaking)
        {
            Task.Factory.StartNew(() =>
            {
                lock (m_submissionQueueLock)
                {
                    foreach (var solution in solutions)
                    {
                        if (NetworkInterface.GetType().IsAssignableFrom(typeof(NetworkInterface.SlaveInterface)))
                            if (((NetworkInterface.SlaveInterface)NetworkInterface).IsPause)
                                return;

                        if (!NetworkInterface.IsPool && NetworkInterface.IsChallengedSubmitted(challenge))
                            return; // Solo mining should submit only 1 valid nonce per challange

                        if (!m_isSubmitStale && !challenge.SequenceEqual(NetworkInterface.CurrentChallenge))
                            return;

                        if (m_isSubmitStale && !challenge.SequenceEqual(NetworkInterface.CurrentChallenge))
                            PrintMessage(platformType, platform, deviceID, "Warn", "Found stale solution, verifying...");
                        else
                            PrintMessage(platformType, platform, deviceID, "Info", "Found solution, verifying...");

                        var solutionBytes = BitConverter.GetBytes(solution);
                        var nonceBytes = Utils.Numerics.FilterByte32Array(m_SolutionTemplateBytes.ToArray());
                        var messageBytes = (byte[])Array.CreateInstance(typeof(byte), UINT256_LENGTH + ADDRESS_LENGTH + UINT256_LENGTH);
                        var digestBytes = (byte[])Array.CreateInstance(typeof(byte), UINT256_LENGTH);
                        var messageHandle = GCHandle.Alloc(messageBytes, GCHandleType.Pinned);
                        var messagePointer = messageHandle.AddrOfPinnedObject();
                        var digestHandle = GCHandle.Alloc(digestBytes, GCHandleType.Pinned);
                        var digestPointer = digestHandle.AddrOfPinnedObject();

                        if (isKingMaking)
                            Array.ConstrainedCopy(solutionBytes, 0, nonceBytes, ADDRESS_LENGTH, UINT64_LENGTH);
                        else
                            Array.ConstrainedCopy(solutionBytes, 0, nonceBytes, (UINT256_LENGTH / 2) - (UINT64_LENGTH / 2), UINT64_LENGTH);

                        Array.ConstrainedCopy(challenge, 0, messageBytes, 0, UINT256_LENGTH);
                        Array.ConstrainedCopy(m_AddressBytes, 0, messageBytes, UINT256_LENGTH, ADDRESS_LENGTH);
                        Array.ConstrainedCopy(nonceBytes, 0, messageBytes, UINT256_LENGTH + ADDRESS_LENGTH, UINT256_LENGTH);

                        Helper.CPU.Solver.SHA3(messagePointer, digestPointer);
                        messageHandle.Free();
                        digestHandle.Free();

                        var nonceString = Utils.Numerics.Byte32ArrayToHexString(nonceBytes);
                        var challengeString = Utils.Numerics.Byte32ArrayToHexString(challenge);
                        var digestString = Utils.Numerics.Byte32ArrayToHexString(digestBytes);
                        var digest = new HexBigInteger(digestString);

                        if (digest.Value >= m_Target.Value)
                        {
                            PrintMessage(platformType, platform, deviceID, "Error",
                                         "Verification failed: invalid solution"
                                         + "\nChallenge: " + challengeString
                                         + "\nAddress: " + m_AddressString
                                         + "\nNonce: " + nonceString
                                         + "\nDigest: " + digestString
                                         + "\nTarget: " + Utils.Numerics.Byte32ArrayToHexString(m_Target.Value.ToByteArray(isUnsigned: true, isBigEndian: true)));
                        }
                        else
                        {
                            PrintMessage(platformType, platform, deviceID, "Info", "Solution verified, submitting nonce " + nonceString + "...");

                            PrintMessage(platformType, platform, deviceID, "Debug",
                                         "Solution details..."
                                         + "\nChallenge: " + challengeString
                                         + "\nAddress: " + m_AddressString
                                         + "\nNonce: " + nonceString
                                         + "\nDigest: " + digestString
                                         + "\nTarget: " + Utils.Numerics.Byte32ArrayToHexString(m_Target.Value.ToByteArray(isUnsigned: true, isBigEndian: true)));

                            NetworkInterface.SubmitSolution(m_AddressString,
                                                            digest,
                                                            challenge,
                                                            NetworkInterface.Difficulty,
                                                            nonceBytes,
                                                            this);
                        }
                    }
                }
            });
        }
    }
}
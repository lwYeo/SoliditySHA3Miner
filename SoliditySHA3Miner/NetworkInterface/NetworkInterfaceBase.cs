using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Timers;
using Nethereum.Hex.HexTypes;

namespace SoliditySHA3Miner.NetworkInterface
{
    public abstract class NetworkInterfaceBase : INetworkInterface
    {
        protected const int MAX_SUBMIT_DTM_COUNT = 50;
        protected readonly List<DateTime> m_submitDateTimeList;
        protected readonly List<byte[]> m_submittedChallengeList;
        protected readonly BigInteger UInt256_MaxValue = BigInteger.Pow(2, 256);
        protected readonly int m_updateInterval;

        protected DateTime m_challengeReceiveDateTime;
        protected MiningParameters m_lastParameters;
        protected Timer m_updateMinerTimer;
        protected Timer m_hashPrintTimer;

        #region INetworkInterface

        public abstract bool IsPool { get; }

        public ulong SubmittedShares { get; protected set; }

        public ulong RejectedShares { get; protected set; }

        public HexBigInteger Difficulty { get; protected set; }

        public HexBigInteger MaxTarget { get; protected set; }

        public int LastSubmitLatency { get; protected set; }

        public int Latency { get; protected set; }

        public string MinerAddress { get; protected set; }

        public virtual string SubmitURL { get; protected set; }

        public byte[] CurrentChallenge { get; protected set; }

        public HexBigInteger CurrentTarget { get; protected set; }

        public abstract event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        public abstract event NewChallengeEvent OnNewChallenge;
        public abstract event NewTargetEvent OnNewTarget;
        public abstract event NewDifficultyEvent OnNewDifficulty;
        public abstract event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;
        public abstract event GetTotalHashrateEvent OnGetTotalHashrate;

        public abstract bool SubmitSolution(string address, byte[] digest, byte[] challenge, HexBigInteger difficulty, byte[] nonce, object sender);

        public virtual void Dispose() // Pool and Web3 Interface has more to dispose
        {
            if (m_submitDateTimeList != null)
                m_submitDateTimeList.Clear();

            if (m_submittedChallengeList != null)
                m_submittedChallengeList.Clear();

            if (m_updateMinerTimer != null)
                try
                {
                    m_updateMinerTimer.Stop();
                    m_updateMinerTimer.Elapsed -= UpdateMinerTimer_Elapsed;
                    m_updateMinerTimer.Dispose();
                }
                catch { }
            m_updateMinerTimer = null;

            if (m_hashPrintTimer != null)
                try
                {
                    m_hashPrintTimer.Stop();
                    m_hashPrintTimer.Elapsed -= HashPrintTimer_Elapsed;
                    m_hashPrintTimer.Dispose();
                }
                catch { }
            m_hashPrintTimer = null;
        }

        /// <summary>
        /// <para>Since a single hash is a random number between 1 and 2^256, and difficulty [1] target = 2^234</para>
        /// <para>Then we can find difficulty [N] target = 2^234 / N</para>
        /// <para>Hence, # of hashes to find block with difficulty [N] = N * 2^256 / 2^234</para>
        /// <para>Which simplifies to # of hashes to find block difficulty [N] = N * 2^22</para>
        /// <para>Time to find block in seconds with difficulty [N] = N * 2^22 / hashes per second</para>
        /// <para>Hashes per second with difficulty [N] and time to find block [T] = N * 2^22 / T</para>
        /// </summary>
        public ulong GetEffectiveHashrate()
        {
            var hashrate = 0ul;

            if (m_submitDateTimeList.Count > 1)
            {
                var avgSolveTime = (ulong)((DateTime.Now - m_submitDateTimeList.First()).TotalSeconds / m_submitDateTimeList.Count - 1);
                hashrate = (ulong)(Difficulty.Value * UInt256_MaxValue / MaxTarget.Value / new BigInteger(avgSolveTime));
            }

            return hashrate;
        }

        /// <summary>
        /// <para>Since a single hash is a random number between 1 and 2^256, and difficulty [1] target = 2^234</para>
        /// <para>Then we can find difficulty [N] target = 2^234 / N</para>
        /// <para>Hence, # of hashes to find block with difficulty [N] = N * 2^256 / 2^234</para>
        /// <para>Which simplifies to # of hashes to find block difficulty [N] = N * 2^22</para>
        /// <para>Time to find block in seconds with difficulty [N] = N * 2^22 / hashes per second</para>
        /// </summary>
        public TimeSpan GetTimeLeftToSolveBlock(ulong hashrate)
        {
            if (MaxTarget == null || MaxTarget.Value == 0 || Difficulty == null || Difficulty.Value == 0 || hashrate == 0 || m_challengeReceiveDateTime == DateTime.MinValue)
                return TimeSpan.Zero;

            var timeToSolveBlock = new BigInteger(Difficulty) * UInt256_MaxValue / MaxTarget.Value / new BigInteger(hashrate);

            var secondsLeftToSolveBlock = timeToSolveBlock - (long)(DateTime.Now - m_challengeReceiveDateTime).TotalSeconds;

            return (secondsLeftToSolveBlock > (long)TimeSpan.MaxValue.TotalSeconds)
                ? TimeSpan.MaxValue
                : TimeSpan.FromSeconds((long)secondsLeftToSolveBlock);
        }

        public bool IsChallengedSubmitted(byte[] challenge)
        {
            return m_submittedChallengeList.Any(s => challenge.SequenceEqual(s));
        }

        public void ResetEffectiveHashrate()
        {
            m_submitDateTimeList.Clear();
            m_submitDateTimeList.Add(DateTime.Now);
        }

        public void UpdateMiningParameters()
        {
            UpdateMinerTimer_Elapsed(this, null);

            if (m_updateMinerTimer == null && m_updateInterval > 0)
            {
                m_updateMinerTimer = new Timer(m_updateInterval);
                m_updateMinerTimer.Elapsed += UpdateMinerTimer_Elapsed;
                m_updateMinerTimer.Start();
            }
        }

        #endregion

        public NetworkInterfaceBase(int updateInterval, int hashratePrintInterval)
        {
            m_submittedChallengeList = new List<byte[]>();
            m_submitDateTimeList = new List<DateTime>(MAX_SUBMIT_DTM_COUNT + 1);
            m_updateInterval = updateInterval;

            Difficulty = new HexBigInteger(0);
            SubmittedShares = 0;
            RejectedShares = 0;
            LastSubmitLatency = -1;
            Latency = -1;

            if (hashratePrintInterval > 0)
            {
                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += HashPrintTimer_Elapsed;
            }
        }
        
        protected abstract void UpdateMinerTimer_Elapsed(object sender, ElapsedEventArgs e);
        protected abstract void HashPrintTimer_Elapsed(object sender, ElapsedEventArgs e);
    }
}

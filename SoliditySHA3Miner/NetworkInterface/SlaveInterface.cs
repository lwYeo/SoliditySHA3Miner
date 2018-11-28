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
using Nethereum.Util;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class SlaveInterface : INetworkInterface
    {
        private readonly BigInteger uint256_MaxValue = BigInteger.Pow(2, 256);
        private DateTime m_challengeReceiveDateTime;

        private const int MAX_SUBMIT_DTM_COUNT = 50;
        private readonly List<DateTime> m_submitDateTimeList;
        private readonly int m_updateInterval;
        private bool m_isGetMiningParameters;
        private Timer m_updateMinerTimer;
        private Timer m_hashPrintTimer;
        private MiningParameters m_lastParameters;

        public event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        public event NewChallengeEvent OnNewChallenge;
        public event NewTargetEvent OnNewTarget;
        public event NewDifficultyEvent OnNewDifficulty;
        public event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;
        public event GetTotalHashrateEvent OnGetTotalHashrate;

        public bool IsPool => true;
        public bool IsPause { get; private set; }
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }
        public HexBigInteger Difficulty { get; private set; }
        public HexBigInteger MaxTarget { get; private set; }
        public int LastSubmitLatency { get; private set; }
        public int Latency { get; private set; }
        public string MinerAddress { get; }
        public string SubmitURL { get; }
        public byte[] CurrentChallenge { get; private set; }
        public HexBigInteger CurrentTarget { get; private set; }

        public SlaveInterface(string masterURL, int updateInterval, int hashratePrintInterval)
        {
            m_updateInterval = updateInterval;
            m_isGetMiningParameters = false;
            LastSubmitLatency = -1;
            Latency = -1;

            SubmitURL = masterURL;
            SubmittedShares = 0;
            RejectedShares = 0;

            m_submitDateTimeList = new List<DateTime>(MAX_SUBMIT_DTM_COUNT + 1);

            Program.Print(string.Format("[INFO] Waiting for master instance ({0}) to start...", SubmitURL));

            var getMasterAddress = MasterInterface.GetMasterParameter(MasterInterface.RequestMethods.GetMasterAddress);
            var getMaximumTarget = MasterInterface.GetMasterParameter(MasterInterface.RequestMethods.GetMaximumTarget);
            var getKingAddress = MasterInterface.GetMasterParameter(MasterInterface.RequestMethods.GetKingAddress);

            var retryCount = 0;
            while (true)
                try
                {
                    var parameters = MiningParameters.GetMiningParameters(SubmitURL,
                                                                          getEthAddress: getMasterAddress,
                                                                          getMaximumTarget: getMaximumTarget,
                                                                          getKingAddress: getKingAddress);
                    MinerAddress = parameters.EthAddress;
                    MaxTarget = parameters.MaximumTarget;
                    Miner.Work.KingAddress = parameters.KingAddressByte20;
                    break;
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount > 3)
                        HandleException(ex);

                    Task.Delay(1000).Wait();
                }

            if (hashratePrintInterval > 0)
            {
                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += HashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();
            }
        }

        public void Dispose()
        {
            if (m_submitDateTimeList != null)
                m_submitDateTimeList.Clear();

            if (m_updateMinerTimer != null)
            {
                try
                {
                    m_updateMinerTimer.Elapsed -= UpdateMinerTimer_Elapsed;
                    m_updateMinerTimer.Stop();
                    m_updateMinerTimer.Dispose();
                }
                catch { }
                m_updateMinerTimer = null;
            }

            if (m_hashPrintTimer != null)
            {
                try
                {
                    m_hashPrintTimer.Elapsed -= HashPrintTimer_Elapsed;
                    m_hashPrintTimer.Stop();
                    m_hashPrintTimer.Dispose();
                }
                catch { }
                m_hashPrintTimer = null;
            }
        }

        public MiningParameters GetMiningParameters()
        {
            Program.Print(string.Format("[INFO] Checking latest parameters from master URL: {0}", SubmitURL));

            var getChallenge = MasterInterface.GetMasterParameter(MasterInterface.RequestMethods.GetChallenge);
            var getDifficulty = MasterInterface.GetMasterParameter(MasterInterface.RequestMethods.GetDifficulty);
            var getTarget = MasterInterface.GetMasterParameter(MasterInterface.RequestMethods.GetTarget);
            var getPause = MasterInterface.GetMasterParameter(MasterInterface.RequestMethods.GetPause);

            var success = true;
            var startTime = DateTime.Now;
            try
            {
                return MiningParameters.GetMiningParameters(SubmitURL,
                                                            getChallenge: getChallenge,
                                                            getDifficulty: getDifficulty,
                                                            getTarget: getTarget,
                                                            getPause: getPause);
            }
            catch (Exception ex)
            {
                HandleException(ex);
                success = false;
            }
            finally
            {
                if (success)
                    Latency = (int)(DateTime.Now - startTime).TotalMilliseconds;
            }
            return null;
        }

        private void HashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            try
            {
                var totalHashRate = 0ul;
                OnGetTotalHashrate(this, ref totalHashRate);
                Program.Print(string.Format("[INFO] Total Hashrate: {0} MH/s (Effective) / {1} MH/s (Local)",
                                            GetEffectiveHashrate() / 1000000.0f, totalHashRate / 1000000.0f));

                var timeLeftToSolveBlock = GetTimeLeftToSolveBlock(totalHashRate);

                if (timeLeftToSolveBlock.TotalSeconds < 0)
                {
                    Program.Print(string.Format("[INFO] Estimated time left to solution: -({0}d {1}h {2}m {3}s)",
                                                Math.Abs(timeLeftToSolveBlock.Days),
                                                Math.Abs(timeLeftToSolveBlock.Hours),
                                                Math.Abs(timeLeftToSolveBlock.Minutes),
                                                Math.Abs(timeLeftToSolveBlock.Seconds)));
                }
                else
                {
                    Program.Print(string.Format("[INFO] Estimated time left to solution: {0}d {1}h {2}m {3}s",
                                                Math.Abs(timeLeftToSolveBlock.Days),
                                                Math.Abs(timeLeftToSolveBlock.Hours),
                                                Math.Abs(timeLeftToSolveBlock.Minutes),
                                                Math.Abs(timeLeftToSolveBlock.Seconds)));
                }
            }
            catch { }
        }

        private void UpdateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            if (m_isGetMiningParameters) return;
            try
            {
                m_isGetMiningParameters = true;
                var miningParameters = GetMiningParameters();
                if (miningParameters == null)
                {
                    OnGetMiningParameterStatus(this, false);
                    return;
                }
                
                CurrentChallenge = miningParameters.ChallengeByte32;

                if (m_lastParameters == null || miningParameters.Challenge.Value != m_lastParameters.Challenge.Value)
                {
                    Program.Print(string.Format("[INFO] New challenge detected {0}...", miningParameters.ChallengeByte32String));
                    OnNewChallenge(this, miningParameters.ChallengeByte32, MinerAddress);

                    if (m_challengeReceiveDateTime == DateTime.MinValue)
                        m_challengeReceiveDateTime = DateTime.Now;
                }

                if (m_lastParameters == null || miningParameters.MiningTarget.Value != m_lastParameters.MiningTarget.Value)
                {
                    Program.Print(string.Format("[INFO] New target detected {0}...", miningParameters.MiningTargetByte32String));
                    OnNewTarget(this, miningParameters.MiningTarget);
                }

                if (m_lastParameters == null || miningParameters.MiningDifficulty.Value != m_lastParameters.MiningDifficulty.Value)
                {
                    Program.Print(string.Format("[INFO] New difficulity detected ({0})...", miningParameters.MiningDifficulty.Value));
                    OnNewDifficulty?.Invoke(this, miningParameters.MiningDifficulty);
                    Difficulty = miningParameters.MiningDifficulty;

                    // Actual difficulty should have decimals
                    var calculatedDifficulty = BigDecimal.Exp(BigInteger.Log(MaxTarget.Value) - BigInteger.Log(miningParameters.MiningTarget.Value));
                    var calculatedDifficultyBigInteger = BigInteger.Parse(calculatedDifficulty.ToString().Split(",.".ToCharArray())[0]);

                    // Only replace if the integer portion is different
                    if (Difficulty.Value != calculatedDifficultyBigInteger)
                    {
                        Difficulty = new HexBigInteger(calculatedDifficultyBigInteger);
                        var expValue = BigInteger.Log10(calculatedDifficultyBigInteger);
                        var calculatedTarget = BigInteger.Parse(
                            (BigDecimal.Parse(MaxTarget.Value.ToString()) * BigDecimal.Pow(10, expValue) / (calculatedDifficulty * BigDecimal.Pow(10, expValue))).
                            ToString().Split(",.".ToCharArray())[0]);
                        var calculatedTargetHex = new HexBigInteger(calculatedTarget);

                        Program.Print(string.Format("[INFO] Update target 0x{0}...", calculatedTarget.ToString("x64")));
                        OnNewTarget(this, calculatedTargetHex);
                        CurrentTarget = calculatedTargetHex;
                    }
                }

                IsPause = miningParameters.IsPause;
                m_lastParameters = miningParameters;

                OnStopSolvingCurrentChallenge(this, stopSolving: miningParameters.IsPause);
                OnGetMiningParameterStatus(this, true);
            }
            catch (Exception ex)
            {
                HandleException(ex);
            }
            finally { m_isGetMiningParameters = false; }
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

            var timeToSolveBlock = new BigInteger(Difficulty) * uint256_MaxValue / MaxTarget.Value / new BigInteger(hashrate);

            var secondsLeftToSolveBlock = timeToSolveBlock - (long)(DateTime.Now - m_challengeReceiveDateTime).TotalSeconds;

            return (secondsLeftToSolveBlock > (long)TimeSpan.MaxValue.TotalSeconds)
                ? TimeSpan.MaxValue
                : TimeSpan.FromSeconds((long)secondsLeftToSolveBlock);
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
                hashrate = (ulong)(new BigInteger(Difficulty) * uint256_MaxValue / MaxTarget.Value / new BigInteger(avgSolveTime));
            }

            return hashrate;
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

        public bool SubmitSolution(string address, byte[] digest, byte[] challenge, HexBigInteger difficulty, byte[] nonce, object sender)
        {
            m_challengeReceiveDateTime = DateTime.Now;
            var startSubmitDateTime = DateTime.Now;

            bool success = false, submitted = false;
            int retryCount = 0, maxRetries = 10;
            do
            {
                try
                {
                    lock (this)
                    {
                        if (IsPause) return false;

                        if (SubmittedShares == ulong.MaxValue)
                        {
                            SubmittedShares = 0;
                            RejectedShares = 0;
                        }

                        Program.Print(string.Format("[INFO] Submitting solution to master URL({0})...", SubmitURL));

                        JObject submitSolution = MasterInterface.GetMasterParameter(MasterInterface.RequestMethods.SubmitSolution,
                                                                                    Utils.Numerics.Byte32ArrayToHexString(digest),
                                                                                    Utils.Numerics.Byte32ArrayToHexString(challenge),
                                                                                    Utils.Numerics.Byte32ArrayToHexString(difficulty.Value.ToByteArray(isUnsigned: true, isBigEndian: true)),
                                                                                    Utils.Numerics.Byte32ArrayToHexString(nonce));

                        var response = Utils.Json.InvokeJObjectRPC(SubmitURL, submitSolution, customTimeout: 10 * 60);

                        LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);

                        var result = response.SelectToken("$.result")?.Value<string>();

                        success = (result ?? string.Empty).Equals("true", StringComparison.OrdinalIgnoreCase);
                        if (!success) RejectedShares++;
                        SubmittedShares++;

                        Program.Print(string.Format("[INFO] Solution submitted to master URL({0}): {1} ({2}ms)",
                                                    SubmitURL,
                                                    (success ? "success" : "failed"),
                                                    LastSubmitLatency));
#if DEBUG
                        Program.Print(submitSolution.ToString());
                        Program.Print(response.ToString());
#endif
                        if (success)
                        {
                            if (m_submitDateTimeList.Count > MAX_SUBMIT_DTM_COUNT) m_submitDateTimeList.RemoveAt(0);
                            m_submitDateTimeList.Add(DateTime.Now);
                        }
                        UpdateMiningParameters();
                    }
                    submitted = true;
                }
                catch (Exception ex)
                {
                    retryCount += 1;

                    if (retryCount >= Math.Min(maxRetries, 3))
                        HandleException(ex, "Master not receiving nonce:");

                    if (retryCount < maxRetries)
                        Task.Delay(500);
                }
            } while (!submitted && retryCount < maxRetries);

            return success;
        }

        private void HandleException(Exception ex, string errorPrefix = null)
        {
            var errorMessage = new StringBuilder("[ERROR] Occured at Slave instance: ");

            if (!string.IsNullOrWhiteSpace(errorPrefix))
                errorMessage.AppendFormat("({0}) ", errorPrefix);

            errorMessage.Append(ex.Message);

            var innerEx = ex.InnerException;
            while (innerEx != null)
            {
                errorMessage.AppendFormat("\n {0}", innerEx.Message);
                innerEx = innerEx.InnerException;
            }
            Program.Print(errorMessage.ToString());
        }
    }
}
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
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.NetworkInformation;
using System.Numerics;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class PoolInterface : INetworkInterface
    {
        private readonly BigInteger uint256_MaxValue = BigInteger.Pow(2, 256);
        private HexBigInteger m_maxTarget;
        private DateTime m_challengeReceiveDateTime;

        private const int MAX_SUBMIT_DTM_COUNT = 50;
        private readonly List<DateTime> m_submitDateTimeList;

        private readonly string s_PoolURL;
        private readonly ulong m_customDifficulity;
        private readonly int m_maxScanRetry;
        private readonly int m_updateInterval;
        private bool m_isGetMiningParameters;
        private Timer m_updateMinerTimer;
        private Timer m_hashPrintTimer;
        private bool m_runFailover;
        private int m_retryCount;
        private MiningParameters m_lastParameters;

        public event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        public event NewChallengeEvent OnNewChallenge;
        public event NewTargetEvent OnNewTarget;

#   pragma warning disable 67 // Unused event
        public event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;
#   pragma warning restore 67

        public event GetTotalHashrateEvent OnGetTotalHashrate;

        public bool IsPool => true;
        public bool IsSecondaryPool { get; }
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }
        public PoolInterface SecondaryPool { get; }
        public ulong Difficulty { get; private set; }
        public int LastSubmitLatency { get; private set; }
        public int Latency { get; private set; }
        public string MinerAddress { get; }

        public string SubmitURL
        {
            get
            {
                if (m_runFailover)
                    return SecondaryPool.SubmitURL;
                else
                    return s_PoolURL;
            }
        }

        public byte[] CurrentChallenge { get; private set; }

        public PoolInterface(string minerAddress, string poolURL, int maxScanRetry, int updateInterval, int hashratePrintInterval,
                             ulong customDifficulity, bool isSecondary, HexBigInteger maxTarget, PoolInterface secondaryPool = null)
        {
            m_retryCount = 0;
            m_maxScanRetry = maxScanRetry;
            m_customDifficulity = customDifficulity;
            m_maxTarget = maxTarget;
            m_updateInterval = updateInterval;
            m_isGetMiningParameters = false;
            LastSubmitLatency = -1;
            Latency = -1;
            SecondaryPool = secondaryPool;
            IsSecondaryPool = isSecondary;

            if (m_customDifficulity > 0)
                Difficulty = m_customDifficulity;

            MinerAddress = minerAddress;
            s_PoolURL = poolURL;
            SubmittedShares = 0ul;
            RejectedShares = 0ul;

            m_submitDateTimeList = new List<DateTime>(MAX_SUBMIT_DTM_COUNT + 1);

            if (hashratePrintInterval > 0)
            {
                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += HashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();
            }
        }

        public void Dispose()
        {
            if (SecondaryPool != null) SecondaryPool.Dispose();

            m_submitDateTimeList.Clear();

            if (m_updateMinerTimer != null)
            {
                m_updateMinerTimer.Stop();
                m_updateMinerTimer.Elapsed -= UpdateMinerTimer_Elapsed;
                m_updateMinerTimer.Dispose();
                m_updateMinerTimer = null;
            }

            if (m_hashPrintTimer != null)
            {
                m_hashPrintTimer.Stop();
                m_hashPrintTimer.Elapsed -= HashPrintTimer_Elapsed;
                m_hashPrintTimer.Dispose();
                m_hashPrintTimer = null;
            }
        }

        private JObject GetPoolParameter(string method, params string[] parameters)
        {
            var paramObject = new JObject
            {
                ["jsonrpc"] = "2.0",
                ["id"] = "1",
                ["method"] = method
            };
            if (parameters != null)
            {
                if (parameters.Length > 0)
                {
                    JArray props = new JArray();
                    foreach (var p in parameters) { props.Add(p); }
                    paramObject.Add(new JProperty("params", props));
                }
            }
            return paramObject;
        }

        public MiningParameters GetMiningParameters()
        {
            Program.Print(string.Format("[INFO] Checking latest parameters from {0} pool...", IsSecondaryPool ? "secondary" : "primary"));

            var getPoolEthAddress = GetPoolParameter("getPoolEthAddress");
            var getPoolChallengeNumber = GetPoolParameter("getChallengeNumber");
            var getPoolMinimumShareDifficulty = GetPoolParameter("getMinimumShareDifficulty", MinerAddress);
            var getPoolMinimumShareTarget = GetPoolParameter("getMinimumShareTarget", MinerAddress);

            bool success = true;
            var startTime = DateTime.Now;
            try
            {
                return MiningParameters.GetPoolMiningParameters(s_PoolURL, getPoolEthAddress, getPoolChallengeNumber,
                                                                getPoolMinimumShareDifficulty, getPoolMinimumShareTarget);
            }
            catch (AggregateException ex)
            {
                success = false;
                m_retryCount++;

                Program.Print("[ERROR] " + ex.Message);
            }
            catch (Exception ex)
            {
                success = false;
                m_retryCount++;

                string errorMsg = ex.Message;
                if (ex.InnerException != null) errorMsg += ("\n " + ex.InnerException.Message);
                Program.Print("[ERROR] " + errorMsg);
            }
            finally
            {
                if (success)
                {
                    m_runFailover = false;
                    var tempLatency = (int)(DateTime.Now - startTime).TotalMilliseconds;
                    try
                    {
                        using (var ping = new Ping())
                        {
                            var poolURL = s_PoolURL.Contains("://") ? s_PoolURL.Split(new string[] { "://" }, StringSplitOptions.None)[1] : s_PoolURL;
                            try
                            {
                                var response = ping.Send(poolURL);
                                if (response.RoundtripTime > 0)
                                    tempLatency = (int)response.RoundtripTime;
                            }
                            catch
                            {
                                try
                                {
                                    poolURL = poolURL.Split('/').First();
                                    var response = ping.Send(poolURL);
                                    if (response.RoundtripTime > 0)
                                        tempLatency = (int)response.RoundtripTime;
                                }
                                catch
                                {
                                    try
                                    {
                                        poolURL = poolURL.Split(':').First();
                                        var response = ping.Send(poolURL);
                                        if (response.RoundtripTime > 0)
                                            tempLatency = (int)response.RoundtripTime;
                                    }
                                    catch { }
                                }
                            }
                        }
                    }
                    catch { }
                    Latency = tempLatency;
                }
            }

            var runFailover = (!success && SecondaryPool != null && m_maxScanRetry > -1 && m_retryCount >= m_maxScanRetry);
            try
            {
                if (runFailover) return SecondaryPool.GetMiningParameters(); 
            }
            finally { m_runFailover = runFailover; }

            return null;
        }

        private void HashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
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
                    OnNewChallenge(this, miningParameters.ChallengeByte32, miningParameters.EthAddress);

                    if (m_challengeReceiveDateTime == DateTime.MinValue)
                        m_challengeReceiveDateTime = DateTime.Now;
                }

                if (m_customDifficulity == 0)
                {
                    if (m_lastParameters == null || miningParameters.MiningTarget.Value != m_lastParameters.MiningTarget.Value)
                    {
                        Program.Print(string.Format("[INFO] New target detected {0}...", miningParameters.MiningTargetByte32String));
                        OnNewTarget(this, miningParameters.MiningTarget);
                    }

                    if (m_lastParameters == null || miningParameters.MiningDifficulty.Value != m_lastParameters.MiningDifficulty.Value)
                    {
                        Program.Print(string.Format("[INFO] New difficulity detected ({0})...", miningParameters.MiningDifficulty.Value));
                        Difficulty = Convert.ToUInt64(miningParameters.MiningDifficulty.Value.ToString());

                        var calculatedTarget = m_maxTarget.Value / Difficulty;
                        if (calculatedTarget != miningParameters.MiningTarget.Value)
                        {
                            Program.Print(string.Format("[INFO] Update target {0}...", calculatedTarget.ToString()));
                            OnNewTarget(this, new HexBigInteger(calculatedTarget));
                        }
                    }
                }
                else
                {
                    Difficulty = m_customDifficulity;
                    var calculatedTarget = m_maxTarget.Value / m_customDifficulity;
                    var newTarget = new HexBigInteger(new BigInteger(m_customDifficulity));

                    OnNewTarget(this, newTarget);
                }

                m_lastParameters = miningParameters;
                OnGetMiningParameterStatus(this, true);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
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
            if (m_maxTarget == null || m_maxTarget.Value == 0 || Difficulty == 0 || hashrate == 0 || m_challengeReceiveDateTime == DateTime.MinValue)
                return TimeSpan.Zero;

            var timeToSolveBlock = new BigInteger(Difficulty) * uint256_MaxValue / m_maxTarget.Value / new BigInteger(hashrate);

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
                hashrate = (ulong)(new BigInteger(Difficulty) * uint256_MaxValue / m_maxTarget.Value / new BigInteger(avgSolveTime));
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

        public bool SubmitSolution(string address, byte[] digest, byte[] challenge, ulong difficulty, byte[] nonce, Miner.IMiner sender)
        {
            m_challengeReceiveDateTime = DateTime.Now;
            var startSubmitDateTime = DateTime.Now;

            if (m_runFailover)
            {
                if (SecondaryPool.SubmitSolution(address, digest, challenge, difficulty, nonce, sender))
                {
                    LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);
                    Program.Print(string.Format("[INFO] Submission roundtrip latency: {0}ms", LastSubmitLatency));

                    if (m_submitDateTimeList.Count >= MAX_SUBMIT_DTM_COUNT) m_submitDateTimeList.RemoveAt(0);
                    m_submitDateTimeList.Add(DateTime.Now);
                }
                return false;
            }

            bool success = false, submitted = false;
            int retryCount = 0, maxRetries = 10;
            var devFee = (ulong)Math.Round(100 / Math.Abs(DevFee.UserPercent));
            do
            {
                try
                {
                    lock (this)
                    {
                        if (SubmittedShares == ulong.MaxValue)
                        {
                            SubmittedShares = 0ul;
                            RejectedShares = 0ul;
                        }
                        var minerAddress = ((SubmittedShares - RejectedShares) % devFee) == 0 ? DevFee.Address : MinerAddress;

                        JObject submitShare = GetPoolParameter("submitShare",
                                                               Utils.Numerics.Byte32ArrayToHexString(nonce),
                                                               minerAddress,
                                                               Utils.Numerics.Byte32ArrayToHexString(digest),
                                                               difficulty.ToString(CultureInfo.InvariantCulture),
                                                               Utils.Numerics.Byte32ArrayToHexString(challenge),
                                                               m_customDifficulity > 0 ? "true" : "false",
                                                               Miner.Work.GetKingAddressString());

                        var response = Utils.Json.InvokeJObjectRPC(s_PoolURL, submitShare);

                        LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);

                        var result = response.SelectToken("$.result")?.Value<string>();

                        success = (result ?? string.Empty).Equals("true", StringComparison.OrdinalIgnoreCase);
                        if (!success) RejectedShares++;
                        SubmittedShares++;

                        Program.Print(string.Format("[INFO] {0} [{1}] submitted to {2} pool: {3} ({4}ms)",
                                                    (minerAddress == DevFee.Address ? "Dev. fee share" : "Miner share"),
                                                    SubmittedShares,
                                                    IsSecondaryPool ? "secondary" : "primary",
                                                    (success ? "success" : "failed"),
                                                    LastSubmitLatency));
#if DEBUG
                        Program.Print(submitShare.ToString());
                        Program.Print(response.ToString());
#endif
                    }
                    submitted = true;
                }
                catch (Exception ex)
                {
                    Program.Print(string.Format("[ERROR] {0}", ex.Message));

                    retryCount += 1;
                    if (retryCount < maxRetries) Task.Delay(500);
                }
            } while (!submitted && retryCount < maxRetries);

            if (success)
            {
                if (m_submitDateTimeList.Count > MAX_SUBMIT_DTM_COUNT) m_submitDateTimeList.RemoveAt(0);
                m_submitDateTimeList.Add(DateTime.Now);
            }
            else UpdateMiningParameters();

            return success;
        }
    }
}
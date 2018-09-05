using Nethereum.Hex.HexTypes;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class PoolInterface : INetworkInterface
    {
        private const int MAX_SUBMIT_DTM_COUNT = 100;
        // Derived from https://github.com/snissn/tokenpool/blob/master/lib/peer-interface.js#L327
        private readonly ulong EFFECTIVE_HASHRATE_CONST = (ulong)Math.Pow(2, 22);
        private readonly List<DateTime> m_submitDateTimeList;

        private readonly string s_MinerAddress;
        private readonly string s_PoolURL;
        private readonly uint m_customDifficulity;
        private readonly HexBigInteger m_maxTarget;
        private readonly int m_maxScanRetry;
        private readonly int m_updateInterval;
        private Timer m_updateMinerTimer;
        private Timer m_hashPrintTimer;
        private bool m_runFailover;
        private int m_retryCount;
        private MiningParameters m_lastParameters;
        private MiningParameters m_cacheParameters;
        
        public event GetMiningParameterStatusEvent OnGetMiningParameterStatusEvent;

        public event NewMessagePrefixEvent OnNewMessagePrefixEvent;

        public event NewTargetEvent OnNewTargetEvent;

        public bool IsPool => true;
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }
        public PoolInterface SecondaryPool { get; }
        public ulong Difficulty { get; private set; }
        public string DifficultyHex { get; private set; }

        public PoolInterface(string minerAddress, string poolURL, int maxScanRetry, int updateInterval, int hashratePrintInterval,
                             uint customDifficulity, HexBigInteger maxTarget, PoolInterface secondaryPool = null)
        {
            m_retryCount = 0;
            m_maxScanRetry = maxScanRetry;
            m_customDifficulity = customDifficulity;
            m_maxTarget = maxTarget;
            m_updateInterval = updateInterval;
            SecondaryPool = secondaryPool;

            if (m_customDifficulity > 0)
                Difficulty = m_customDifficulity;

            s_MinerAddress = minerAddress;
            s_PoolURL = poolURL;
            SubmittedShares = 0ul;
            RejectedShares = 0ul;

            m_submitDateTimeList = new List<DateTime>(MAX_SUBMIT_DTM_COUNT + 1);

            if (hashratePrintInterval > 0)
            {
                m_hashPrintTimer = new Timer(hashratePrintInterval);
                m_hashPrintTimer.Elapsed += m_hashPrintTimer_Elapsed;
                m_hashPrintTimer.Start();
            }
        }

        public void Dispose()
        {
            if (SecondaryPool != null) SecondaryPool.Dispose();
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
            Program.Print("[INFO] Checking latest parameters from pool...");

            var getPoolEthAddress = GetPoolParameter("getPoolEthAddress");
            var getPoolChallengeNumber = GetPoolParameter("getChallengeNumber");
            var getPoolMinimumShareDifficulty = GetPoolParameter("getMinimumShareDifficulty", s_MinerAddress);
            var getPoolMinimumShareTarget = GetPoolParameter("getMinimumShareTarget", s_MinerAddress);

            bool success = true;
            try
            {
                m_cacheParameters = MiningParameters.GetPoolMiningParameters(s_PoolURL, getPoolEthAddress, getPoolChallengeNumber,
                                                                             getPoolMinimumShareDifficulty, getPoolMinimumShareTarget);

                return m_cacheParameters;
            }
            catch (AggregateException ex)
            {
                success = false;
                m_retryCount++;

                string errorMsg = ex.Message;
                foreach (var iEx in ex.InnerExceptions) errorMsg += ("\n " + iEx.Message);

                Program.Print("[ERROR] " + errorMsg);
            }
            catch (Exception ex)
            {
                success = false;
                m_retryCount++;

                string errorMsg = ex.Message;
                if (ex.InnerException != null) errorMsg += ("\n " + ex.InnerException.Message);
                Program.Print("[ERROR] " + errorMsg);
            }

            if (!success && SecondaryPool != null && m_retryCount >= m_maxScanRetry)
            {
                m_runFailover = true;
                Program.Print("[INFO] Checking mining parameters from secondary pool...");
                return SecondaryPool.GetMiningParameters();
            }

            return null;
        }

        private void m_hashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            Program.Print(string.Format("[INFO] Pool Effective Hashrate: {0} MH/s", GetEffectiveHashrate() / 1000000.0f));
        }

        private void m_updateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            try
            {
                var miningParameters = GetMiningParameters();
                if (miningParameters == null)
                {
                    OnGetMiningParameterStatusEvent(this, false, null);
                    return;
                }

                var address = miningParameters.EthAddress;
                var challenge = miningParameters.ChallengeNumberByte32String;
                var target = miningParameters.MiningTargetByte32String;

                if (m_lastParameters == null || miningParameters.ChallengeNumber.Value != m_lastParameters.ChallengeNumber.Value)
                {
                    Program.Print(string.Format("[INFO] New challenge detected {0}...", challenge));
                    OnNewMessagePrefixEvent(this, challenge + address.Replace("0x", string.Empty));
                }

                if (m_customDifficulity == 0)
                {
                    DifficultyHex = miningParameters.MiningDifficulty.HexValue;

                    if (m_lastParameters == null || miningParameters.MiningTarget.Value != m_lastParameters.MiningTarget.Value)
                    {
                        Program.Print(string.Format("[INFO] New target detected {0}...", target));
                        OnNewTargetEvent(this, target);
                    }

                    if (m_lastParameters == null || miningParameters.MiningDifficulty.Value != m_lastParameters.MiningDifficulty.Value)
                    {
                        Program.Print(string.Format("[INFO] New difficulity detected ({0})...", miningParameters.MiningDifficulty.Value));
                        Difficulty = Convert.ToUInt64(miningParameters.MiningDifficulty.Value.ToString());

                        var calculatedTarget = m_maxTarget.Value / Difficulty;
                        if (calculatedTarget != miningParameters.MiningTarget.Value)
                        {
                            var newTarget = calculatedTarget.ToString();
                            Program.Print(string.Format("[INFO] Update target {0}...", newTarget));
                            OnNewTargetEvent(this, newTarget);
                        }
                    }
                }
                else
                {
                    Difficulty = m_customDifficulity;
                    var calculatedTarget = m_maxTarget.Value / m_customDifficulity;
                    var newTarget = new HexBigInteger(new BigInteger(m_customDifficulity)).HexValue;

                    OnNewTargetEvent(this, newTarget);
                }

                m_lastParameters = miningParameters;
                OnGetMiningParameterStatusEvent(this, true, miningParameters);
            }
            catch (Exception ex)
            {
                Program.Print(string.Format("[ERROR] {0}", ex.Message));
            }
        }

        public ulong GetEffectiveHashrate()
        {
            var hashrate = 0ul;

            if (m_submitDateTimeList.Count > 1)
            {
                var avgSolveTime = (ulong)((m_submitDateTimeList.Last() - m_submitDateTimeList.First()).TotalSeconds / m_submitDateTimeList.Count - 1);
                hashrate = Difficulty * EFFECTIVE_HASHRATE_CONST / avgSolveTime;
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
            m_updateMinerTimer_Elapsed(this, null);

            if (m_updateMinerTimer == null && m_updateInterval > 0)
            {
                m_updateMinerTimer = new Timer(m_updateInterval);
                m_updateMinerTimer.Elapsed += m_updateMinerTimer_Elapsed;
                m_updateMinerTimer.Start();
            }
        }

        public bool SubmitSolution(string digest, string fromAddress, string challenge, string difficulty, string target, string solution, Miner.IMiner sender)
        {
            if (string.IsNullOrWhiteSpace(solution) || solution == "0x") return false;

            var startSubmitDateTime = DateTime.Now;

            if (m_runFailover)
            {
                if (SecondaryPool.SubmitSolution(digest, fromAddress, challenge, difficulty, target, solution, sender))
                {
                    var submitDurationMS = (uint)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);
                    Program.Print(string.Format("[INFO] Submission roundtrip latency: {0}ms", submitDurationMS));

                    if (m_submitDateTimeList.Count >= MAX_SUBMIT_DTM_COUNT) m_submitDateTimeList.RemoveAt(0);
                    m_submitDateTimeList.Add(DateTime.Now);
                }
                return false;
            }

            difficulty = new HexBigInteger(difficulty).Value.ToString(); // change from hex to base 10 numerics

            var success = false;
            var submitted = false;
            int retryCount = 0, maxRetries = 10;
            var devFee = (ulong)Math.Round(100 / Math.Abs(DevFee.UserPercent));
            do
            {
                try
                {
                    var poolAddress = fromAddress;
                    lock (this)
                    {
                        if (SubmittedShares == ulong.MaxValue) SubmittedShares = 0u;
                        var minerAddress = ((SubmittedShares) % devFee) == 0 ? DevFee.Address : s_MinerAddress;

                        JObject submitShare;
                        submitShare = GetPoolParameter("submitShare", solution, minerAddress, digest, difficulty, challenge,
                                                       m_customDifficulity > 0 ? "true" : "false", Miner.Work.GetKingAddressString());

                        var response = Utils.Json.InvokeJObjectRPC(s_PoolURL, submitShare);

                        var responseDuration = (uint)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);

                        var result = response.SelectToken("$.result")?.Value<string>();

                        success = (result ?? string.Empty).Equals("true", StringComparison.OrdinalIgnoreCase);
                        if (!success) RejectedShares++;
                        SubmittedShares++;

                        Program.Print(string.Format("[INFO] {0} [{1}] submitted: {2} ({3}ms)",
                                                    (minerAddress == DevFee.Address ? "Dev. fee share" : "Miner share"),
                                                    SubmittedShares,
                                                    (success ? "success" : "failed"),
                                                    responseDuration));
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
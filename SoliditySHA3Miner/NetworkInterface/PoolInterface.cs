using System;
using System.Threading.Tasks;
using Nethereum.Hex.HexTypes;
using Nethereum.Util;
using Newtonsoft.Json.Linq;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class PoolInterface : INetworkInterface
    {
        private readonly string s_MinerAddress;
        private readonly string s_PoolURL;
        private readonly int m_maxScanRetry;
        private bool m_runFailover;
        private int m_retryCount;

        public bool IsPool => true;
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }
        public PoolInterface SecondaryPool { get; }


        public PoolInterface(string minerAddress, string poolURL, int maxScanRetry, PoolInterface secondaryPool = null)
        {
            m_retryCount = 0;
            m_maxScanRetry = maxScanRetry;
            SecondaryPool = secondaryPool;

            s_MinerAddress = minerAddress;
            s_PoolURL = poolURL;
            SubmittedShares = 0ul;
            RejectedShares = 0ul;
        }

        public MiningParameters GetMiningParameters()
        {
            var getPoolEthAddress = GetPoolParameter("getPoolEthAddress");
            var getPoolChallengeNumber = GetPoolParameter("getChallengeNumber");
            var getPoolMinimumShareDifficulty = GetPoolParameter("getMinimumShareDifficulty", s_MinerAddress);
            var getPoolMinimumShareTarget = GetPoolParameter("getMinimumShareTarget", s_MinerAddress);

            bool success = true;
            try
            {
                return MiningParameters.GetPoolMiningParameters(s_PoolURL, getPoolEthAddress, getPoolChallengeNumber,
                                                                getPoolMinimumShareDifficulty, getPoolMinimumShareTarget);
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
                Program.Print("[ERROR] " + errorMsg );
            }

            if (!success && SecondaryPool != null && m_retryCount >= m_maxScanRetry)
            {
                m_runFailover = true;
                Program.Print("[INFO] Getting mining parameters from secondary pool...");
                return SecondaryPool.GetMiningParameters();
            }

            return null;
        }

        void INetworkInterface.SubmitSolution(string digest, string fromAddress, string challenge, string difficulty, string target, string solution, bool isCustomDifficulty)
        {
            if (m_runFailover)
            {
                ((INetworkInterface)SecondaryPool).SubmitSolution(digest, fromAddress, challenge, difficulty, target, solution, isCustomDifficulty);
                return;
            }

            if (string.IsNullOrWhiteSpace(solution) || solution == "0x") return;

            difficulty = new HexBigInteger(difficulty).Value.ToString();

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
                        submitShare = GetPoolParameter("submitShare", solution, minerAddress, digest, difficulty, challenge, isCustomDifficulty ? "true" : "false");

                        var response = Utils.Json.InvokeJObjectRPC(s_PoolURL, submitShare);
                        var result = response.SelectToken("$.result")?.Value<string>();

                        var success = (result ?? string.Empty).Equals("true", StringComparison.OrdinalIgnoreCase);
                        if (!success) RejectedShares++;
                        SubmittedShares++;

                        Program.Print(string.Format("[INFO] {0} [{1}] submitted: {2}", 
                                                    (minerAddress == DevFee.Address ? "Dev. share" : "Share"),
                                                    SubmittedShares, 
                                                    (success ? "success" : "failed")));
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

        public void Dispose()
        {
            if (SecondaryPool != null) SecondaryPool.Dispose();
        }
    }
}

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

using Nethereum.ABI.Model;
using Nethereum.Contracts;
using Nethereum.Hex.HexTypes;
using Nethereum.RPC.Eth.DTOs;
using Nethereum.Util;
using Nethereum.Web3;
using Nethereum.Web3.Accounts;
using Newtonsoft.Json.Linq;
using System;
using System.IO;
using System.Linq;
using System.Net.NetworkInformation;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class Web3Interface : NetworkInterfaceBase
    {
        public HexBigInteger LastSubmitGasPrice { get; private set; }

        private const int MAX_TIMEOUT = 15;

        private readonly Web3 m_web3;
        private readonly Contract m_contract;
        private readonly Account m_account;
        private readonly Function m_mintMethod;
        private readonly Function m_transferMethod;
        private readonly Function m_getMiningDifficulty;
        private readonly Function m_getMiningTarget;
        private readonly Function m_getChallengeNumber;
        private readonly Function m_getMiningReward;
        private readonly Function m_MAXIMUM_TARGET;

        private readonly Function m_CLM_ContractProgress;

        private readonly int m_mintMethodInputParamCount;

        private readonly float m_gasToMine;
        private readonly float m_gasApiMax;
        private readonly ulong m_gasLimit;

        private readonly string m_gasApiURL;
        private readonly string m_gasApiPath;
        private readonly float m_gasApiOffset;
        private readonly float m_gasApiMultiplier;

        private System.Threading.ManualResetEvent m_newChallengeResetEvent;

        #region NetworkInterfaceBase

        public override bool IsPool => false;

        public override event GetMiningParameterStatusEvent OnGetMiningParameterStatus;
        public override event NewChallengeEvent OnNewChallenge;
        public override event NewTargetEvent OnNewTarget;
        public override event NewDifficultyEvent OnNewDifficulty;
        public override event StopSolvingCurrentChallengeEvent OnStopSolvingCurrentChallenge;
        public override event GetTotalHashrateEvent OnGetTotalHashrate;

        public override bool SubmitSolution(string address, byte[] digest, byte[] challenge, HexBigInteger difficulty, byte[] nonce, object sender)
        {
            lock (this)
            {
                if (IsChallengedSubmitted(challenge))
                {
                    Program.Print(string.Format("[INFO] Submission cancelled, nonce has been submitted for the current challenge."));
                    return false;
                }
                m_challengeReceiveDateTime = DateTime.MinValue;

                var transactionID = string.Empty;
                var gasLimit = new HexBigInteger(m_gasLimit);
                var userGas = new HexBigInteger(UnitConversion.Convert.ToWei(new BigDecimal(m_gasToMine), UnitConversion.EthUnit.Gwei));

                if (!string.IsNullOrWhiteSpace(m_gasApiURL))
                {
                    try
                    {
                        var apiGasPrice = Utils.Json.DeserializeFromURL(m_gasApiURL).SelectToken(m_gasApiPath).Value<float>();
                        if (apiGasPrice > 0)
                        {
                            apiGasPrice *= m_gasApiMultiplier;
                            apiGasPrice += m_gasApiOffset;

                            if (apiGasPrice < m_gasToMine)
                            {
                                Program.Print(string.Format("[INFO] Using 'gasToMine' price of {0} GWei, due to lower gas price from API: {1}",
                                                            m_gasToMine, m_gasApiURL));
                            }
                            else if (apiGasPrice > m_gasApiMax)
                            {
                                userGas = new HexBigInteger(UnitConversion.Convert.ToWei(new BigDecimal(m_gasApiMax), UnitConversion.EthUnit.Gwei));
                                Program.Print(string.Format("[INFO] Using 'gasApiMax' price of {0} GWei, due to higher gas price from API: {1}",
                                                            m_gasApiMax, m_gasApiURL));
                            }
                            else
                            {
                                userGas = new HexBigInteger(UnitConversion.Convert.ToWei(new BigDecimal(apiGasPrice), UnitConversion.EthUnit.Gwei));
                                Program.Print(string.Format("[INFO] Using gas price of {0} GWei (after {1} offset) from API: {2}",
                                                            apiGasPrice, m_gasApiOffset, m_gasApiURL));
                            }
                        }
                        else
                        {
                            Program.Print(string.Format("[ERROR] Gas price of 0 GWei was retuned by API: {0}", m_gasApiURL));
                            Program.Print(string.Format("[INFO] Using 'gasToMine' parameter of {0} GWei.", m_gasToMine));
                        }
                    }
                    catch (Exception ex)
                    {
                        HandleException(ex, string.Format("Failed to read gas price from API ({0})", m_gasApiURL));

                        if (LastSubmitGasPrice == null || LastSubmitGasPrice.Value <= 0)
                            Program.Print(string.Format("[INFO] Using 'gasToMine' parameter of {0} GWei.", m_gasToMine));
                        else
                        {
                            Program.Print(string.Format("[INFO] Using last submitted gas price of {0} GWei.",
                                                        UnitConversion.Convert.FromWeiToBigDecimal(LastSubmitGasPrice, UnitConversion.EthUnit.Gwei).ToString()));
                            userGas = LastSubmitGasPrice;
                        }
                    }
                }

                object[] dataInput = null;

                if (m_mintMethodInputParamCount > 1) // 0xBitcoin compatibility
                    dataInput = new object[] { new BigInteger(nonce, isBigEndian: true), digest };

                else // Draft EIP-918 compatibility [2018-03-07]
                    dataInput = new object[] { new BigInteger(nonce, isBigEndian: true) };

                var retryCount = 0;
                var startSubmitDateTime = DateTime.Now;
                do
                {
                    try
                    {
                        var txCount = m_web3.Eth.Transactions.GetTransactionCount.SendRequestAsync(address).Result;

                        // Commented as gas limit is dynamic in between submissions and confirmations
                        //var estimatedGasLimit = m_mintMethod.EstimateGasAsync(from: address,
                        //                                                      gas: gasLimit,
                        //                                                      value: new HexBigInteger(0),
                        //                                                      functionInput: dataInput).Result;

                        var transaction = m_mintMethod.CreateTransactionInput(from: address,
                                                                              gas: gasLimit /*estimatedGasLimit*/,
                                                                              gasPrice: userGas,
                                                                              value: new HexBigInteger(0),
                                                                              functionInput: dataInput);

                        var encodedTx = Web3.OfflineTransactionSigner.SignTransaction(privateKey: m_account.PrivateKey,
                                                                                      to: m_contract.Address,
                                                                                      amount: 0,
                                                                                      nonce: txCount.Value,
                                                                                      gasPrice: userGas,
                                                                                      gasLimit: gasLimit /*estimatedGasLimit*/,
                                                                                      data: transaction.Data);

                        if (!Web3.OfflineTransactionSigner.VerifyTransaction(encodedTx))
                            throw new Exception("Failed to verify transaction.");

                        transactionID = m_web3.Eth.Transactions.SendRawTransaction.SendRequestAsync("0x" + encodedTx).Result;

                        LastSubmitLatency = (int)((DateTime.Now - startSubmitDateTime).TotalMilliseconds);

                        if (!string.IsNullOrWhiteSpace(transactionID))
                        {
                            Program.Print("[INFO] Nonce submitted with transaction ID: " + transactionID);
                            LastSubmitGasPrice = userGas;

                            if (!IsChallengedSubmitted(challenge))
                            {
                                m_submittedChallengeList.Insert(0, challenge.ToArray());
                                if (m_submittedChallengeList.Count > 100) m_submittedChallengeList.Remove(m_submittedChallengeList.Last());
                            }

                            Task.Factory.StartNew(() => GetTransactionReciept(transactionID, address, gasLimit, userGas, LastSubmitLatency, DateTime.Now));

                            if (challenge.SequenceEqual(CurrentChallenge))
                                OnStopSolvingCurrentChallenge(this);
                        }
                    }
                    catch (AggregateException ex)
                    {
                        HandleAggregateException(ex);

                        if (IsChallengedSubmitted(challenge))
                            return false;
                    }
                    catch (Exception ex)
                    {
                        HandleException(ex);

                        if (IsChallengedSubmitted(challenge) || ex.Message == "Failed to verify transaction.")
                            return false;
                    }

                    if (string.IsNullOrWhiteSpace(transactionID))
                    {
                        retryCount++;

                        if (retryCount > 10)
                        {
                            Program.Print("[ERROR] Failed to submit solution for 10 times, submission cancelled.");
                            return false;
                        }
                        else { Task.Delay(m_updateInterval / 2).Wait(); }
                    }
                } while (string.IsNullOrWhiteSpace(transactionID));

                return !string.IsNullOrWhiteSpace(transactionID);
            }
        }

        public override void Dispose()
        {
            base.Dispose();

            if (m_newChallengeResetEvent != null)
                try
                {
                    m_newChallengeResetEvent.Dispose();
                    m_newChallengeResetEvent = null;
                }
                catch { }
            m_newChallengeResetEvent = null;
        }

        protected override void HashPrintTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var totalHashRate = 0ul;
            try
            {
                OnGetTotalHashrate(this, ref totalHashRate);
                Program.Print(string.Format("[INFO] Total Hashrate: {0} MH/s (Effective) / {1} MH/s (Local)",
                                            GetEffectiveHashrate() / 1000000.0f, totalHashRate / 1000000.0f));
            }
            catch (Exception)
            {
                try
                {
                    totalHashRate = GetEffectiveHashrate();
                    Program.Print(string.Format("[INFO] Effective Hashrate: {0} MH/s", totalHashRate / 1000000.0f));
                }
                catch { }
            }
            try
            {
                if (totalHashRate > 0)
                {
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
            }
            catch { }
        }

        protected override void UpdateMinerTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            try
            {
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

                    m_newChallengeResetEvent.Set();
                }

                if (m_lastParameters == null || miningParameters.MiningTarget.Value != m_lastParameters.MiningTarget.Value)
                {
                    Program.Print(string.Format("[INFO] New target detected {0}...", miningParameters.MiningTargetByte32String));
                    OnNewTarget(this, miningParameters.MiningTarget);
                    CurrentTarget = miningParameters.MiningTarget;
                }

                if (m_lastParameters == null || miningParameters.MiningDifficulty.Value != m_lastParameters.MiningDifficulty.Value)
                {
                    Program.Print(string.Format("[INFO] New difficulty detected ({0})...", miningParameters.MiningDifficulty.Value));
                    OnNewDifficulty?.Invoke(this, miningParameters.MiningDifficulty);
                    Difficulty = miningParameters.MiningDifficulty;

                    // Actual difficulty should have decimals
                    var calculatedDifficulty = BigDecimal.Exp(BigInteger.Log(MaxTarget.Value) - BigInteger.Log(miningParameters.MiningTarget.Value));
                    var calculatedDifficultyBigInteger = BigInteger.Parse(calculatedDifficulty.ToString().Split(",.".ToCharArray())[0]);

                    try // Perform rounding
                    {
                        if (uint.Parse(calculatedDifficulty.ToString().Split(",.".ToCharArray())[1].First().ToString()) >= 5)
                            calculatedDifficultyBigInteger++;
                    }
                    catch { }

                    if (Difficulty.Value != calculatedDifficultyBigInteger)
                    {
                        Difficulty = new HexBigInteger(calculatedDifficultyBigInteger);
                        var expValue = BigInteger.Log10(calculatedDifficultyBigInteger);
                        var calculatedTarget = BigInteger.Parse(
                            (BigDecimal.Parse(MaxTarget.Value.ToString()) * BigDecimal.Pow(10, expValue) / (BigDecimal.Parse(calculatedDifficultyBigInteger.ToString()) * BigDecimal.Pow(10, expValue))).
                            ToString().Split(",.".ToCharArray())[0]);
                        var calculatedTargetHex = new HexBigInteger(calculatedTarget);

                        Program.Print(string.Format("[INFO] Update target 0x{0}...", calculatedTarget.ToString("x64")));
                        OnNewTarget(this, calculatedTargetHex);
                        CurrentTarget = calculatedTargetHex;
                    }
                }

                m_lastParameters = miningParameters;
                OnGetMiningParameterStatus(this, true);
            }
            catch (Exception ex)
            {
                HandleException(ex);
            }
        }

        #endregion

        public Web3Interface(string web3ApiPath, string contractAddress, string minerAddress, string privateKey,
                             float gasToMine, string abiFileName, int updateInterval, int hashratePrintInterval,
                             ulong gasLimit, string gasApiURL, string gasApiPath, float gasApiMultiplier, float gasApiOffset, float gasApiMax)
            : base(updateInterval, hashratePrintInterval)
        {
            Nethereum.JsonRpc.Client.ClientBase.ConnectionTimeout = MAX_TIMEOUT * 1000;
            m_newChallengeResetEvent = new System.Threading.ManualResetEvent(false);

            if (string.IsNullOrWhiteSpace(contractAddress))
            {
                Program.Print("[INFO] Contract address not specified, default 0xBTC");
                contractAddress = Config.Defaults.Contract0xBTC_mainnet;
            }

            var addressUtil = new AddressUtil();
            if (!addressUtil.IsValidAddressLength(contractAddress))
                throw new Exception("Invalid contract address provided, ensure address is 42 characters long (including '0x').");

            else if (!addressUtil.IsChecksumAddress(contractAddress))
                throw new Exception("Invalid contract address provided, ensure capitalization is correct.");

            Program.Print("[INFO] Contract address : " + contractAddress);

            if (!string.IsNullOrWhiteSpace(privateKey))
                try
                {
                    m_account = new Account(privateKey);
                    minerAddress = m_account.Address;
                }
                catch (Exception)
                {
                    throw new FormatException("Invalid private key: " + privateKey ?? string.Empty);
                }

            if (!addressUtil.IsValidAddressLength(minerAddress))
            {
                throw new Exception("Invalid miner address provided, ensure address is 42 characters long (including '0x').");
            }
            else if (!addressUtil.IsChecksumAddress(minerAddress))
            {
                throw new Exception("Invalid miner address provided, ensure capitalization is correct.");
            }

            Program.Print("[INFO] Miner's address : " + minerAddress);

            MinerAddress = minerAddress;
            SubmitURL = string.IsNullOrWhiteSpace(web3ApiPath) ? Config.Defaults.InfuraAPI_mainnet : web3ApiPath;

            m_web3 = new Web3(SubmitURL);

            var erc20AbiPath = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), "ERC-20.abi");

            if (!string.IsNullOrWhiteSpace(abiFileName))
                Program.Print(string.Format("[INFO] ABI specified, using \"{0}\"", abiFileName));
            else
            {
                Program.Print("[INFO] ABI not specified, default \"0xBTC.abi\"");
                abiFileName = Config.Defaults.AbiFile0xBTC;
            }
            var tokenAbiPath = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), abiFileName);

            var erc20Abi = JArray.Parse(File.ReadAllText(erc20AbiPath));
            var tokenAbi = JArray.Parse(File.ReadAllText(tokenAbiPath));
            tokenAbi.Merge(erc20Abi, new JsonMergeSettings { MergeArrayHandling = MergeArrayHandling.Union });

            m_contract = m_web3.Eth.GetContract(tokenAbi.ToString(), contractAddress);
            var contractABI = m_contract.ContractBuilder.ContractABI;
            FunctionABI mintABI = null;

            if (string.IsNullOrWhiteSpace(privateKey)) // look for maximum target method only
            {
                if (m_MAXIMUM_TARGET == null)
                {
                    #region ERC918 methods

                    if (contractABI.Functions.Any(f => f.Name == "MAX_TARGET"))
                        m_MAXIMUM_TARGET = m_contract.GetFunction("MAX_TARGET");

                    #endregion

                    #region ABI methods checking

                    if (m_MAXIMUM_TARGET == null)
                    {
                        var maxTargetNames = new string[] { "MAX_TARGET", "MAXIMUM_TARGET", "maxTarget", "maximumTarget" };

                        // ERC541 backwards compatibility
                        if (contractABI.Functions.Any(f => f.Name == "_MAXIMUM_TARGET"))
                        {
                            m_MAXIMUM_TARGET = m_contract.GetFunction("_MAXIMUM_TARGET");
                        }
                        else
                        {
                            var maxTargetABI = contractABI.Functions.
                                                           FirstOrDefault(function =>
                                                           {
                                                               return maxTargetNames.Any(targetName =>
                                                               {
                                                                   return function.Name.IndexOf(targetName, StringComparison.OrdinalIgnoreCase) > -1;
                                                               });
                                                           });
                            if (maxTargetABI == null)
                                m_MAXIMUM_TARGET = null; // Mining still can proceed without MAX_TARGET
                            else
                            {
                                if (!maxTargetABI.OutputParameters.Any())
                                    Program.Print(string.Format("[ERROR] '{0}' function must have output parameter.", maxTargetABI.Name));

                                else if (maxTargetABI.OutputParameters[0].Type != "uint256")
                                    Program.Print(string.Format("[ERROR] '{0}' function output parameter type must be uint256.", maxTargetABI.Name));

                                else
                                    m_MAXIMUM_TARGET = m_contract.GetFunction(maxTargetABI.Name);
                            }
                        }
                    }

                    #endregion
                }
            }
            else
            {
                m_gasToMine = gasToMine;
                Program.Print(string.Format("[INFO] Gas to mine: {0} GWei", m_gasToMine));

                m_gasLimit = gasLimit;
                Program.Print(string.Format("[INFO] Gas limit: {0}", m_gasLimit));

                if (!string.IsNullOrWhiteSpace(gasApiURL))
                {
                    m_gasApiURL = gasApiURL;
                    Program.Print(string.Format("[INFO] Gas API URL: {0}", m_gasApiURL));

                    m_gasApiPath = gasApiPath;
                    Program.Print(string.Format("[INFO] Gas API path: {0}", m_gasApiPath));

                    m_gasApiOffset = gasApiOffset;
                    Program.Print(string.Format("[INFO] Gas API offset: {0}", m_gasApiOffset));

                    m_gasApiMultiplier = gasApiMultiplier;
                    Program.Print(string.Format("[INFO] Gas API multiplier: {0}", m_gasApiMultiplier));

                    m_gasApiMax = gasApiMax;
                    Program.Print(string.Format("[INFO] Gas API maximum: {0} GWei", m_gasApiMax));
                }

                #region ERC20 methods

                m_transferMethod = m_contract.GetFunction("transfer");

                #endregion

                #region ERC918-B methods

                mintABI = contractABI.Functions.FirstOrDefault(f => f.Name == "mint");
                if (mintABI != null) m_mintMethod = m_contract.GetFunction(mintABI.Name);

                if (contractABI.Functions.Any(f => f.Name == "getMiningDifficulty"))
                    m_getMiningDifficulty = m_contract.GetFunction("getMiningDifficulty");

                if (contractABI.Functions.Any(f => f.Name == "getMiningTarget"))
                    m_getMiningTarget = m_contract.GetFunction("getMiningTarget");

                if (contractABI.Functions.Any(f => f.Name == "getChallengeNumber"))
                    m_getChallengeNumber = m_contract.GetFunction("getChallengeNumber");

                if (contractABI.Functions.Any(f => f.Name == "getMiningReward"))
                    m_getMiningReward = m_contract.GetFunction("getMiningReward");

                #endregion

                #region ERC918 methods

                if (contractABI.Functions.Any(f => f.Name == "MAX_TARGET"))
                    m_MAXIMUM_TARGET = m_contract.GetFunction("MAX_TARGET");

                #endregion

                #region CLM MN/POW methods

                if (contractABI.Functions.Any(f => f.Name == "contractProgress"))
                    m_CLM_ContractProgress = m_contract.GetFunction("contractProgress");

                if (m_CLM_ContractProgress != null)
                    m_getMiningReward = null; // Do not start mining if cannot get POW reward value, exception will be thrown later

                #endregion

                #region ABI methods checking

                if (m_mintMethod == null)
                {
                    mintABI = contractABI.Functions.
                                          FirstOrDefault(f => f.Name.IndexOf("mint", StringComparison.OrdinalIgnoreCase) > -1);
                    if (mintABI == null)
                        throw new InvalidOperationException("'mint' function not found, mining cannot proceed.");

                    else if (!mintABI.InputParameters.Any())
                        throw new InvalidOperationException("'mint' function must have input parameter, mining cannot proceed.");

                    else if (mintABI.InputParameters[0].Type != "uint256")
                        throw new InvalidOperationException("'mint' function first input parameter type must be uint256, mining cannot proceed.");

                    m_mintMethod = m_contract.GetFunction(mintABI.Name);
                }

                if (m_getMiningDifficulty == null)
                {
                    var miningDifficultyABI = contractABI.Functions.
                                                          FirstOrDefault(f => f.Name.IndexOf("miningDifficulty", StringComparison.OrdinalIgnoreCase) > -1);
                    if (miningDifficultyABI == null)
                        miningDifficultyABI = contractABI.Functions.
                                                          FirstOrDefault(f => f.Name.IndexOf("mining_difficulty", StringComparison.OrdinalIgnoreCase) > -1);
                    if (miningDifficultyABI == null)
                        throw new InvalidOperationException("'miningDifficulty' function not found, mining cannot proceed.");

                    else if (!miningDifficultyABI.OutputParameters.Any())
                        throw new InvalidOperationException("'miningDifficulty' function must have output parameter, mining cannot proceed.");

                    else if (miningDifficultyABI.OutputParameters[0].Type != "uint256")
                        throw new InvalidOperationException("'miningDifficulty' function output parameter type must be uint256, mining cannot proceed.");

                    m_getMiningDifficulty = m_contract.GetFunction(miningDifficultyABI.Name);
                }

                if (m_getMiningTarget == null)
                {
                    var miningTargetABI = contractABI.Functions.
                                                      FirstOrDefault(f => f.Name.IndexOf("miningTarget", StringComparison.OrdinalIgnoreCase) > -1);
                    if (miningTargetABI == null)
                        miningTargetABI = contractABI.Functions.
                                                      FirstOrDefault(f => f.Name.IndexOf("mining_target", StringComparison.OrdinalIgnoreCase) > -1);
                    if (miningTargetABI == null)
                        throw new InvalidOperationException("'miningTarget' function not found, mining cannot proceed.");

                    else if (!miningTargetABI.OutputParameters.Any())
                        throw new InvalidOperationException("'miningTarget' function must have output parameter, mining cannot proceed.");

                    else if (miningTargetABI.OutputParameters[0].Type != "uint256")
                        throw new InvalidOperationException("'miningTarget' function output parameter type must be uint256, mining cannot proceed.");

                    m_getMiningTarget = m_contract.GetFunction(miningTargetABI.Name);
                }

                if (m_getChallengeNumber == null)
                {
                    var challengeNumberABI = contractABI.Functions.
                                                         FirstOrDefault(f => f.Name.IndexOf("challengeNumber", StringComparison.OrdinalIgnoreCase) > -1);
                    if (challengeNumberABI == null)
                        challengeNumberABI = contractABI.Functions.
                                                         FirstOrDefault(f => f.Name.IndexOf("challenge_number", StringComparison.OrdinalIgnoreCase) > -1);
                    if (challengeNumberABI == null)
                        throw new InvalidOperationException("'challengeNumber' function not found, mining cannot proceed.");

                    else if (!challengeNumberABI.OutputParameters.Any())
                        throw new InvalidOperationException("'challengeNumber' function must have output parameter, mining cannot proceed.");

                    else if (challengeNumberABI.OutputParameters[0].Type != "bytes32")
                        throw new InvalidOperationException("'challengeNumber' function output parameter type must be bytes32, mining cannot proceed.");

                    m_getChallengeNumber = m_contract.GetFunction(challengeNumberABI.Name);
                }

                if (m_getMiningReward == null)
                {
                    var miningRewardABI = contractABI.Functions.
                                                      FirstOrDefault(f => f.Name.IndexOf("miningReward", StringComparison.OrdinalIgnoreCase) > -1);
                    if (miningRewardABI == null)
                        miningRewardABI = contractABI.Functions.
                                                      FirstOrDefault(f => f.Name.IndexOf("mining_reward", StringComparison.OrdinalIgnoreCase) > -1);
                    if (miningRewardABI == null)
                        throw new InvalidOperationException("'miningReward' function not found, mining cannot proceed.");

                    else if (!miningRewardABI.OutputParameters.Any())
                        throw new InvalidOperationException("'miningReward' function must have output parameter, mining cannot proceed.");

                    else if (miningRewardABI.OutputParameters[0].Type != "uint256")
                        throw new InvalidOperationException("'miningReward' function output parameter type must be uint256, mining cannot proceed.");

                    m_getMiningReward = m_contract.GetFunction(miningRewardABI.Name);
                }

                if (m_MAXIMUM_TARGET == null)
                {
                    var maxTargetNames = new string[] { "MAX_TARGET", "MAXIMUM_TARGET", "maxTarget", "maximumTarget" };

                    // ERC541 backwards compatibility
                    if (contractABI.Functions.Any(f => f.Name == "_MAXIMUM_TARGET"))
                    {
                        m_MAXIMUM_TARGET = m_contract.GetFunction("_MAXIMUM_TARGET");
                    }
                    else
                    {
                        var maxTargetABI = contractABI.Functions.
                                                       FirstOrDefault(function =>
                                                       {
                                                           return maxTargetNames.Any(targetName =>
                                                           {
                                                               return function.Name.IndexOf(targetName, StringComparison.OrdinalIgnoreCase) > -1;
                                                           });
                                                       });
                        if (maxTargetABI == null)
                            m_MAXIMUM_TARGET = null; // Mining still can proceed without MAX_TARGET
                        else
                        {
                            if (!maxTargetABI.OutputParameters.Any())
                                Program.Print(string.Format("[ERROR] '{0}' function must have output parameter.", maxTargetABI.Name));

                            else if (maxTargetABI.OutputParameters[0].Type != "uint256")
                                Program.Print(string.Format("[ERROR] '{0}' function output parameter type must be uint256.", maxTargetABI.Name));

                            else
                                m_MAXIMUM_TARGET = m_contract.GetFunction(maxTargetABI.Name);
                        }
                    }
                }

                m_mintMethodInputParamCount = mintABI?.InputParameters.Count() ?? 0;

                #endregion

                if (m_hashPrintTimer != null)
                    m_hashPrintTimer.Start();
            }
        }

        public void OverrideMaxTarget(HexBigInteger maxTarget)
        {
            if (maxTarget.Value > 0u)
            {
                Program.Print("[INFO] Override maximum difficulty: " + maxTarget.HexValue);
                MaxTarget = maxTarget;
            }
            else { MaxTarget = GetMaxTarget(); }
        }

        public HexBigInteger GetMaxTarget()
        {
            if (MaxTarget != null && MaxTarget.Value > 0)
                return MaxTarget;

            Program.Print("[INFO] Checking maximum target from network...");
            while (true)
            {
                try
                {
                    if (m_MAXIMUM_TARGET == null) // assume the same as 0xBTC
                        return new HexBigInteger("0x40000000000000000000000000000000000000000000000000000000000");

                    var maxTarget = new HexBigInteger(m_MAXIMUM_TARGET.CallAsync<BigInteger>().Result);

                    if (maxTarget.Value > 0)
                        return maxTarget;
                    else
                        throw new InvalidOperationException("Network returned maximum target of zero.");
                }
                catch (Exception ex)
                {
                    HandleException(ex, "Failed to get maximum target");
                    Task.Delay(m_updateInterval / 2).Wait();
                }
            }
        }

        private MiningParameters GetMiningParameters()
        {
            Program.Print("[INFO] Checking latest parameters from network...");
            var success = true;
            var startTime = DateTime.Now;
            try
            {
                return MiningParameters.GetSoloMiningParameters(MinerAddress, m_getMiningDifficulty, m_getMiningTarget, m_getChallengeNumber);
            }
            catch (Exception ex)
            {
                HandleException(ex);
                success = false;
                return null;
            }
            finally
            {
                if (success)
                {
                    var tempLatency = (int)(DateTime.Now - startTime).TotalMilliseconds;
                    try
                    {
                        using (var ping = new Ping())
                        {
                            var submitUrl = SubmitURL.Contains("://") ? SubmitURL.Split(new string[] { "://" }, StringSplitOptions.None)[1] : SubmitURL;
                            try
                            {
                                var response = ping.Send(submitUrl);
                                if (response.RoundtripTime > 0)
                                    tempLatency = (int)response.RoundtripTime;
                            }
                            catch
                            {
                                try
                                {
                                    submitUrl = submitUrl.Split('/').First();
                                    var response = ping.Send(submitUrl);
                                    if (response.RoundtripTime > 0)
                                        tempLatency = (int)response.RoundtripTime;
                                }
                                catch
                                {
                                    try
                                    {
                                        submitUrl = submitUrl.Split(':').First();
                                        var response = ping.Send(submitUrl);
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
        }

        private void GetTransactionReciept(string transactionID, string address, HexBigInteger gasLimit, HexBigInteger userGas,
                                           int responseTime, DateTime submitDateTime)
        {
            try
            {
                var success = false;
                var hasWaited = false;
                TransactionReceipt reciept = null;
                while (reciept == null)
                {
                    try
                    {
                        reciept = m_web3.Eth.Transactions.GetTransactionReceipt.SendRequestAsync(transactionID).Result;
                    }
                    catch (AggregateException ex)
                    {
                        HandleAggregateException(ex);
                    }
                    catch (Exception ex)
                    {
                        HandleException(ex);
                    }

                    if (reciept == null)
                    {
                        if (hasWaited) Task.Delay(m_updateInterval).Wait();
                        else
                        {
                            m_newChallengeResetEvent.Reset();
                            m_newChallengeResetEvent.WaitOne(m_updateInterval * 2);
                            hasWaited = true;
                        }
                    }
                }

                success = (reciept.Status.Value == 1);

                if (!success) RejectedShares++;

                if (SubmittedShares == ulong.MaxValue)
                {
                    SubmittedShares = 0ul;
                    RejectedShares = 0ul;
                }
                else SubmittedShares++;

                Program.Print(string.Format("[INFO] Miner share [{0}] submitted: {1} ({2}ms), block: {3}, transaction ID: {4}",
                                            SubmittedShares,
                                            success ? "success" : "failed",
                                            responseTime,
                                            reciept.BlockNumber.Value,
                                            reciept.TransactionHash));

                if (success)
                {
                    if (m_submitDateTimeList.Count >= MAX_SUBMIT_DTM_COUNT)
                        m_submitDateTimeList.RemoveAt(0);

                    m_submitDateTimeList.Add(submitDateTime);

                    var devFee = (ulong)Math.Round(100 / Math.Abs(DevFee.UserPercent));

                    if (((SubmittedShares - RejectedShares) % devFee) == 0)
                        SubmitDevFee(address, gasLimit, userGas, SubmittedShares);
                }

                UpdateMinerTimer_Elapsed(this, null);
            }
            catch (AggregateException ex)
            {
                HandleAggregateException(ex);
            }
            catch (Exception ex)
            {
                HandleException(ex);
            }
        }

        private BigInteger GetMiningReward()
        {
            var failCount = 0;
            Program.Print("[INFO] Checking mining reward amount from network...");
            while (failCount < 10)
            {
                try
                {
                    if (m_CLM_ContractProgress != null)
                        return m_CLM_ContractProgress.CallDeserializingToObjectAsync<CLM_ContractProgress>().Result.PowReward;

                    return m_getMiningReward.CallAsync<BigInteger>().Result; // including decimals
                }
                catch (Exception) { failCount++; }
            }
            throw new Exception("Failed checking mining reward amount.");
        }

        private void SubmitDevFee(string address, HexBigInteger gasLimit, HexBigInteger userGas, ulong shareNo)
        {
            var success = false;
            var devTransactionID = string.Empty;
            TransactionReceipt devReciept = null;
            try
            {
                var miningReward = GetMiningReward();

                Program.Print(string.Format("[INFO] Transferring dev. fee for successful miner share [{0}]...", shareNo));

                var txInput = new object[] { DevFee.Address, miningReward };

                var txCount = m_web3.Eth.Transactions.GetTransactionCount.SendRequestAsync(address).Result;

                // Commented as gas limit is dynamic in between submissions and confirmations
                //var estimatedGasLimit = m_transferMethod.EstimateGasAsync(from: address,
                //                                                          gas: gasLimit,
                //                                                          value: new HexBigInteger(0),
                //                                                          functionInput: txInput).Result;

                var transaction = m_transferMethod.CreateTransactionInput(from: address,
                                                                          gas: gasLimit /*estimatedGasLimit*/,
                                                                          gasPrice: userGas,
                                                                          value: new HexBigInteger(0),
                                                                          functionInput: txInput);

                var encodedTx = Web3.OfflineTransactionSigner.SignTransaction(privateKey: m_account.PrivateKey,
                                                                              to: m_contract.Address,
                                                                              amount: 0,
                                                                              nonce: txCount.Value,
                                                                              gasPrice: userGas,
                                                                              gasLimit: gasLimit /*estimatedGasLimit*/,
                                                                              data: transaction.Data);

                if (!Web3.OfflineTransactionSigner.VerifyTransaction(encodedTx))
                    throw new Exception("Failed to verify transaction.");

                devTransactionID = m_web3.Eth.Transactions.SendRawTransaction.SendRequestAsync("0x" + encodedTx).Result;

                if (string.IsNullOrWhiteSpace(devTransactionID)) throw new Exception("Failed to submit dev fee.");

                while (devReciept == null)
                {
                    try
                    {
                        Task.Delay(m_updateInterval / 2).Wait();
                        devReciept = m_web3.Eth.Transactions.GetTransactionReceipt.SendRequestAsync(devTransactionID).Result;
                    }
                    catch (AggregateException ex)
                    {
                        HandleAggregateException(ex);
                    }
                    catch (Exception ex)
                    {
                        HandleException(ex);
                    }
                }

                success = (devReciept.Status.Value == 1);

                if (!success) throw new Exception("Failed to submit dev fee.");
                else
                {
                    Program.Print(string.Format("[INFO] Transferred dev fee for successful mint share [{0}] : {1}, block: {2}," +
                                                "\n transaction ID: {3}",
                                                shareNo,
                                                success ? "success" : "failed",
                                                devReciept.BlockNumber.Value,
                                                devReciept.TransactionHash));
                }
            }
            catch (AggregateException ex)
            {
                HandleAggregateException(ex);
            }
            catch (Exception ex)
            {
                HandleException(ex);
            }
        }

        private void HandleException(Exception ex, string errorPrefix = null)
        {
            var errorMessage = new StringBuilder("[ERROR] ");

            if (!string.IsNullOrWhiteSpace(errorPrefix))
                errorMessage.AppendFormat("{0}: ", errorPrefix);

            errorMessage.Append(ex.Message);

            var innerEx = ex.InnerException;
            while (innerEx != null)
            {
                errorMessage.AppendFormat("\n {0}", innerEx.Message);
                innerEx = innerEx.InnerException;
            }
            Program.Print(errorMessage.ToString());
        }

        private void HandleAggregateException(AggregateException ex, string errorPrefix = null)
        {
            var errorMessage = new StringBuilder("[ERROR] ");

            if (!string.IsNullOrWhiteSpace(errorPrefix))
                errorMessage.AppendFormat("{0}: ", errorPrefix);

            errorMessage.Append(ex.Message);

            foreach (var innerException in ex.InnerExceptions)
            {
                errorMessage.AppendFormat("\n {0}", innerException.Message);

                var innerEx = ex.InnerException;
                while (innerEx != null)
                {
                    errorMessage.AppendFormat("\n  {0}", innerEx.Message);
                    innerEx = innerEx.InnerException;
                }
            }
            Program.Print(errorMessage.ToString());
        }
    }
}
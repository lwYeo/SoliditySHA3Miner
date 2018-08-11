using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Nethereum.Contracts;
using Nethereum.Hex.HexConvertors.Extensions;
using Nethereum.Hex.HexTypes;
using Nethereum.RPC.Eth.DTOs;
using Nethereum.Util;
using Nethereum.Web3;
using Nethereum.Web3.Accounts;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class Web3Interface : INetworkInterface
    {
        private const int MAX_TIMEOUT = 10;
        private const string DEFAULT_WEB3_API = Defaults.InfuraAPI_mainnet;

        private readonly Web3 m_web3;
        private readonly Contract m_contract;
        private readonly Account m_account;
        private readonly float m_gasToMine;
        private readonly string m_minerAddress;
        private readonly Function m_mintMethod;
        private readonly Function m_transferMethod;
        private readonly List<string> m_submittedChallengeList;
        private readonly int m_updateInterval;

        private readonly object m_cacheParamLock = new object();
        private MiningParameters m_cacheParameters;

        public bool IsPool => false;
        public ulong SubmittedShares { get; private set; }
        public ulong RejectedShares { get; private set; }

        public bool IsChallengedSubmitted(string challenge) => m_submittedChallengeList.Contains(challenge);

        public Web3Interface(string web3ApiPath, string contractAddress, string minerAddress, string privateKey, float gasToMine, string abiFileName, int updateInterval)
        {
            m_updateInterval = updateInterval;
            m_submittedChallengeList = new List<string>();
            Nethereum.JsonRpc.Client.ClientBase.ConnectionTimeout = MAX_TIMEOUT * 1000;

            if (string.IsNullOrWhiteSpace(contractAddress))
            {
                Program.Print("[INFO] Contract address not specified, default 0xBTC");
                contractAddress = Defaults.Contract0xBTC_mainnet;
            }

            var addressUtil = new AddressUtil();
            if (!addressUtil.IsValidAddressLength(contractAddress))
            {
                throw new Exception("Invalid contract address provided, ensure address is 42 characters long (including '0x').");
            }
            else if (!addressUtil.IsChecksumAddress(contractAddress))
            {
                throw new Exception("Invalid contract address provided, ensure capitalization is correct.");
            }

            if (!string.IsNullOrWhiteSpace(privateKey))
            {
                m_account = new Account(privateKey);
                minerAddress = m_account.Address;
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

            m_minerAddress = minerAddress;

            m_web3 = new Web3(string.IsNullOrWhiteSpace(web3ApiPath) ? DEFAULT_WEB3_API : web3ApiPath);

            var abi = File.ReadAllText(Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), abiFileName));

            m_contract = m_web3.Eth.GetContract(abi, contractAddress);

            if (!string.IsNullOrWhiteSpace(privateKey))
            {
                m_mintMethod = m_contract.GetFunction("mint");

                m_transferMethod = m_contract.GetFunction("transfer");

                m_gasToMine = gasToMine;
            }
        }

        public HexBigInteger GetMaxDifficulity()
        {
            Program.Print("[INFO] Getting maximum difficulity from network...");
            while (true)
            {
                try
                {
                    return new HexBigInteger(m_contract.GetFunction("_MAXIMUM_TARGET").CallAsync<BigInteger>().Result);
                }
                catch (AggregateException ex)
                {
                    Program.Print("[ERROR] " + ex.InnerExceptions[0].Message);
                }
            }
        }

        public MiningParameters GetMiningParameters()
        {
            lock (m_cacheParamLock)
            {
                if (m_cacheParameters != null) return m_cacheParameters;
                
                Program.Print("[INFO] Getting latest parameters from network...");

                m_cacheParameters = MiningParameters.GetSoloMiningParameters(m_contract, m_minerAddress);

                Task.Factory.StartNew(() =>
                {
                    Task.Delay(m_updateInterval / 3);
                    m_cacheParameters = null;
                });

                return m_cacheParameters;
            }
        }

        void INetworkInterface.SubmitSolution(string digest, string fromAddress, string challenge, string difficulty, string target, string solution, bool isCustomDifficulty)
        {
            lock (this)
            {
                if (m_submittedChallengeList.Contains(challenge))
                {
                    Program.Print(string.Format("[INFO] Submission cancelled, nonce has been submitted for the current challenge."));
                    return;
                }

                var transactionID = string.Empty;
                var gasLimit = new HexBigInteger(1704624ul);
                var userGas = new HexBigInteger(UnitConversion.Convert.ToWei(new BigDecimal(m_gasToMine), UnitConversion.EthUnit.Gwei));

                var oSolution = new BigInteger(new HexBigInteger(solution).ToHexByteArray().Reverse().ToArray());
                // Note: do not directly use -> new HexBigInteger(solution).Value
                //Because two's complement representation always interprets the highest-order bit of the last byte in the array
                //(the byte at position Array.Length- 1) as the sign bit,
                //the method returns a byte array with an extra element whose value is zero
                //to disambiguate positive values that could otherwise be interpreted as having their sign bits set.

                var dataInput = new object[] { oSolution, HexByteConvertorExtensions.HexToByteArray(digest) };
                
                while (string.IsNullOrWhiteSpace(transactionID))
                {
                    try
                    {
                        System.Threading.Thread.Sleep(1000);

                        var txCount = m_web3.Eth.Transactions.GetTransactionCount.SendRequestAsync(fromAddress).Result;

                        var estimatedGasLimit = m_mintMethod.EstimateGasAsync(from: fromAddress,
                                                                              gas: gasLimit,
                                                                              value: new HexBigInteger(0),
                                                                              functionInput: dataInput).Result;

                        var transaction = m_mintMethod.CreateTransactionInput(from: fromAddress,
                                                                              gas: estimatedGasLimit,
                                                                              gasPrice: userGas,
                                                                              value: new HexBigInteger(0),
                                                                              functionInput: dataInput);

                        var encodedTx = Web3.OfflineTransactionSigner.SignTransaction(privateKey: m_account.PrivateKey,
                                                                                      to: m_contract.Address,
                                                                                      amount: 0,
                                                                                      nonce: txCount.Value,
                                                                                      gasPrice: userGas,
                                                                                      gasLimit: estimatedGasLimit,
                                                                                      data: transaction.Data);

                        if (!Web3.OfflineTransactionSigner.VerifyTransaction(encodedTx))
                            throw new Exception("Failed to verify transaction.");

                        transactionID = m_web3.Eth.Transactions.SendRawTransaction.SendRequestAsync("0x" + encodedTx).Result;
                        
                        if (!string.IsNullOrWhiteSpace(transactionID))
                        {
                            if (!m_submittedChallengeList.Contains(challenge))
                            {
                                m_submittedChallengeList.Insert(0, challenge);
                                if (m_submittedChallengeList.Count > 100) m_submittedChallengeList.Remove(m_submittedChallengeList.Last());
                            }

                            Task.Factory.StartNew(() => GetTransactionReciept(transactionID, fromAddress, gasLimit, userGas));
                        }
                    }
                    catch (AggregateException ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        foreach (var iEx in ex.InnerExceptions)
                            errorMessage += "\n " + iEx.Message;

                        Program.Print(errorMessage);
                        if (m_submittedChallengeList.Contains(challenge)) return;
                    }
                    catch (Exception ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        if (ex.InnerException != null)
                            errorMessage += "\n " + ex.InnerException.Message;

                        Program.Print(errorMessage);
                        if (m_submittedChallengeList.Contains(challenge) || ex.Message == "Failed to verify transaction.") return;
                    }
                }
            }
        }

        public void Dispose()
        {

        }

        private BigInteger GetMiningReward()
        {
            var failCount = 0;
            Program.Print("[INFO] Getting mining reward amount from network...");
            while (failCount < 10)
            {
                try
                {
                    return m_contract.GetFunction("getMiningReward").CallAsync<BigInteger>().Result; // including decimals
                }
                catch (Exception) { failCount++; }
            }
            throw new Exception("Failed getting mining reward amount.");
        }

        private void GetTransactionReciept(string transactionID, string fromAddress, HexBigInteger gasLimit, HexBigInteger userGas)
        {
            try
            {
                var success = false;
                TransactionReceipt reciept = null;
                while (reciept == null)
                {
                    try
                    {
                        System.Threading.Thread.Sleep(3000);
                        reciept = m_web3.Eth.Transactions.GetTransactionReceipt.SendRequestAsync(transactionID).Result;
                    }
                    catch (AggregateException ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        foreach (var iEx in ex.InnerExceptions)
                            errorMessage += "\n " + iEx.Message;

                        Program.Print(errorMessage);
                    }
                    catch (Exception ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        if (ex.InnerException != null)
                            errorMessage += "\n " + ex.InnerException.Message;

                        Program.Print(errorMessage);
                    }
                }

                success = (reciept.Status.Value == 1);

                if (!success) RejectedShares++;
                if (SubmittedShares == ulong.MaxValue) SubmittedShares = 0ul;
                else SubmittedShares++;

                Program.Print(string.Format("[INFO] Miner share [{0}] submitted: {1}, block: {2}," +
                                            "\n transaction ID: {3}",
                                            SubmittedShares,
                                            success ? "success" : "failed",
                                            reciept.BlockNumber.Value,
                                            reciept.TransactionHash));

                if (success)
                {
                    var devFee = (ulong)Math.Round(100 / Math.Abs(DevFee.UserPercent));
                    if (SubmittedShares == ulong.MaxValue) SubmittedShares = 0u;

                    if (((SubmittedShares) % devFee) == 0) SubmitDevFee(fromAddress, gasLimit, userGas, SubmittedShares);
                }
            }
            catch (AggregateException ex)
            {
                var errorMessage = "[ERROR] " + ex.Message;

                foreach (var iEx in ex.InnerExceptions)
                    errorMessage += "\n " + iEx.Message;

                Program.Print(errorMessage);
            }
            catch (Exception ex)
            {
                var errorMessage = "[ERROR] " + ex.Message;

                if (ex.InnerException != null)
                    errorMessage += "\n " + ex.InnerException.Message;

                Program.Print(errorMessage);
            }
        }

        private void SubmitDevFee(string fromAddress, HexBigInteger gasLimit, HexBigInteger userGas, ulong shareNo)
        {
            var success = false;
            var devTransactionID = string.Empty;
            TransactionReceipt devReciept = null;
            try
            {
                var miningReward = GetMiningReward();

                Program.Print(string.Format("[INFO] Transferring dev. fee for successful miner share [{0}]...", shareNo));

                var txInput = new object[] { DevFee.Address, miningReward };

                var txCount = m_web3.Eth.Transactions.GetTransactionCount.SendRequestAsync(fromAddress).Result;

                var estimatedGasLimit = m_transferMethod.EstimateGasAsync(from: fromAddress,
                                                                          gas: gasLimit,
                                                                          value: new HexBigInteger(0),
                                                                          functionInput: txInput).Result;

                var transaction = m_transferMethod.CreateTransactionInput(from: fromAddress,
                                                                          gas: estimatedGasLimit,
                                                                          gasPrice: userGas,
                                                                          value: new HexBigInteger(0),
                                                                          functionInput: txInput);

                var encodedTx = Web3.OfflineTransactionSigner.SignTransaction(privateKey: m_account.PrivateKey,
                                                                              to: m_contract.Address,
                                                                              amount: 0,
                                                                              nonce: txCount.Value,
                                                                              gasPrice: userGas,
                                                                              gasLimit: estimatedGasLimit,
                                                                              data: transaction.Data);

                if (!Web3.OfflineTransactionSigner.VerifyTransaction(encodedTx))
                    throw new Exception("Failed to verify transaction.");

                devTransactionID = m_web3.Eth.Transactions.SendRawTransaction.SendRequestAsync("0x" + encodedTx).Result;

                if (string.IsNullOrWhiteSpace(devTransactionID)) throw new Exception("Failed to submit dev fee.");

                while (devReciept == null)
                {
                    try
                    {
                        System.Threading.Thread.Sleep(3000);
                        devReciept = m_web3.Eth.Transactions.GetTransactionReceipt.SendRequestAsync(devTransactionID).Result;
                    }
                    catch (AggregateException ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        foreach (var iEx in ex.InnerExceptions)
                            errorMessage += "\n " + iEx.Message;

                        Program.Print(errorMessage);
                    }
                    catch (Exception ex)
                    {
                        var errorMessage = "[ERROR] " + ex.Message;

                        if (ex.InnerException != null)
                            errorMessage += "\n " + ex.InnerException.Message;

                        Program.Print(errorMessage);
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
                var errorMessage = "[ERROR] " + ex.Message;

                foreach (var iEx in ex.InnerExceptions)
                    errorMessage += "\n " + iEx.Message;

                Program.Print(errorMessage);
            }
            catch (Exception ex)
            {
                var errorMessage = "[ERROR] " + ex.Message;

                if (ex.InnerException != null)
                    errorMessage += "\n " + ex.InnerException.Message;

                Program.Print(errorMessage);
            }
        }
    }
}

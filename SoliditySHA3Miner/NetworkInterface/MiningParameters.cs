using Nethereum.Contracts;
using Nethereum.Hex.HexConvertors.Extensions;
using Nethereum.Hex.HexTypes;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class MiningParameters
    {
        public HexBigInteger MiningDifficulty { get; private set; }
        public HexBigInteger MiningTarget { get; private set; }
        public HexBigInteger ChallengeNumber { get; private set; }
        public byte[] MiningTargetByte32 { get; private set; }
        public byte[] ChallengeNumberByte32 { get; private set; }
        public string EthAddress { get; private set; }
        public string MiningTargetByte32String { get; private set; }
        public string ChallengeNumberByte32String { get; private set; }

        public static MiningParameters GetSoloMiningParameters(Contract contract, string ethAddress)
        {
            lock (contract) { return new MiningParameters(contract, ethAddress); }
        }

        public static MiningParameters GetPoolMiningParameters(string poolURL,
                                                               JObject getPoolEthAddress,
                                                               JObject getChallengeNumber,
                                                               JObject getMinimumShareDifficulty,
                                                               JObject getMinimumShareTarget)
        {
            return new MiningParameters(poolURL, getPoolEthAddress, getChallengeNumber, getMinimumShareDifficulty, getMinimumShareTarget);
        }

        private MiningParameters(Contract contract, string ethAddress)
        {
            EthAddress = ethAddress;
            bool success = false;
            while (!success)
            {
                var exceptions = new List<System.AggregateException>();
                try
                {
                    Task.WaitAll(new Task[]
                    {
                        Task.Factory.StartNew(() =>
                        {
                            try { MiningDifficulty = new HexBigInteger(contract.GetFunction("getMiningDifficulty").CallAsync<BigInteger>().Result); }
                            catch (System.AggregateException ex)
                            {
                                exceptions.Add(ex);
                                throw ex;
                            }
                        }),
                        Task.Factory.StartNew(() =>
                        {
                            try { MiningTarget = new HexBigInteger(contract.GetFunction("getMiningTarget").CallAsync<BigInteger>().Result); }
                            catch (System.AggregateException ex)
                            {
                                exceptions.Add(ex);
                                throw ex;
                            }
                        }).
                        ContinueWith(task => MiningTargetByte32 = Utils.Numerics.FilterByte32Array(MiningTarget.Value.ToByteArray(littleEndian: false))).
                        ContinueWith(task => MiningTargetByte32String = Utils.Numerics.BigIntegerToByte32HexString(MiningTarget.Value)),
                        Task.Factory.StartNew(() =>
                        {
                            try { ChallengeNumberByte32 = Utils.Numerics.FilterByte32Array(contract.GetFunction("getChallengeNumber").CallAsync<byte[]>().Result); }
                            catch (System.AggregateException ex)
                            {
                                exceptions.Add(ex);
                                throw ex;
                            }
                        }).
                        ContinueWith(Task => ChallengeNumber = new HexBigInteger(HexByteConvertorExtensions.ToHex(ChallengeNumberByte32, prefix: true))).
                        ContinueWith(task => ChallengeNumberByte32String = Utils.Numerics.BigIntegerToByte32HexString(ChallengeNumber.Value))
                    });
                    success = true;
                }
                catch (System.Exception) { exceptions.ForEach(ex => Program.Print(ex.InnerExceptions[0].Message)); }
            }
        }

        private MiningParameters(string poolURL,
                                 JObject getPoolEthAddress,
                                 JObject getChallengeNumber,
                                 JObject getMinimumShareDifficulty,
                                 JObject getMinimumShareTarget)
        {
            Task.WaitAll(new Task[]
            {
                Task.Factory.StartNew(() => EthAddress = Utils.Json.InvokeJObjectRPC(poolURL, getPoolEthAddress).SelectToken("$.result").Value<string>()),
                Task.Factory.StartNew(() => ChallengeNumber = new HexBigInteger(Utils.Json.InvokeJObjectRPC(poolURL, getChallengeNumber).SelectToken("$.result").Value<string>())).
                             ContinueWith(task => ChallengeNumberByte32 = Utils.Numerics.FilterByte32Array(ChallengeNumber.Value.ToByteArray(littleEndian: false))).
                             ContinueWith(task => ChallengeNumberByte32String = Utils.Numerics.BigIntegerToByte32HexString(ChallengeNumber.Value)),
                Task.Factory.StartNew(() => MiningDifficulty = new HexBigInteger(BigInteger.Parse(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareDifficulty).SelectToken("$.result").Value<string>()))),
                Task.Factory.StartNew(() => MiningTarget = new HexBigInteger(BigInteger.Parse(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareTarget).SelectToken("$.result").Value<string>()))).
                             ContinueWith(task => MiningTargetByte32 = Utils.Numerics.FilterByte32Array(MiningTarget.Value.ToByteArray(littleEndian: false))).
                             ContinueWith(task => MiningTargetByte32String = Utils.Numerics.BigIntegerToByte32HexString(MiningTarget.Value))
            });
        }
    }
}
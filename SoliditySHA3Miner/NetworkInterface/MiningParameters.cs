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
                        ContinueWith(task => MiningTargetByte32 = ByteArrToByte32(MiningTarget.Value.ToByteArray(littleEndian: false))).
                        ContinueWith(task => MiningTargetByte32String = BigIntegerToByte32HexString(MiningTarget.Value)),
                        Task.Factory.StartNew(() =>
                        {
                            try { ChallengeNumberByte32 = ByteArrToByte32(contract.GetFunction("getChallengeNumber").CallAsync<byte[]>().Result); }
                            catch (System.AggregateException ex)
                            {
                                exceptions.Add(ex);
                                throw ex;
                            }
                        }).
                        ContinueWith(Task => ChallengeNumber = new HexBigInteger(HexByteConvertorExtensions.ToHex(ChallengeNumberByte32, prefix: true))).
                        ContinueWith(task => ChallengeNumberByte32String = BigIntegerToByte32HexString(ChallengeNumber.Value))
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
                             ContinueWith(task => ChallengeNumberByte32 = ByteArrToByte32(ChallengeNumber.Value.ToByteArray(littleEndian: false))).
                             ContinueWith(task => ChallengeNumberByte32String = BigIntegerToByte32HexString(ChallengeNumber.Value)),
                Task.Factory.StartNew(() => MiningDifficulty = new HexBigInteger(BigInteger.Parse(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareDifficulty).SelectToken("$.result").Value<string>()))),
                Task.Factory.StartNew(() => MiningTarget = new HexBigInteger(BigInteger.Parse(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareTarget).SelectToken("$.result").Value<string>()))).
                             ContinueWith(task => MiningTargetByte32 = ByteArrToByte32(MiningTarget.Value.ToByteArray(littleEndian: false))).
                             ContinueWith(task => MiningTargetByte32String = BigIntegerToByte32HexString(MiningTarget.Value))
            });
        }

        private byte[] ByteArrToByte32(byte[] bytes)
        {
            var outBytes = (byte[])System.Array.CreateInstance(typeof(byte), 32);
            for (int i = 0; i < 32; i++) outBytes[i] = 0;

            for (int i = 31, j = (bytes.Length - 1); i >= 0 && j >= 0; i--, j--) outBytes[i] = bytes[j];
            return outBytes;
        }

        private string BigIntegerToByte32HexString(BigInteger value)
        {
            var tempString = value.ToString("x64");
            var hexString = "0x";
            for (int i = tempString.Length - (32 * 2); i < tempString.Length; i++) hexString += tempString[i];
            return hexString;
        }
    }
}

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

using Nethereum.Contracts;
using Nethereum.Hex.HexConvertors.Extensions;
using Nethereum.Hex.HexTypes;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

namespace SoliditySHA3Miner.NetworkInterface
{
    public class MiningParameters
    {
        public HexBigInteger MiningDifficulty { get; private set; }
        public HexBigInteger MiningTarget { get; private set; }
        public HexBigInteger Challenge { get; private set; }
        public byte[] MiningTargetByte32 { get; private set; }
        public byte[] ChallengeByte32 { get; private set; }
        public string EthAddress { get; private set; }
        public string MiningTargetByte32String { get; private set; }
        public string ChallengeByte32String { get; private set; }

        public static MiningParameters GetSoloMiningParameters(string ethAddress,
                                                               Function getMiningDifficulty,
                                                               Function getMiningTarget,
                                                               Function getChallengeNumber)
        {
            return new MiningParameters(ethAddress, getMiningDifficulty, getMiningTarget, getChallengeNumber);
        }

        public static MiningParameters GetPoolMiningParameters(string poolURL,
                                                               JObject getPoolEthAddress,
                                                               JObject getChallengeNumber,
                                                               JObject getMinimumShareDifficulty,
                                                               JObject getMinimumShareTarget)
        {
            return new MiningParameters(poolURL, getPoolEthAddress, getChallengeNumber, getMinimumShareDifficulty, getMinimumShareTarget);
        }

        private MiningParameters(string ethAddress,
                                 Function getMiningDifficulty,
                                 Function getMiningTarget,
                                 Function getChallengeNumber)
        {
            EthAddress = ethAddress;

            var retryCount = 0;
            var exceptions = new List<Exception>();

            while (retryCount < 10)
            {
                try
                {
                    MiningDifficulty = new HexBigInteger(getMiningDifficulty.CallAsync<BigInteger>().Result);
                    break;
                }
                catch (AggregateException ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex.InnerExceptions[0]);
                    else { Task.Delay(200).Wait(); }
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex);
                    else { Task.Delay(200).Wait(); }
                }
            }

            while (retryCount < 10)
            {
                try
                {
                    MiningTarget = new HexBigInteger(getMiningTarget.CallAsync<BigInteger>().Result);
                    MiningTargetByte32 = Utils.Numerics.FilterByte32Array(MiningTarget.Value.ToByteArray(littleEndian: false));
                    MiningTargetByte32String = Utils.Numerics.Byte32ArrayToHexString(MiningTargetByte32);
                    break;
                }
                catch (AggregateException ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex.InnerExceptions[0]);
                    else { Task.Delay(200).Wait(); }
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex);
                    else { Task.Delay(200).Wait(); }
                }
            }

            while (retryCount < 10)
            {
                try
                {
                    ChallengeByte32 = Utils.Numerics.FilterByte32Array(getChallengeNumber.CallAsync<byte[]>().Result);
                    Challenge = new HexBigInteger(HexByteConvertorExtensions.ToHex(ChallengeByte32, prefix: true));
                    ChallengeByte32String = Utils.Numerics.Byte32ArrayToHexString(ChallengeByte32);
                    break;
                }
                catch (AggregateException ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex.InnerExceptions[0]);
                    else { Task.Delay(200).Wait(); }
                }
                catch (Exception ex)
                {
                    retryCount++;
                    if (retryCount == 10) exceptions.Add(ex);
                    else { Task.Delay(200).Wait(); }
                }
            }

            var exMessage = string.Join(Environment.NewLine, exceptions.Select(ex => ex.Message));
            if (exceptions.Any()) throw new Exception(exMessage);
        }

        private MiningParameters(string poolURL,
                                 JObject getPoolEthAddress,
                                 JObject getChallengeNumber,
                                 JObject getMinimumShareDifficulty,
                                 JObject getMinimumShareTarget)
        {
            try
            {
                EthAddress = Utils.Json.InvokeJObjectRPC(poolURL, getPoolEthAddress).SelectToken("$.result").Value<string>();
            }
            catch (Exception ex)
            {
                throw new OperationCanceledException("Failed to get pool address: " + ex.Message, ex.InnerException);
            }
            try
            {
                Challenge = new HexBigInteger(Utils.Json.InvokeJObjectRPC(poolURL, getChallengeNumber).SelectToken("$.result").Value<string>());
                ChallengeByte32 = Utils.Numerics.FilterByte32Array(Challenge.Value.ToByteArray(littleEndian: false));
                ChallengeByte32String = Utils.Numerics.Byte32ArrayToHexString(ChallengeByte32);
            }
            catch (Exception ex)
            {
                throw new OperationCanceledException("Failed to get pool challenge: " + ex.Message, ex.InnerException);
            }
            try
            {
                MiningDifficulty = new HexBigInteger(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareDifficulty).SelectToken("$.result").Value<ulong>());
            }
            catch (Exception ex)
            {
                throw new OperationCanceledException("Failed to get pool difficulty: " + ex.Message, ex.InnerException);
            }
            try
            {
                MiningTarget = new HexBigInteger(BigInteger.Parse(Utils.Json.InvokeJObjectRPC(poolURL, getMinimumShareTarget).SelectToken("$.result").Value<string>()));
                MiningTargetByte32 = Utils.Numerics.FilterByte32Array(MiningTarget.Value.ToByteArray(littleEndian: false));
                MiningTargetByte32String = Utils.Numerics.Byte32ArrayToHexString(MiningTargetByte32);
            }
            catch (Exception ex)
            {
                throw new OperationCanceledException("Failed to get pool target: " + ex.Message, ex.InnerException);
            }
        }
    }
}
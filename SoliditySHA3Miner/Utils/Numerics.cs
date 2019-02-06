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

using Nethereum.Util;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SoliditySHA3Miner.Utils
{
    public static class Numerics
    {
        private static readonly AddressUtil m_addressUtil = new AddressUtil();
        private static readonly object m_Byte32ArrayToHexStringLock = new object();
        private static readonly object m_HexStringToByte32ArrayLock = new object();
        private static readonly object m_FilterByte32ArrayLock = new object();
        private static readonly object m_HexStringToByte32Array = new object();
        private static readonly object m_BigIntegerToByte32HexString = new object();

        private static readonly string[] ASCII = new string[]
        {
            "00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f",
            "10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f",
            "20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f",
            "30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f",
            "40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f",
            "50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f",
            "60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f",
            "70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f",
            "80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f",
            "90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f",
            "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af",
            "b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf",
            "c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf",
            "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df",
            "e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef",
            "f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"
        };

        public static string Byte32ArrayToHexString(byte[] byte32, bool prefix = true)
        {
            lock (m_Byte32ArrayToHexStringLock)
            {
                var byte32String = prefix ? "0x" : string.Empty;

                if (byte32.Length != Miner.MinerBase.UINT256_LENGTH)
                    byte32 = FilterByte32Array(byte32);

                for (var i = 0; i < Miner.MinerBase.UINT256_LENGTH; i++)
                    byte32String += ASCII[byte32[i]];

                return byte32String;
            }
        }

        public static void HexStringToByte32Array(string byte32String, ref byte[] byte32)
        {
            lock (m_HexStringToByte32ArrayLock)
            {
                var bytePosition = -1;
                var byte32Length = Miner.MinerBase.UINT256_LENGTH * 2;

                for (var i = 0; i < byte32Length; i += 2)
                {
                    var valueStr = string.Concat(byte32String.Skip(i).Take(2));
                    var value = (byte)Array.IndexOf(ASCII, valueStr);

                    bytePosition++;
                    byte32[bytePosition] = value;
                }
            }
        }

        public static byte[] HexStringToByte32Array(string byte32String)
        {
            lock (m_HexStringToByte32Array)
            {
                var byte32Length = Miner.MinerBase.UINT256_LENGTH * 2;
                byte32String = (byte32String ?? string.Empty).Replace("0x", string.Empty);

                if (byte32String.Length != byte32Length)
                    throw new InvalidOperationException(string.Format("Invalid byte32 length of {0}.", byte32String.Length));

                var byte32 = (byte[])Array.CreateInstance(typeof(byte), Miner.MinerBase.UINT256_LENGTH);
                HexStringToByte32Array(byte32String, ref byte32);

                return byte32;
            }
        }
        
        public static byte[] FilterByte32Array(IEnumerable<byte> bytes)
        {
            lock (m_FilterByte32ArrayLock)
            {
                var outBytes = (byte[])Array.CreateInstance(typeof(byte), Miner.MinerBase.UINT256_LENGTH);

                for (int i = 0; i < Miner.MinerBase.UINT256_LENGTH; i++)
                    outBytes[i] = 0;

                for (int i = Miner.MinerBase.UINT256_LENGTH - 1, j = (bytes.Count() - 1); i >= 0 && j >= 0; i--, j--)
                    outBytes[i] = bytes.ElementAt(j);

                return outBytes;
            }
        }

        public static void AddressStringToByte20Array(string addressString, ref byte[] address, string type = null, bool isChecksum = true)
        {
            if (!m_addressUtil.IsValidAddressLength(addressString))
                throw new Exception(string.Format("Invalid {0} provided, ensure address is 42 characters long (including '0x').",
                                                  string.IsNullOrWhiteSpace(type) ? "address" : type));

            else if (isChecksum && !m_addressUtil.IsChecksumAddress(addressString))
                throw new Exception(string.Format("Invalid {0} provided, ensure capitalization is correct.",
                                                  string.IsNullOrWhiteSpace(type) ? "address" : type));

            var bytePosition = -1;
            var addressLength = Miner.MinerBase.ADDRESS_LENGTH * 2;
            var addressLowerCase = addressString.Replace("0x", string.Empty).ToLowerInvariant();

            for (var i = 0; i < addressLength; i += 2)
            {
                var valueStr = string.Concat(addressLowerCase.Skip(i).Take(2));
                var value = (byte)Array.IndexOf(ASCII, valueStr);

                bytePosition++;
                address[bytePosition] = value;
            }
        }

        public static string Byte20ArrayToAddressString(byte[] address, bool prefix = true)
        {
            var addressString = prefix ? "0x" : string.Empty;

            if (address.Length != Miner.MinerBase.ADDRESS_LENGTH)
                address = FilterByte20Array(address);

            for (var i = 0; i < Miner.MinerBase.ADDRESS_LENGTH; i++)
                addressString += ASCII[address[i]];
            
            return addressString;
        }

        public static byte[] FilterByte20Array(IEnumerable<byte> bytes)
        {
            var outBytes = (byte[])Array.CreateInstance(typeof(byte), Miner.MinerBase.ADDRESS_LENGTH);

            for (int i = 0; i < Miner.MinerBase.ADDRESS_LENGTH; i++)
                outBytes[i] = 0;

            for (int i = Miner.MinerBase.ADDRESS_LENGTH - 1, j = (bytes.Count() - 1); i >= 0 && j >= 0; i--, j--)
                outBytes[i] = bytes.ElementAt(j);

            return outBytes;
        }
    }
}
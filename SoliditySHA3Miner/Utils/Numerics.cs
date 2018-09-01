using Nethereum.Hex.HexTypes;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SoliditySHA3Miner.Utils
{
    public static class Numerics
    {
        private static readonly object m_FilterByte32ArrayLock = new object();
        private static readonly object m_HexStringToByte32Array = new object();
        private static readonly object m_BigIntegerToByte32HexString = new object();

        public static byte[] FilterByte32Array(IEnumerable<byte> bytes)
        {
            lock (m_FilterByte32ArrayLock)
            {
                var outBytes = (byte[])System.Array.CreateInstance(typeof(byte), 32);

                for (int i = 0; i < 32; i++)
                    outBytes[i] = 0;

                for (int i = 31, j = (bytes.Count() - 1); i >= 0 && j >= 0; i--, j--)
                    outBytes[i] = bytes.ElementAt(j);

                return outBytes;
            }
        }

        public static byte[] HexStringToByte32Array(string hexString)
        {
            lock (m_HexStringToByte32Array)
            {
                return FilterByte32Array(new HexBigInteger(hexString).ToHexByteArray().Reverse());
            }
        }

        public static string BigIntegerToByte32HexString(BigInteger value)
        {
            lock (m_BigIntegerToByte32HexString)
            {
                string tempString = value.ToString("x64"), hexString = "0x";

                for (int i = tempString.Length - 64; i < tempString.Length; i++)
                    hexString += tempString[i];

                return hexString;
            }
        }
    }
}
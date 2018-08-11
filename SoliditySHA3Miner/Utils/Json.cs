using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;

namespace SoliditySHA3Miner.Utils
{
    static class Json
    {
        private const int MAX_TIMEOUT = 10;
        private static object _Object1 = new object();
        private static object _Object2 = new object();
        private static object _Object3 = new object();
        private static object _Object4 = new object();
        private static object _Object5 = new object();

        public static readonly JsonSerializerSettings BaseClassFirstSettings =
            new JsonSerializerSettings { ContractResolver = BaseFirstContractResolver.Instance };

        public static string SerializeFromObject(object obj, JsonSerializerSettings settings = null)
        {
            lock (_Object1)
            {
                try
                {
                    return (settings == null) ?
                        JsonConvert.SerializeObject(obj, Formatting.Indented) :
                        JsonConvert.SerializeObject(obj, Formatting.Indented, settings);
                }
                catch { }
                return string.Empty;
            }
        }

        public static JObject DeserializeFromURL(string url)
        {
            lock (_Object2)
            {
                string sJSON = string.Empty;
                JObject jObject = null;
                try
                {
                    using (var oClient = new HttpClient())
                    {
                        oClient.Timeout = new TimeSpan(0, 0, MAX_TIMEOUT);
                        using (HttpResponseMessage oResponse = oClient.GetAsync(url).Result)
                        {
                            using (HttpContent oContent = oResponse.Content)
                            {
                                sJSON = oContent.ReadAsStringAsync().Result;
                            }
                        }
                    }
                    jObject = (JObject)JsonConvert.DeserializeObject(sJSON);
                }
                catch { }
                return jObject;
            }
        }

        public static T DeserializeFromURL<T>(string url)
        {
            lock (_Object3)
            {
                string sJSON = string.Empty;
                var jObject = (T)Activator.CreateInstance(typeof(T));
                try
                {
                    using (var oClient = new HttpClient()) 
                    {
                        oClient.Timeout = new TimeSpan(0, 0, MAX_TIMEOUT);
                        using (HttpResponseMessage oResponse = oClient.GetAsync(url).Result)
                        {
                            using (HttpContent oContent = oResponse.Content)
                            {
                                sJSON = oContent.ReadAsStringAsync().Result;
                            }
                        }
                    }
                    jObject = JsonConvert.DeserializeObject<T>(sJSON);
                }
                catch { }
                return jObject;
            }
        }

        public static JObject InvokeJObjectRPC(string url, JObject obj, JsonSerializerSettings settings = null)
        {
            lock (_Object5)
            {
                var serializedJSON = SerializeFromObject(obj, settings);

                var httpRequest = (HttpWebRequest)WebRequest.Create(url);
                httpRequest.ReadWriteTimeout = MAX_TIMEOUT * 1000;
                httpRequest.ContentLength = serializedJSON.Length;
                httpRequest.ContentType = "application/json-rpc";
                httpRequest.Method = "POST";

                using (var streamWriter = new StreamWriter(httpRequest.GetRequestStream()))
                {
                    streamWriter.Write(serializedJSON);
                    streamWriter.Flush();
                    streamWriter.Close();
                }

                var httpResponse = (HttpWebResponse)httpRequest.GetResponse();
                using (var streamReader = new StreamReader(httpResponse.GetResponseStream()))
                {
                    var sResponse = streamReader.ReadToEnd();
                    var jResponse = JsonConvert.DeserializeObject<JObject>(sResponse);
                    return jResponse;
                }
            }
        }

        public static T CloneObject<T>(T objectToClone)
        {
            lock (_Object4)
            {
                if (objectToClone == null) { return default(T); }
                try
                {
                    return JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(objectToClone),
                                                            new JsonSerializerSettings() { ObjectCreationHandling = ObjectCreationHandling.Replace });

                }
                catch { return default(T); }
            }
        }

        public class ClassNameContractResolver : DefaultContractResolver
        {
            private ClassNameContractResolver() { }

            static ClassNameContractResolver() => Instance = new ClassNameContractResolver();

            public static ClassNameContractResolver Instance { get; private set; }

            protected override IList<JsonProperty> CreateProperties(Type type, MemberSerialization memberSerialization)
            {
                return base.CreateProperties(type, memberSerialization).
                            Select(p =>
                            {
                                p.PropertyName = p.UnderlyingName;
                                return p;
                            }).
                            ToList();
            }
        }

        public class BaseFirstContractResolver : DefaultContractResolver
        {
            private BaseFirstContractResolver() { }

            static BaseFirstContractResolver() => Instance = new BaseFirstContractResolver();

            public static BaseFirstContractResolver Instance { get; private set; }

            protected override IList<JsonProperty> CreateProperties(Type type, MemberSerialization memberSerialization)
            {
                return base.CreateProperties(type, memberSerialization).
                            OrderBy(p => p.DeclaringType.BaseTypesAndSelf().Count()).
                            ToList();
            }
        }
    }

    public static class TypeExtensions
    {
        public static IEnumerable<Type> BaseTypesAndSelf(this Type type)
        {
            while (type != null)
            {
                yield return type;
                type = type.BaseType;
            }
        }
    }
}

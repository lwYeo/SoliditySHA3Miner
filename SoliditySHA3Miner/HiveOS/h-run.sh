#!/usr/bin/env bash

cd `dirname $0`

[ -t 1 ] && . colors

. h-manifest.conf

[[ -z $CUSTOM_LOG_BASENAME ]] && echo -e "${RED}No CUSTOM_LOG_BASENAME is set${NOCOLOR}" && exit 1
[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME is set${NOCOLOR}" && exit 1
[[ ! -f $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}Custom config ${YELLOW}$CUSTOM_CONFIG_FILENAME${RED} is not found${NOCOLOR}" && exit 1
CUSTOM_LOG_BASEDIR=`dirname "$CUSTOM_LOG_BASENAME"`
[[ ! -d $CUSTOM_LOG_BASEDIR ]] && mkdir -p $CUSTOM_LOG_BASEDIR

function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

function install_deps() {
  . install-deps.sh

  # below is the temp fix for the following issue as of version 0.5-74, not required from 0.5-75 onwards.
  # /usr/lib/x86_64-linux-gnu/libcurl.so.4: version `CURL_OPENSSL_4' not found (required by curl)
  #if [[ $(lsb_release -sr) == "18.04" ]]; then
  #  apt-get install libtool m4 automake -y
  #  git clone https://github.com/curl/curl.git
  #  cd curl
  #  ./buildconf
  #  ./configure --disable-shared
  #  make
  #  make install
  #  cd ..
  #  rm -r -f curl
  #fi
}

command -v dotnet >/dev/null 2>&1 || {
  echo "dotnet not found, running install_deps..."
  install_deps
}

dotnetVersion=$(dotnet --info | grep "Version")
dotnetVersion=${dotnetVersion#*: }

if version_gt "2.2.0" $dotnetVersion; then
  echo "Found older version of dotnet, running install_deps..."
  install_deps
fi

depsFound=false
dotnetRuntimeVersion=$(dotnet --list-runtimes)

if [[ -z $dotnetRuntimeVersion ]]; then
 echo "dotnet runtime not found, running install_deps..."
 install_deps

else
 while read line || [ -n "$line" ]; do
  line=${line#*App }
  line=${line% [*}

  if version_gt $line "2.2.0"; then
   depsFound=true
  fi
 done < <(printf %s "$dotnetRuntimeVersion")

 if ! $depsFound; then
  echo "Found older version of dotnet runtime, running install_deps..."
  install_deps
 fi
fi

#echo "dotnet ./SoliditySHA3Miner.dll $CUSTOM_USER_CONFIG 2>&1 | tee $CUSTOM_LOG_BASENAME.log"
dotnet ./SoliditySHA3Miner.dll $(< ${CUSTOM_CONFIG_FILENAME}.param) 2>&1 | tee $CUSTOM_LOG_BASENAME.log

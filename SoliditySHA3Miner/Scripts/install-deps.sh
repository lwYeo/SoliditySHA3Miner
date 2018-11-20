#!/usr/bin/env bash

rm -f packages-microsoft-prod.deb

version=$(lsb_release -sr)

case $version in
14.04)
    wget https://packages.microsoft.com/config/ubuntu/14.04/packages-microsoft-prod.deb
    ;;
16.04)
    wget https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb
    ;;
17.10)
    wget https://packages.microsoft.com/config/ubuntu/17.10/packages-microsoft-prod.deb
    ;;
18.04)
    wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb
    ;;
*)
    echo "Unrecognized version"
    wget https://packages.microsoft.com/config/ubuntu/14.04/packages-microsoft-prod.deb
    ;;
esac

sudo dpkg -P packages-microsoft-prod
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update

rm -f packages-microsoft-prod.deb

sudo apt-get install apt-transport-https -y
sudo apt-get install dotnet-runtime-2.1 -y

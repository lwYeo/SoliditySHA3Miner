#!/usr/bin/env bash

rm -f packages-microsoft-prod.deb

wget https://packages.microsoft.com/config/ubuntu/14.04/packages-microsoft-prod.deb

sudo dpkg -P packages-microsoft-prod
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get-ubuntu update

rm -f packages-microsoft-prod.deb

sudo apt-get-ubuntu install apt-transport-https -y
sudo apt-get-ubuntu install dotnet-runtime-2.2 -y

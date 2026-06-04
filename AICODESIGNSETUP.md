## Setup Linux Machine (Ubuntu-22.04)

```bash
cd ~
sudo apt update && sudo apt install build-essential python3-dev unzip
```

## Setup Conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -yn executorch python=3.10
conda activate executorch
```

## Setup QNN SDK

```bash
cd ~
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip
unzip v2.37.0.250724.zip
mv qairt/2.37.0.250724 qnn-sdk
rm -r qairt
rm v2.37.0.250724.zip

source qnn-sdk/bin/envsetup.sh
sudo ${QAIRT_SDK_ROOT}/bin/check-linux-dependency.sh
${QAIRT_SDK_ROOT}/bin/envcheck -c
pip install "setuptools<82"
python "${QAIRT_SDK_ROOT}/bin/check-python-dependency"
```

## Setup Hexagon NPU SDK

Install QPM3 for Linux via https://qpm.qualcomm.com/#/main/tools/details/QPM3?version=3.0.131.7

```bash
qpm-cli --license-activate hexagonsdk6.x
qpm-cli --install hexagonsdk6.x --version 6.6.0.0 --path ~/hexagon-sdk-6.6.0
```

## Setup Android NDK

```bash
wget https://dl.google.com/android/repository/android-ndk-r27d-linux.zip
unzip android-ndk-r27d-linux.zip
mv android-ndk-r27d android-ndk
rm android-ndk-r27d-linux.zip
```

## Setup .bashrc (Optional)

Setup the environment variables.

Append the following to `~/.bashrc` and run `source ~/.bashrc`

```bash
export EXECUTORCH_ROOT=$HOME/executorch

export QNN_SDK_ROOT=$HOME/qnn-sdk
export HEXAGON_SDK_ROOT=$HOME/hexagon-sdk-6.0.0
source $QNN_SDK_ROOT/bin/envsetup.sh

export ANDROID_NDK_ROOT=$HOME/android-ndk
export PATH=$PATH:$ANDROID_NDK_ROOT

export PYTHONPATH=$EXECUTORCH_ROOT/..:$PYTHONPATH

conda activate executorch
```

## Setup Executorch

> You need around 20GB of RAM (Mem + Swap) to build

```bash
cd ~
git clone https://github.com/MystericalTH/executorch.git
cd executorch
./install_executorch.sh
./backends/qualcomm/scripts/build.sh
```

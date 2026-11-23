#!/usr/bin/env bash
# Local CUDA build of strata-poly, with the MSVC environment set by hand (vcvars64.bat needs reg.exe, which the
# sandbox blocks).  Used only to test this branch on the RTX 5060 laptop.
set -e
unset CODEBUDDY_MCP_CONFIG CODEBUDDY_SKILL_DESC_MAX_OVERRIDES BAGGAGE CODEBUDDY_APPROVAL_MEMORY_GROUPS
export MSVC="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
export VCINSTALLDIR="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC"
export VCToolsInstallDir="$MSVC/"
export VSINSTALLDIR="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"
export WindowsSdkDir="C:/Program Files (x86)/Windows Kits/10"
export WindowsSDKVersion="10.0.26100.0/"
export UCRTVersion="10.0.26100.0"
export INCLUDE="$(cygpath -w "$MSVC/include");C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.26100.0\\ucrt;C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.26100.0\\shared;C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.26100.0\\um"
export LIB="$(cygpath -w "$MSVC/lib/x64");C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.26100.0\\ucrt\\x64;C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.26100.0\\um\\x64"
export PATH="$MSVC/bin/Hostx64/x64:/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64:/c/Users/zshua/anaconda3/Scripts:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin:$PATH"
cd /c/Users/zshua/PolyStrata
cmake -G Ninja -DCMAKE_MAKE_PROGRAM=/c/Users/zshua/anaconda3/Scripts/ninja.exe -S . -B build-poly \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe" \
  "-DSTRATA_GGML_DIR=C:/Users/zshua/PolyStrata/third_party/llama.cpp" -DSTRATA_PORTABLE=OFF
cmake --build build-poly --target strata-poly poly_inkling2_test -j "$(nproc)"
echo BUILD_OK

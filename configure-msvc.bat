@echo off
set "INCLUDE=D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\winrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\cppwinrt"
set "LIB=D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
set "PATH=D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\Program Files\CMake\bin;C:\Users\zhuan\AppData\Local\Programs\Python\Python313\Scripts;C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%"
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
cd /d I:\llama\kvmem-llama.cpp-0.17.0\build-cuda13.4-dll
cmake -S I:\llama\kvmem-llama.cpp-0.17.0 -B I:\llama\kvmem-llama.cpp-0.17.0\build-cuda13.4-dll -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_HOST_COMPILER=cl -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON -DKVMEM_ENABLE_NVME=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
exit /b %ERRORLEVEL%

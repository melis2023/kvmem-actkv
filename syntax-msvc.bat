@echo off
set "INCLUDE=D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\winrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\cppwinrt"
set "LIB=D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
set "PATH=D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\Program Files\CMake\bin;C:\Users\zhuan\AppData\Local\Programs\Python\Python313\Scripts;C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%"
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
cd /d I:\llama\kvmem-llama.cpp-0.17.0\build-cuda13.4-dll
cl.exe /nologo /TP /Zs /std:c++17 /utf-8 /bigobj /D_CRT_SECURE_NO_WARNINGS /DGGML_USE_CPU /DGGML_USE_CUDA /DKVMEM_ENABLE_NVME=0 /DLLAMA_KVMEM=1 /II:\llama\kvmem-llama.cpp-0.17.0\llama.cpp\src /II:\llama\kvmem-llama.cpp-0.17.0\llama.cpp\include /II:\llama\kvmem-llama.cpp-0.17.0\llama.cpp\ggml\include /II:\llama\kvmem-llama.cpp-0.17.0\llama.cpp\ggml\src /II:\llama\kvmem-llama.cpp-0.17.0\llama.cpp\common /II:\llama\kvmem-llama.cpp-0.17.0\kvmem\include /II:\llama\kvmem-llama.cpp-0.17.0\src\adapter /external:I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\include" /external:W0 %*
exit /b %ERRORLEVEL%

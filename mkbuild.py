"""Generate the MSVC/Ninja helper .bat files for this checkout.

The 0.17.0 source drop ships only the Linux/ROCm entry points, so the Windows
toolchain environment is written out explicitly here.  Do NOT call
vcvars64.bat: it invokes reg.exe (blocked by the sandbox security policy) and
injects duplicate-cased proxy variables that crash the MSBuild environment
dictionary.

Usage:  python mkbuild.py
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROOT_WIN = ROOT.replace("/", "\\")
BUILD_DIR = os.environ.get("KVMEM_BUILD_DIR") or (ROOT_WIN + r"\build-cuda13.4-dll")

VS = r"D:\Program Files\Microsoft Visual Studio\2022\Professional"
MSVC = VS + r"\VC\Tools\MSVC\14.44.35207"
SDK = r"C:\Program Files (x86)\Windows Kits\10"
SDKV = "10.0.26100.0"
CUDA = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"

inc = ";".join([
    MSVC + r"\include",
    SDK + r"\Include" + "\\" + SDKV + r"\ucrt",
    SDK + r"\Include" + "\\" + SDKV + r"\shared",
    SDK + r"\Include" + "\\" + SDKV + r"\um",
    SDK + r"\Include" + "\\" + SDKV + r"\winrt",
    SDK + r"\Include" + "\\" + SDKV + r"\cppwinrt",
])
lib = ";".join([
    MSVC + r"\lib\x64",
    SDK + r"\Lib" + "\\" + SDKV + r"\ucrt\x64",
    SDK + r"\Lib" + "\\" + SDKV + r"\um\x64",
])
path = ";".join([
    MSVC + r"\bin\Hostx64\x64",
    CUDA + r"\bin",
    r"C:\Program Files\CMake\bin",
    r"C:\Users\zhuan\AppData\Local\Programs\Python\Python313\Scripts",
    SDK + r"\bin" + "\\" + SDKV + r"\x64",
    "%PATH%",
])

lines = [
    "@echo off",
    'set "INCLUDE=%s"' % inc,
    'set "LIB=%s"' % lib,
    'set "PATH=%s"' % path,
    "set HTTP_PROXY=",
    "set HTTPS_PROXY=",
    "set http_proxy=",
    "set https_proxy=",
    "cd /d %s" % BUILD_DIR,
]


def write(name, body):
    p = os.path.join(ROOT, name)
    with open(p, "w", newline="\r\n") as f:
        f.write("\r\n".join(body) + "\r\n")
    print("wrote", p)


write("env-msvc.bat", lines)

# Pin the same options the shipped Windows CUDA build used, so a rebuild is
# comparable with the binaries this checkout is replacing.
write("configure-msvc.bat", lines + [
    "cmake -S " + ROOT_WIN + " -B " + BUILD_DIR + " -G Ninja "
    "-DCMAKE_BUILD_TYPE=Release "
    "-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl "
    "-DCMAKE_CUDA_HOST_COMPILER=cl "
    "-DCMAKE_CUDA_ARCHITECTURES=120a-real "
    "-DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON "
    "-DKVMEM_ENABLE_NVME=OFF "
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "exit /b %ERRORLEVEL%",
])

write("ninja-msvc.bat", lines + [
    "ninja %*",
    "exit /b %ERRORLEVEL%",
])

write("ctest-msvc.bat", lines + [
    "ctest --output-on-failure",
    "exit /b %ERRORLEVEL%",
])

# Syntax-only check for the adapter TU (fast feedback while editing).
write("syntax-msvc.bat", lines + [
    "cl.exe /nologo /TP /Zs /std:c++17 /utf-8 /bigobj /D_CRT_SECURE_NO_WARNINGS "
    "/DGGML_USE_CPU /DGGML_USE_CUDA /DKVMEM_ENABLE_NVME=0 /DLLAMA_KVMEM=1 "
    "/I" + ROOT_WIN + r"\llama.cpp\src "
    "/I" + ROOT_WIN + r"\llama.cpp\include "
    "/I" + ROOT_WIN + r"\llama.cpp\ggml\include "
    "/I" + ROOT_WIN + r"\llama.cpp\ggml\src "
    "/I" + ROOT_WIN + r"\llama.cpp\common "
    "/I" + ROOT_WIN + r"\kvmem\include "
    "/I" + ROOT_WIN + r"\src\adapter "
    '/external:I"' + CUDA + r'\include" '
    "/external:W0 "
    "%*",
    "exit /b %ERRORLEVEL%",
])

print("build dir:", BUILD_DIR)

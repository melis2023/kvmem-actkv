# 编译 KVMem adapter(本机 Windows)

本记录沉淀 2026-10-10 在 Windows 上增量编译 `build-cuda13.4-dll`
(KVMem adapter + llama.cpp + CUDA 13.4)的完整过程与坑。
下次编译直接照此执行。

## 关键前提(为什么不能裸跑 `cmake --build`)

- 构建目录 `build-cuda13.4-dll/` 已配置:**Ninja** 生成器、**Release**、CUDA v13.4。
- `build.ninja` 里 **cl.exe 固化为 D 盘 MSVC `14.44.352`**(Visual Studio 2022 Professional,
  路径 `D:\Program Files\Microsoft Visual Studio\2022\Professional`)。
- `INCLUDES` 只固化了源码 include(`-I...`),**MSVC 标准头文件(`stdbool.h` 等)靠环境变量 `INCLUDE`/`LIB`**。
- 因此必须在编译前 **source 与 cl.exe 同盘的 VsDevCmd**,设好 `INCLUDE`/`LIB`/`PATH`。

## 踩过的坑(避免重复)

1. **裸跑 `cmake --build build-cuda13.4-dll`** → `C1083: 无法打开标准文件 stdbool.h`。
   原因:`INCLUDE` 没设,cl.exe 找不到标准头文件。

2. **用 C 盘 BuildTools 的 VsDevCmd**(`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\...`)
   → 仍然 `C1083 stdbool.h`。
   原因:C 盘 BuildTools 设的 `INCLUDE` 指向 C 盘 MSVC,与固化的 **D 盘 cl.exe 14.44.352** 不匹配。
   **必须用 D 盘 VS 2022 Professional 的 VsDevCmd。**

3. **VsDevCmd 参数用 `x64`** → `[ERROR:VsDevCmd.bat] Invalid command line argument: 'x64'`。
   **正确参数是 `-arch=x64`**(D 盘 VS Pro 的 VsDevCmd 不接受裸 `x64`)。

4. **PowerShell 不支持 `&&`**;且 cmd 里 `%INCLUDE%` 会在 VsDevCmd 前就展开,
   无法用来"看一眼" INCLUDE 是否正确(要用临时 .bat 或直接用 `-no_logo` 静默)。

## 正确编译命令

在任意 PowerShell 里执行(工作目录 `I:\llama\kvmem-llama.cpp-0.17.0`):

```powershell
cmd /c 'call "D:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" arch=x64 -no_logo && cmake --build build-cuda13.4-dll --config Release'
```

说明:
- `call "D:\...\VsDevCmd.bat" arch=x64 -no_logo`:设 D 盘 VS Pro 环境(对齐固化的 cl.exe 14.44.352)。
  `-no_logo` 抑制 banner,输出干净。
- `&& cmake --build ...`:source 完环境后增量编译。Ninja 自动并行。
- 全量重配(改了 CMake 选项才需要)才用 `cmake -B build-cuda13.4-dll -G Ninja`,平时不要重配。

确认成功:输出末尾出现 `Linking CXX executable bin\llama-kvmem-server.exe`(或类似),
无 `error C`/`FAILED`。可再跑一次 `cmake --build` 无变化即稳定。

## 验证(编译后必跑)

编译 `kvmem` 宿主库 + 测试(`build-cuda13.4-dll\bin\` 下):

```powershell
cd build-cuda13.4-dll\bin
.\kvmem_store_test.exe       # 含 LRFU 时间累积测试
.\confidence_monitor_test.exe
```

全部打印 `OK` 且退出码 0 即通过。改动 adapter 后这两组测试必须绿。

## 纯宿主测试(无 CUDA 依赖)

`kvmem` 库本身是纯 host C++17,可在无 CUDA 的机器上用单独的 host 构建
(`build-kvmem-host/`,Ninja)跑测试,只编 `kvmem` 库 + 测试,不编 adapter/server。
adapter/server 改动必须走上面的完整 CUDA 构建才能验证。

## 环境变量(仅当要单独确认 INCLUDE 时用)

临时 .bat 写法(避免 PowerShell 变量展开时机问题):

```bat
@echo off
call "D:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" arch=x64 -no_logo
echo %INCLUDE%
```

正常编译不需要手动设 INCLUDE,VsDevCmd 会在同一 cmd 进程内完成。

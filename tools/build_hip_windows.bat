@echo off
rem Builds the engine on Windows with the AMD ROCm toolchain (HIP), for the gfx12 family.
rem
rem   tools\build_hip_windows.bat [gfx-architecture] [build-folder]
rem
rem Example: tools\build_hip_windows.bat gfx1200 build-hip-port
rem
rem What it needs, all of it on this PC:
rem   - ROCm 7.2 (the HIP compiler, the runtime and the hipBLASLt development files)
rem   - Visual Studio 2022 Build Tools (the MSVC runtime libraries the ROCm clang links
rem     against, and the Ninja and CMake that come with it)
rem
rem The build is a Release build of this tree's engine: the variant suffix q4u32 goes into
rem the version string ("0.1.63-q4u32"), so a fork build is never mistaken for an
rem upstream one.
rem
rem The result is build-hip-port\strata.exe (and the test binaries next to it).  A
rem hand-built engine carries no tuning table: see docs/AMD_HIP.md ("A hand-built engine
rem has no tuning table").
rem
rem The paths below carry "(x86)", so this script uses goto instead of the
rem "if ... ( ... )" blocks: a parenthesis inside a quoted path closes such a block.

setlocal

set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=gfx1200"
set "BUILD=%~2"
if "%BUILD%"=="" set "BUILD=build-hip-port"

rem The ROCm installation and the Visual Studio Build Tools.  Both can be overridden:
rem   set ROCM_ROOT=...
rem   set VS_ROOT=...
if "%ROCM_ROOT%"=="" set "ROCM_ROOT=C:\Program Files\AMD\ROCm\7.2"
if "%VS_ROOT%"=="" set "VS_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if "%WINKIT%"=="" set "WINKIT=C:\Program Files (x86)\Windows Kits\10"

if not exist "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" goto no_vs
if not exist "%ROCM_ROOT%\bin\clang++.exe" goto no_rocm

rem The MSVC runtime libraries are only visible inside the Visual Studio environment.
call "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto no_vcvars

rem Ninja: the one of the Visual Studio Build Tools, else the one on the PATH.
set "NINJA=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if exist "%NINJA%" goto have_ninja
for %%N in (ninja.exe) do set "NINJA=%%~$PATH:N"
if exist "%NINJA%" goto have_ninja
echo ninja not found: it comes with the Visual Studio Build Tools.
exit /b 1
:have_ninja

rem CMake: the one of the Build Tools, else the one on the PATH.
set "CMAKE=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if exist "%CMAKE%" goto have_cmake
set "CMAKE=cmake.exe"
:have_cmake

rem The resource compiler (the Windows kit) and the linker (ROCm's lld-link).
rem The Windows kit's rc.exe: the newest version's x64 folder.  The version folder
rem changes with the Windows update, so it is searched and not written down.
rem The search runs in a subroutine: inside a parenthesised block cmd splits the
rem path at the "(x86)" of "Program Files (x86)" and loses it.
set "RC="
for /f "delims=" %%V in ('dir /b /ad /o-d "%WINKIT%\bin"') do call :find_rc "%%V"
if defined RC goto have_rc
echo The Windows kit's rc.exe not found under %WINKIT%: install the Windows 10/11 SDK.
exit /b 1
:have_rc
rem CMake reads the path as cmake code, so it wants forward slashes.
set "RC=%RC:\=/%"
set "LDLLINK=%ROCM_ROOT%\bin\lld-link.exe"

rem The ROCm cmake packages: the Windows packages are named hip-config.cmake, not
rem hipConfig.cmake, so find_package("hip") does not find them by itself.
set "HIP_CMAKE=%ROCM_ROOT%\lib\cmake"

echo ---- cmake (Architektur %ARCH%, nach %BUILD%) ----
"%CMAKE%" -S engine -B "%BUILD%" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_SYCL=OFF ^
  -DCMAKE_HIP_ARCHITECTURES=%ARCH% ^
  -Dhip_DIR="%HIP_CMAKE%/hip" ^
  -Dhipblas_DIR="%HIP_CMAKE%/hipblas" ^
  -Dhipblas-common_DIR="%HIP_CMAKE%/hipblas-common" ^
  -Dhipblaslt_DIR="%HIP_CMAKE%/hipblaslt" ^
  -Drocblas_DIR="%HIP_CMAKE%/rocblas" ^
  -DCMAKE_C_COMPILER="%ROCM_ROOT%/bin/clang.exe" ^
  -DCMAKE_CXX_COMPILER="%ROCM_ROOT%/bin/clang++.exe" ^
  -DCMAKE_RC_COMPILER="%RC%" ^
  -DCMAKE_LINKER="%LDLLINK:/=/%" ^
  -DSTRATA_ENGINE_FLAVOR=q4u32 ^
  -DSTRATA_GEOMETRY=flash_next ^
  -DSTRATA_BUILD_TESTS=OFF ^
  -DSTRATA_BUILD_CONVERSATION_TESTS=OFF ^
  -DSTRATA_BUILD_SPLIT=OFF ^
  -DSTRATA_BUILD_DSV41=OFF
if errorlevel 1 exit /b 1

echo ---- build ----
"%NINJA%" -C "%BUILD%" strata
if errorlevel 1 exit /b 1

echo.
echo Fertig: %BUILD%\strata.exe
"%BUILD%\strata.exe" --version
endlocal
exit /b 0

:no_vs
echo Visual Studio Build Tools not found: %VS_ROOT%
echo Set VS_ROOT to the installation that holds vcvars64.bat.
exit /b 1

:no_rocm
echo ROCm not found: %ROCM_ROOT%
echo Set ROCM_ROOT to the ROCm installation.
exit /b 1

:no_vcvars
echo vcvars64.bat failed
exit /b 1

rem The newest rc.exe of the Windows kit, if it has one.
:find_rc
if defined RC goto :eof
if exist "%WINKIT%\bin\%~1\x64\rc.exe" set "RC=%WINKIT%\bin\%~1\x64\rc.exe"
goto :eof

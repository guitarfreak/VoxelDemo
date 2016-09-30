@echo off

set scriptpath=%~d0%~p0
cd %scriptpath%

if not exist "..\buildWin32" mkdir ..\buildWin32
pushd ..\buildWin32

set LIBFOLDER=C:\Projects\Libs
set MYLIBFOLDER=C:\Projects\MyLibs
set INCLUDES=
set LIBS=
set LINKER_INCLUDES=

set LINKER_LIBS= -DEFAULTLIB:Opengl32.lib -DEFAULTLIB:ws2_32.lib -DEFAULTLIB:Shell32.lib -DEFAULTLIB:user32.lib -DEFAULTLIB:Gdi32.lib -DEFAULTLIB:Shlwapi.lib
rem set LINKER_LIBS= -DEFAULTLIB:Opengl32.lib -DEFAULTLIB:ws2_32.lib -DEFAULTLIB:Shell32.lib -DEFAULTLIB:Gdi32.lib -DEFAULTLIB:Shlwapi.lib

rem call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\vcvars32.bat"
rem call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat"


set INCLUDES=%INCLUDES% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include"
set LINKER_INCLUDES=%LINKER_INCLUDES% -LIBPATH:"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\lib\amd64"
set INCLUDES=%INCLUDES% -I"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Include"
set LINKER_INCLUDES=%LINKER_INCLUDES% -LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Lib\x64"
set PATH=C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64;%PATH%


rem set INCLUDES=%INCLUDES% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include"
rem set LINKER_INCLUDES=%LINKER_INCLUDES% -LIBPATH:"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\lib"
rem set INCLUDES=%INCLUDES% -I"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Include"
rem set LINKER_INCLUDES=%LINKER_INCLUDES% -LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Lib"
rem set PATH=C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin;%PATH%


set COMPILER_OPTIONS= -MD -EHsc -wd4005 -Od -nologo -Gm -GR -Oi -Zi -FC -wd4838
set LINKER_OPTIONS= -link -SUBSYSTEM:WINDOWS -OUT:main.exe -incremental:no -opt:ref


del main_*.pdb > NUL 2> NUL
echo. 2>lock.tmp
cl %COMPILER_OPTIONS% ..\code\app.cpp -LD %INCLUDES% -link -incremental:no -opt:ref -PDB:main_%random%.pdb -EXPORT:appMain %LINKER_INCLUDES% %LINKER_LIBS%

del lock.tmp
cl %COMPILER_OPTIONS% ..\code\main.cpp %INCLUDES% %LINKER_OPTIONS% %LINKER_INCLUDES% %LINKER_LIBS%

:parseParameters
IF "%~1"=="" GOTO parseParametersEnd
IF "%~1"=="-run" call main.exe
SHIFT
GOTO parseParameters
:parseParametersEnd

popd
set LOCATION=

rem exit -b

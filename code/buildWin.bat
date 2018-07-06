@echo off

set 7ZIP_PATH=C:\Program Files\7-Zip\7z.exe
set APP_NAME=VoxelGame

set scriptpath=%~d0%~p0
cd %scriptpath%

set PLATFORM=win64
if "%~1"=="-x86" set PLATFORM=win32

set PLATFORM2=x64
if "%~1"=="-x86" set PLATFORM2=x86

set BUILD_FOLDER=buildWin64
if "%~1"=="-x86" set BUILD_FOLDER=buildWin32

if "%~4"=="-ship" goto buildSetup
goto buildSetupEnd
:buildSetup
	set BUILD_FOLDER=releaseBuild
	if exist "..\%BUILD_FOLDER%" rmdir "..\%BUILD_FOLDER%" /S /Q
:buildSetupEnd

if not exist "..\%BUILD_FOLDER%" mkdir "..\%BUILD_FOLDER%"
pushd "..\%BUILD_FOLDER%"

set INC=
set LINC=

set LINKER_LIBS= -DEFAULTLIB:Opengl32.lib -DEFAULTLIB:ws2_32.lib -DEFAULTLIB:Shell32.lib -DEFAULTLIB:user32.lib -DEFAULTLIB:Gdi32.lib -DEFAULTLIB:Shlwapi.lib -DEFAULTLIB:Winmm.lib -DEFAULTLIB:Ole32.lib

set          INC=%INC% -I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include"
set          INC=%INC% -I"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Include"

if "%~1"=="-x86" goto compilerSelectionX86

set                  PATH=C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64;%PATH%
set LINC=%LINC% -LIBPATH:"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\lib\amd64"
set LINC=%LINC% -LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Lib\x64"

goto compilerSelectionEnd
:compilerSelectionX86

set                  PATH=C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin;%PATH%
set LINC=%LINC% -LIBPATH:"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\lib"
set LINC=%LINC% -LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Lib"

:compilerSelectionEnd


set INC=%INC% -I"..\libs\freetype 2.9\include"
set LINC=%LINC% -LIBPATH:"..\libs\freetype 2.9\lib\%PLATFORM%"
set LINKER_LIBS=%LINKER_LIBS% -DEFAULTLIB:freetype.lib


set BUILD_MODE=-Od
set MODE_DEFINE=
if "%~3"=="-release" (
	rem -Oy -Zo
	set BUILD_MODE=-O2
	set MODE_DEFINE=-DRELEASE_BUILD
)

set RUNTIME=-MD

if "%~4"=="-ship" (
	set MODE_DEFINE=%MODE_DEFINE% -DSHIPPING_MODE
	set RUNTIME=-MT
)

if "%~5"=="-fullOptimize" (
	set MODE_DEFINE=%MODE_DEFINE% -DFULL_OPTIMIZE
)

rem -d2cgsummary -Bt
set COMPILER_OPTIONS= %RUNTIME% %BUILD_MODE% -nologo -Oi -FC -wd4838 -wd4005 -fp:fast -fp:except- -Gm- -GR- -EHa- -Z7
set LINKER_OPTIONS= -link -SUBSYSTEM:WINDOWS -OUT:"%APP_NAME%.exe" -incremental:no -opt:ref


if "%~4"=="-ship" goto noDLL

del main_*.pdb > NUL 2> NUL
echo. 2>lock.tmp
cl %COMPILER_OPTIONS% ..\code\app.cpp %MODE_DEFINE% -LD %INC% -link -incremental:no -opt:ref -PDB:main_%random%.pdb -EXPORT:appMain %LINC% %LINKER_LIBS%
del lock.tmp

:noDLL


cl %COMPILER_OPTIONS% ..\code\main.cpp %MODE_DEFINE% %INC% %LINKER_OPTIONS% %LINC% %LINKER_LIBS%



if "%~4"=="-ship" goto packShippingFolder
goto packShippingFolderEnd

:packShippingFolder

	rem This is suboptimal.

	cd ..

	mkdir ".\%BUILD_FOLDER%\data"
	xcopy ".\data" ".\%BUILD_FOLDER%\data" /E /Q

	if "%~3"=="" goto nodelete
		del ".\%BUILD_FOLDER%\*.pdb"
		del ".\%BUILD_FOLDER%\*.exp"
		del ".\%BUILD_FOLDER%\*.lib"
		del ".\%BUILD_FOLDER%\*.obj"
	:nodelete

	xcopy ".\libs\freetype 2.9\lib\%PLATFORM%\*.dll" ".\%BUILD_FOLDER%" /Q

	xcopy ".\README.txt" ".\%BUILD_FOLDER%" /Q
	xcopy ".\Licenses.txt" ".\%BUILD_FOLDER%" /Q



	call "C:\\Standalone\\rcedit.exe" "%BUILD_FOLDER%\\%APP_NAME%.exe" --set-icon icon.ico

	set RELEASE_FOLDER=.\releases\%PLATFORM%\%APP_NAME%
	if exist "%RELEASE_FOLDER%" rmdir "%RELEASE_FOLDER%" /S /Q
	mkdir "%RELEASE_FOLDER%"

	xcopy %BUILD_FOLDER% "%RELEASE_FOLDER%" /E /Q

	rmdir ".\%BUILD_FOLDER%" /S /Q

	"C:\Program Files\7-Zip\7z.exe" a "%RELEASE_FOLDER% %PLATFORM2%.zip" "%RELEASE_FOLDER%"

:packShippingFolderEnd


:parseParameters
IF "%~2"=="" GOTO parseParametersEnd
IF "%~2"=="-run" call %APP_NAME%.exe
SHIFT
GOTO parseParameters
:parseParametersEnd

rem popd
set LOCATION=

rem exit -b
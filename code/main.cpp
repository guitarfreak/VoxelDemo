#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "rt_misc.h"
#include "rt_hotload.h"

MemoryBlock* globalMemory;

int CALLBACK WinMain(HINSTANCE instance, HINSTANCE prevInstance, LPSTR commandLine, int showCode) {

	globalMemory = (MemoryBlock*)malloc(sizeof(MemoryBlock));
	initMemorySizes(globalMemory, megaBytes(10), megaBytes(10), megaBytes(10), kiloBytes(1));
	initMemory(globalMemory);

	HotloadDll hotloadDll;
	initDll(&hotloadDll, "app.dll", "appTemp.dll", "lock.tmp");

	WindowsData wData = windowsData(instance, prevInstance, commandLine, showCode);

    bool firstFrame = true;
    bool isRunning = true;
    while(isRunning) {
        bool reload = updateDll(&hotloadDll);
     	platform_appMain = (appMainType*)getDllFunction(&hotloadDll, "appMain");
        platform_appMain(firstFrame, reload, &isRunning, globalMemory, wData);

        firstFrame = false;
    }

	return 0;
}
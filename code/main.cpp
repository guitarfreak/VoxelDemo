#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "rt_misc.h"
#include "rt_hotload.h"
#include "rt_misc_win32.h"


int CALLBACK WinMain(HINSTANCE instance, HINSTANCE prevInstance, LPSTR commandLine, int showCode) {
	HotloadDll hotloadDll;
	initDll(&hotloadDll, "app.dll", "appTemp.dll", "lock.tmp");

	WindowsData wData = windowsData(instance, prevInstance, commandLine, showCode);

	ThreadQueue threadQueue;
	threadInit(&threadQueue, 7);

	AppMemory appMemory = {};

    bool firstFrame = true;
    bool secondFrame = false;
    bool isRunning = true;
    while(isRunning) {

    	bool reload = false;
		if(threadQueueFinished(&threadQueue)) reload = updateDll(&hotloadDll);
     	platform_appMain = (appMainType*)getDllFunction(&hotloadDll, "appMain");
        platform_appMain(firstFrame, reload, &isRunning, wData, &threadQueue, &appMemory);

        if(firstFrame) firstFrame = false;

    }

	return 0;
}

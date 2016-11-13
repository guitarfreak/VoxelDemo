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
    	// push delta time, input, 

    	bool reload = false;
    	// if(threadQueueFull()threadQueue.completionCount == threadQueue.completionGoal)
		if(threadQueueFinished(&threadQueue)) reload = updateDll(&hotloadDll);
     	platform_appMain = (appMainType*)getDllFunction(&hotloadDll, "appMain");
        platform_appMain(firstFrame, secondFrame, reload, &isRunning, wData, &threadQueue, &appMemory);

     	// platform_postMain = (postMainType*)getDllFunction(&hotloadDll, "postMain");
        // platform_postMain(firstFrame, secondFrame, reload, &isRunning, &debugMemory, wData, &threadQueue);

        secondFrame = false;
        if(firstFrame) secondFrame = true;
        firstFrame = false;
    }

	return 0;
}

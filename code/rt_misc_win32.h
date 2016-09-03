#pragma once
#include <Shlwapi.h> // PathFileExists()

void* mallocWithBaseAddress(void* baseAddress, int sizeInBytes) {
    void* mem = VirtualAlloc(baseAddress, (size_t)sizeInBytes, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    DWORD error = GetLastError();

    return mem;
}

struct ThreadJob {
    void (*function)(void* data);
    void* data;
};

struct ThreadQueue {
    volatile uint completionGoal;
    volatile uint completionCount;
    volatile uint writeIndex;
    volatile uint readIndex;
    HANDLE semaphore;

    // int threadId[16];

    ThreadJob jobs[256];
    // ThreadJob jobs[1024];
};

int globalThreadIds[16];

int getThreadQueueId() {
	int currentId = GetCurrentThreadId();

	int id = -1;
	for(int i = 0; i < arrayCount(globalThreadIds); i++) {
		if(globalThreadIds[i] == currentId) {
			id = i;
			break;
		}
	}

	assert(id != -1);
	return id;
}

bool doNextThreadJob(ThreadQueue* queue) {
    bool shouldSleep = false;

    uint currentReadIndex = queue->readIndex;
    uint newReadIndex = (currentReadIndex + 1) % arrayCount(queue->jobs);

    if(currentReadIndex != queue->writeIndex) {
        LONG oldValue = InterlockedCompareExchange(&queue->readIndex, newReadIndex, currentReadIndex);
        if(newReadIndex == queue->readIndex) {
            ThreadJob job = queue->jobs[currentReadIndex];
            job.function(job.data);
            InterlockedIncrement(&queue->completionCount);
        }
    } else {
        shouldSleep = true;
    }

    return shouldSleep;
}

DWORD WINAPI threadProcess(LPVOID data) {
    ThreadQueue* queue = (ThreadQueue*)data;
    for(;;) {
        if(doNextThreadJob(queue)) {
            WaitForSingleObject(queue->semaphore, INFINITE);
        }
    }

    return 0;
}

void threadInit(ThreadQueue* queue, int numOfThreads) {
    queue->completionGoal = 0;
    queue->completionCount = 0;
    queue->writeIndex = 0;
    queue->readIndex = 0;
    queue->semaphore = CreateSemaphore(0, 0, 255, "Semaphore");

    for(int i = 0; i < numOfThreads; i++) {
        HANDLE thread = CreateThread(0, 0, threadProcess, (void*)(queue), 0, 0);
        // BOOL WINAPI SetThreadPriority(
        //   _In_ HANDLE hThread,
        //   _In_ int    nPriority
        // );

        SetThreadPriority(thread, -2);

        int id = GetThreadId(thread);
        // queue->threadId[i] = id;
        globalThreadIds[i] = id;
        if(!thread) printf("Could not create thread\n");
        CloseHandle(thread);
    }
}

bool threadQueueAdd(ThreadQueue* queue, void (*function)(void*), void* data, bool skipIfFull = false) {
    int newWriteIndex = (queue->writeIndex + 1) % arrayCount(queue->jobs);
    if(skipIfFull) {
    	if(newWriteIndex == queue->readIndex) return false;
    } else {
	    assert(newWriteIndex != queue->readIndex);
    }
    ThreadJob* job = queue->jobs + queue->writeIndex;
    job->function = function;
    job->data = data;
    InterlockedIncrement(&queue->completionGoal);
    _ReadWriteBarrier(); // doesn't work on 32 bit?
    InterlockedExchange(&queue->writeIndex, newWriteIndex);
    ReleaseSemaphore(queue->semaphore, 1, 0);

    return true;
}

bool threadQueueFull(ThreadQueue* queue) {
	int newWriteIndex = (queue->writeIndex + 1) % arrayCount(queue->jobs);
    bool result = newWriteIndex == queue->readIndex;
    return result;
}

void threadQueueComplete(ThreadQueue* queue) {
    while(queue->completionCount != queue->completionGoal) {
        doNextThreadJob(queue);
    }
    InterlockedExchange(&queue->completionGoal, 0);
    InterlockedExchange(&queue->completionCount, 0);
}
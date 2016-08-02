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

    ThreadJob jobs[256];
};

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
        if(!thread) printf("Could not create thread\n");
        CloseHandle(thread);
    }
}

void threadQueueAdd(ThreadQueue* queue, void (*function)(void*), void* data) {
    int newWriteIndex = (queue->writeIndex + 1) % arrayCount(queue->jobs);
    assert(newWriteIndex != queue->readIndex);
    ThreadJob* job = queue->jobs + queue->writeIndex;
    job->function = function;
    job->data = data;
    InterlockedIncrement(&queue->completionGoal);
    _ReadWriteBarrier();
    InterlockedExchange(&queue->writeIndex, newWriteIndex);
    ReleaseSemaphore(queue->semaphore, 1, 0);
}


void threadQueueComplete(ThreadQueue* queue) {
    while(queue->completionCount != queue->completionGoal) {
        doNextThreadJob(queue);
    }
    InterlockedExchange(&queue->completionGoal, 0);
    InterlockedExchange(&queue->completionCount, 0);
}

struct MemoryBlock {
	ExtendibleMemoryArray* pMemory;
	MemoryArray* tMemory;
	ExtendibleBucketMemory* dMemory;

	MemoryArray* pDebugMemory;
	MemoryArray* tMemoryDebug;
	ExtendibleMemoryArray* debugMemory;
};

extern MemoryBlock* globalMemory;

#define getPStruct(type) 		(type*)(getPMemory(sizeof(type)))
#define getPArray(type, count) 	(type*)(getPMemory(sizeof(type) * count))
#define getTStruct(type) 		(type*)(getTMemory(sizeof(type)))
#define getTArray(type, count) 	(type*)(getTMemory(sizeof(type) * count))
#define getTString(size) 		(char*)(getTMemory(size)) 
#define getDStruct(type) 		(type*)(getDMemory(sizeof(type)))
#define getDArray(type, count) 	(type*)(getDMemory(sizeof(type) * count))

// void* getPMemory(int size, MemoryBlock * memory = 0);
// void* getTMemory(int size, MemoryBlock * memory = 0);

void *getPMemory(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getExtendibleMemoryArray(size, memory->pMemory);
    return location;
}

void * getTMemory(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getMemoryArray(size, memory->tMemory);
    return location;
}

#define getPStructDebug(type) 		(type*)(getPMemoryDebug(sizeof(type)))
#define getPArrayDebug(type, count) 	(type*)(getPMemoryDebug(sizeof(type) * count))
#define getTStructDebug(type) 		(type*)(getTMemoryDebug(sizeof(type)))
#define getTArrayDebug(type, count) 	(type*)(getTMemoryDebug(sizeof(type) * count))
#define getTStringDebug(size) 		(char*)(getTMemoryDebug(size)) 
// #define getDStructDebug(type) 		(type*)(getDMemoryDebug(sizeof(type)))
// #define getDArrayDebug(type, count) 	(type*)(getDMemoryDebug(sizeof(type) * count))

void clearTMemory(MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	clearMemoryArray(memory->tMemory);
}

void *getPMemoryDebug(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getMemoryArray(size, memory->pDebugMemory);
    return location;
}

void * getTMemoryDebug(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getMemoryArray(size, memory->tMemoryDebug);
    return location;
}

void clearTMemoryDebug(MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	clearMemoryArray(memory->tMemoryDebug);
}

// void pushMarkerTMemory(MemoryBlock * memory = 0)  {
    // if(!memory) memory = globalMemory;
    // memory->markerStack[memory->markerStackIndex] = memory->temporaryIndex;
    // memory->markerStackIndex++;
// }

// void popMarkerTMemory(MemoryBlock * memory = 0)  {
    // if(!memory) memory = globalMemory;
    // int size = memory->temporaryIndex - memory->markerStack[memory->markerStackIndex];
    // memory->markerStackIndex--;
    // freeTMemory(size, memory);
// }

void * getDMemory(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getExtendibleBucketMemory(memory->dMemory);
    return location;
}

void freeDMemory(void* address, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	freeExtendibleBucketMemory(address, memory->dMemory);
}

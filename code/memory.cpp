
struct MemoryBlock {
	bool debugMode;

	ExtendibleMemoryArray* pMemory;
	MemoryArray* tMemory;
	ExtendibleBucketMemory* dMemory;

	MemoryArray* pDebugMemory;
	MemoryArray* tMemoryDebug;
	ExtendibleMemoryArray* debugMemory;
};

extern MemoryBlock* globalMemory;

#define getPStruct(type) 		(type*)(getPMemoryMain(sizeof(type)))
#define getPArray(type, count) 	(type*)(getPMemoryMain(sizeof(type) * count))
#define getPString(count) 	(char*)(getPMemoryMain(count))
#define getTStruct(type) 		(type*)(getTMemoryMain(sizeof(type)))
#define getTArray(type, count) 	(type*)(getTMemoryMain(sizeof(type) * count))
#define getTString(size) 		(char*)(getTMemoryMain(size)) 
#define getDStruct(type) 		(type*)(getDMemoryMain(sizeof(type)))
#define getDArray(type, count) 	(type*)(getDMemoryMain(sizeof(type) * count))

// void* getPMemory(int size, MemoryBlock * memory = 0);
// void* getTMemory(int size, MemoryBlock * memory = 0);

void *getPMemoryMain(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getExtendibleMemoryArray(size, memory->pMemory);
    return location;
}

void * getTMemoryMain(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getMemoryArray(size, memory->tMemory);
    return location;
}

#define getPStructDebug(type) 		(type*)(getPMemoryDebug(sizeof(type)))
#define getPArrayDebug(type, count) 	(type*)(getPMemoryDebug(sizeof(type) * count))
#define getPStringDebug(count) (char*)(getPMemoryDebug(count))
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

void * getDMemoryMain(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getExtendibleBucketMemory(memory->dMemory);
    return location;
}

void freeDMemory(void* address, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	freeExtendibleBucketMemory(address, memory->dMemory);
}


// Choose right function according to memory mode thats set globally.

#define getPStructX(type) 		(type*)(getPMemory(sizeof(type)))
#define getPArrayX(type, count) 	(type*)(getPMemory(sizeof(type) * count))
#define getPStringX(count) 	(char*)(getPMemory(count))
#define getTStructX(type) 		(type*)(getTMemory(sizeof(type)))
#define getTArrayX(type, count) 	(type*)(getTMemory(sizeof(type) * count))
#define getTStringX(size) 		(char*)(getTMemory(size)) 

void *getPMemory(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	if(!memory->debugMode) return getPMemoryMain(size, memory);
	else return getPMemoryDebug(size, memory);
}

void *getTMemory(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	if(!memory->debugMode) return getTMemoryMain(size, memory);
	else return getTMemoryDebug(size, memory);
}





inline char* getPStringCpy(char* str, int size = -1) {
	char* newStr = getPString((size == -1 ? strLen(str) : size) + 1);
	strCpy(newStr, str, size);
	return newStr;
}

inline char* getTStringCpy(char* str, int size = -1) {
	char* newStr = getTString((size == -1 ? strLen(str) : size) + 1);
	strCpy(newStr, str, size);
	return newStr;
}

inline char* getPStringClr(int size) { 
	char* s = getPString(size);
	s[0] = '\0';
	return s;
}

inline char* getTStringClr(int size) { 
	char* s = getTString(size);
	s[0] = '\0';
	return s;
}

char** getTStringArray(char** strings, int count) {

	char** array = getTArray(char*, count);
	for(int i = 0; i < count; i++) {
		array[i] = getTString(strLen(strings[i]));
	}

	return array;
}

#pragma once 

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits>

typedef unsigned int uint;
typedef unsigned char uchar;
typedef uint64_t u64;
typedef double r64;

void assert(bool check) {
	if(!check) {

		printf("Assert fired!\n");
		if(IsDebuggerPresent()) {
			__debugbreak();
		}
		exit(1);
	}
}

#define arrayCount(array) (sizeof(array) / sizeof((array)[0]))
#define addPointer(ptr, int) ptr = (char*)ptr + int
#define memberSize(type, member) sizeof(((type *)0)->member)


void memSet(void* dest, int value, int numOfBytes) {
	char* destPointer = (char*)dest;
	for(int i = 0; i < numOfBytes; i++) {
		destPointer[i] = value;
	}
}

void zeroMemory(void* memory, int size) {
	memSet(memory, 0, size);
}

void memCpy(void* dest, void* source, int numOfBytes) {
	char* destPointer = (char*)dest;
	char* sourcePointer = (char*)source;
	for(int i = 0; i < numOfBytes; i++) {
		destPointer[i] = sourcePointer[i];
	}
}

//
// Strings
// 

int strLen(char* str) {
	int len = 0;
	while(str[len] != '\0') len++;

	return len;
}

inline void strClear(char* string) {
	string[0] = '\0';
}

int intDigits(int n) {
	int count = 0;
	if(n == 0) return 1;

	while(n != 0) {
		n /= 10;
		count++;
	}

	return count;
}

inline char * intToStr(char * buffer, int var) {
	int digits = intDigits(var);
	if(var < 0) digits++;
	buffer[digits--] = '\0';

	if(var < 0) {
		buffer[0] = '-';
		var *= -1;
	}

	if(var == 0) {
		buffer[0] = '0';
		return buffer;
	}


	while(var != 0) {
		buffer[digits--] = (char)((var % 10) + '0');
		var /= 10;
	}

	return buffer;
};

char * floatToStr(char * buffer, float f, int precision = 0)
{
	int stringSize = sprintf(buffer, "%1.7f", f); // TODO: Make more robust.
	if (precision > 0) {
		buffer[stringSize - (7-precision)] = '\0';
	} else {
		int stringIndex = stringSize-1;
		while(buffer[stringIndex] == '0') {
			stringIndex--;
		}
		if(buffer[stringIndex] == '.') {
			stringIndex++;
			buffer[stringIndex] = '0';
		}
		buffer[stringIndex + 1] = '\0'; 
	}

	return buffer;
}

void strCpyBackwards(char* dest, char* str, int size) {
	for(int i = size-1; i >= 0; i--) dest[i] = str[i];
}

inline void strInsert(char* string, int index, char ch) {
	int stringLength = strLen(string);
	// memCpy(string+index+1, string+index, stringLength-index+1);
	strCpyBackwards(string+index+1, string+index, stringLength-index+1);
	string[index] = ch;
}

void strInsert(char* destination, int index, char* str) {
	int size = strLen(str);
	int sizeDest = strLen(destination);

	strCpyBackwards(destination + size + index, destination + index, sizeDest - index + 1);
	strCpyBackwards(destination + index, str, size);
}

void strAppend(char* destination, char* str, int size = -1) {
	int destSize = strLen(destination);
	int strSize = size == -1? strLen(str) : size;

	destination[destSize+strSize] = '\0';

	for(int i = 0; i < strSize; i++) {
		destination[destSize+i] = str[i];
	}
}

inline int charToInt(char c) {
	return (int)c - '0';
}

inline int strToInt(char* string) {
	int result = atoi(string);
	return result;
}

inline int strHexToInt(char* string) {
	int result = (int)strtol(string, 0, 16);
	return result;
}

inline float strToFloat(char* string) {
	float result = atof(string);		
	return result;
}

inline void strErase(char* string, int index, int size) {
	int amount = strLen(string) - index - size;
	memCpy(string+index, string+index+size, amount+1);
}

void copySubstring(char * destination, char * source, int start, int end) {
	int size = end - start;
	memCpy(destination, source + start, size + 1);
	destination[size + 1] = '\0';
}

void strCpy(char * destination, char * source, int size = -1) {
	if(size != 0) {
		if(size == -1) size = strLen(source);
		memCpy(destination, source, size);
	}
	destination[size] = '\0';
}

bool strCompare(char* str, char* str2, int size = -1) {
	int length = size == -1 ? strLen(str2) : size;

	if(length != strLen(str)) return false;

	bool result = true;
	for(int i = 0; i < length; i++) {
		if(str[i] != str2[i]) {
			result = false;
			break;
		}
	}

	return result;
}

bool strCompare(char* str, int size1, char* str2, int size2) {
	if(size1 != size2) return false;

	bool result = true;
	for(int i = 0; i < size1; i++) {
		if(str[i] != str2[i]) {
			result = false;
			break;
		}
	}

	return result;
}

int strFind(const char* str, char chr, int startIndex = 0) {
	int index = startIndex;
	int pos = -1;
	while(str[index] != '\0') {
		if(str[index] == chr) {
			pos = index;
			break;
		}
		index++;
	}

	return pos+1;
}

int strFindBackwards(char* str, char chr, int startIndex = -1) {
	int length = startIndex == -1 ? strLen(str) : startIndex;

	int pos = -1;
	for(int i = length - 1; i >= 0; i--) {
		if(str[i] == chr) {
			pos = i;
			break;
		}
	}

	return pos+1;
}

int strFind(char* source, char* str, int to = 0, int from = 0) {
	int sourceLength = to > 0 ? to : strLen(source);
	int length = strLen(str);

	if(length > sourceLength) return -1;

	bool found = false;
	int pos = -1;
	for(int i = from; i < (sourceLength-length) + 1; i++) {
		if(source[i] == str[0]) {
			bool check = true;
			pos = i;
			for(int subIndex = 1; subIndex < length; subIndex++) {
				if(source[i+subIndex] != str[subIndex]) {
					check = false;
					break;
				}
			}
			if(check) {
				found = true;
				break;
			}
		}
	}

	int result = -1;
	if(found) result = pos;

	return result;
}

int strFindRight(char* source, char* str, int searchDistance = 0) {
	int result = strFind(source, str, searchDistance);
	if(result != -1) result += strLen(str);

	return result;
}

// int strFindBackwarts(const char* source, const char* str, int to = 0, int from = 0) {
// 	int sourceLength = to > 0 ? to : strLen(source);
// 	int length = strLen(str);

// 	if(length > sourceLength) return -1;

// 	bool found = false;
// 	int pos = -1;
// 	for(int i = from; i < (sourceLength-length) + 1; i++) {
// 		if(source[i] == str[0]) {
// 			bool check = true;
// 			pos = i;
// 			for(int subIndex = 1; subIndex < length; subIndex++) {
// 				if(source[i+subIndex] != str[subIndex]) {
// 					check = false;
// 					break;
// 				}
// 			}
// 			if(check) {
// 				found = true;
// 				break;
// 			}
// 		}
// 	}

// 	int result = -1;
// 	if(found) result = pos;

// 	return result;
// }

// void strAttach(char* destination, char* source) {
	// int length = strLen(destination);
// }

bool strSearchEnd(char* str, int searchLength) {
	bool foundEnd = false;
	for(int i = 0; i < searchLength; i++) {
		if(str[i] == '\0') {
			foundEnd = true;
			break;
		}
	}

	return foundEnd;
}

void strRemove(char* str, int index, int size = 0) {
	int length = size == 0 ? strLen(str) : size;

	memCpy(str + index - 1, str + (index), length-index+1);
}

void strRemoveWhitespace(char* str, int size = 0) {
	int length = size == 0 ? strLen(str) : size;

	int i = 0;
	while(i < length) {
		if(str[i] == '\t' || str[i] == ' ') {
			strRemove(str, i+1, length); 
			length--;
		} else {
			i++;
		}
	}
}



#define initString(function, size) init(function(size), size)
struct String {
	char* data;
	int size;
	int maxSize;

	void init(void* data, int maxSize) {
		this->data = (char*)data;
		this->maxSize = maxSize;
		this->size = 0;
		setEndZero();
	}

	void append(char c) {
		data[size++] = c;
		setEndZero();
	}

	void append(char* str, int length = 0) {
		if(!length) length = strLen(str);
		memCpy(data + size, str, length);
		size += length;
		setEndZero();
	}

	int find(char* str, int start) {
		int len = strLen(str);
		bool found = false;
		int position = 0;
		for(int index = start; index < size-len; index++) {
			if(data[index] == str[0]) {
				position = index+1;
				found = true;
				for(int i = 0; i < len; i++) {
					if(data[index + i] != str[i]) {
						position = 0;
						found = false;
					}
				}
				if(found) break;
			}
		} 

		return position;
	}

	void setEndZero() { data[size] = '\0'; }
};

int createFileAndOverwrite(char* fileName, int fileSize) {
	FILE* file = fopen(fileName, "wb");
	if(file == 0) return -1;

	fileSize += 1;

	char* tempBuffer = (char*)malloc(fileSize);
	zeroMemory(tempBuffer, fileSize);
	fwrite(tempBuffer, fileSize, 1, file);
	free(tempBuffer);

	fclose(file);

	return 0;
}

int fileSize(char* fileName) {
	FILE* file = fopen(fileName, "rb");

	if(file == 0) return -1;

	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	fclose(file);

	return size;
}

bool fileExists(char* fileName) {
	bool result = false;
	FILE* file = fopen(fileName, "rb");

	if(file) {
		result = true;
		fclose(file);
	}
	
	return result;
}

int readFileToBuffer(char* buffer, char* fileName) {
	FILE* file = fopen(fileName, "rb");
	if(file == 0) return -1;

	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	fseek(file, 0, SEEK_SET);

	fread(buffer, size, 1, file);

	fclose(file);
	return size;
}

int readFileSectionToBuffer(char* buffer, char* fileName, int offsetInBytes, int sizeInBytes) {
	FILE* file = fopen(fileName, "r+b");
	if(file == 0) return -1;

	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	fseek(file, 0, SEEK_SET);

	assert(offsetInBytes+sizeInBytes < size);

	fseek(file, offsetInBytes, SEEK_SET);
	fread(buffer, sizeInBytes, 1, file);

	fclose(file);
	return size;
}

void readFileToString(String* str, char* fileName) {
	str->size = readFileToBuffer(str->data, fileName);
	str->setEndZero();

	assert(str->size <= str->maxSize);
}

int writeBufferToFile(char* buffer, char* fileName, int size = -1) {
	if(size == -1) size = strLen(buffer);

	FILE* file = fopen(fileName, "wb");
	if(!file) {
		printf("Could not open file!\n");
		return -1;
	} 

	fwrite(buffer, size, 1, file);

	fclose(file);
	return size;
}

int writeBufferSectionToFile(char* buffer, char* fileName, int offsetInBytes, int sizeInBytes) {
	FILE* file = fopen(fileName, "r+b");
	if(!file) {
		printf("Could not open file!\n");
		return -1;
	} 

	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	fseek(file, 0, SEEK_SET);

	assert(offsetInBytes+sizeInBytes < size);

	fseek(file, offsetInBytes, SEEK_SET);
	fwrite(buffer, sizeInBytes, 1, file);

	fclose(file);
	return size;
}

void writeStringToFile(String str, char* fileName) {
	writeBufferToFile(str.data, fileName, str.size);
}

//
// ???
//

int timeToSeconds(int year = 0, int month = 0, int day = 0, int hour = 0, int minute = 0, int seconds = 0)
{
	// year is 20XX, only works up to ~2060
	return (year * 31556926 + month * 2629743.83 + day * 86400 +
		hour * 3600 + minute * 60 + seconds);
};

inline int terraBytes(int count)
{
	int result = count * 1024 * 1024 * 1024 * 1024;
	return result;
}

inline int gigaBytes(int count)
{
	int result = count * 1024 * 1024 * 1024;
	return result;
}

inline int megaBytes(int count)
{
	int result = count * 1024 * 1024;
	return result;
}

inline int kiloBytes(int count)
{
	int result = count * 1024;
	return result;
}

//
//
//

struct MemoryBlock {
    int totalSize;

    void * permanent;
    int permanentIndex;
    int permanentSize;

    void * temporary;
    int temporaryIndex;
    int temporarySize;
    int markerStack[32];
    int markerStackIndex;

    void * dynamic;
    int dynamicSize;
    char * chunks;
    int chunkCount;
    int chunkSize;
};

extern MemoryBlock* globalMemory;

void initMemorySizes(MemoryBlock* memory, int permanentSize, int temporarySize, int dynamicSize, int chunkSize) {

	float chunkCount = (float)dynamicSize / chunkSize;
	if(chunkCount != (int)chunkCount) chunkCount = (int)chunkCount + 1;
	dynamicSize = chunkCount*chunkSize;
	
	memory->permanentSize = permanentSize;
	memory->temporarySize = temporarySize;
	memory->dynamicSize = dynamicSize;
	memory->chunkSize = chunkSize;
	memory->totalSize = permanentSize + temporarySize + dynamicSize + chunkCount;
}

void initMemory(MemoryBlock * memory, void* baseAddress = 0) {
    void * mem;
    if(baseAddress) mem = baseAddress;
    else mem = malloc(memory->totalSize);

    assert(mem);

    memory->permanent = mem;
    memory->permanentIndex = 0;

    memory->temporary = (char*)mem + memory->permanentSize;
    memory->temporaryIndex = 0;
    memory->markerStackIndex = 0;

    memory->dynamic = (char*)mem + memory->permanentSize + memory->temporarySize;
    memory->chunks = (char*)mem + memory->permanentSize + memory->temporarySize + memory->dynamicSize;
    memory->chunkCount = memory->chunkCount;

    // zeroMemory(memory->dynamic, memory->dynamicSize + memory->chunkCount);
}

// #define getPStruct(type, memory) 		(type*)(getPMemory(sizeof(type), memory)
// #define getPArray(type, count, memory) 	(type*)(getPMemory(sizeof(type) * count, memory))
// #define getTStruct(type, memory) 		(type*)(getTMemory(sizeof(type), memory))
// #define getTArray(type, count, memory) 	(type*)(getTMemory(sizeof(type) * count, memory))
// #define getTString(size, memory) 		(char*)(getTMemory(size, memory)) 
// #define getDStruct(type, memory) 		(type*)(getDMemory(sizeof(type), memory))
// #define getDArray(type, count, memory) 	(type*)(getDMemory(sizeof(type) * count, memory))

#define getPStruct(type) 		(type*)(getPMemory(sizeof(type))
#define getPArray(type, count) 	(type*)(getPMemory(sizeof(type) * count))
#define getTStruct(type) 		(type*)(getTMemory(sizeof(type)))
#define getTArray(type, count) 	(type*)(getTMemory(sizeof(type) * count))
#define getTString(size) 		(char*)(getTMemory(size)) 
#define getDStruct(type) 		(type*)(getDMemory(sizeof(type)))
#define getDArray(type, count) 	(type*)(getDMemory(sizeof(type) * count))

void *getPMemory(int size, MemoryBlock * memory = 0) {
    if(!memory) memory = globalMemory;
    assert(memory->permanentIndex + size <= memory->permanentSize);

    void * location = (char*)memory->permanent + memory->permanentIndex;
    memory->permanentIndex += size;
    return location;
}

void * getTMemory(int size, MemoryBlock * memory = 0) {
    if(!memory) memory = globalMemory;
    assert(memory->temporaryIndex + size <= memory->temporarySize);

    void * location = (char*)memory->temporary + memory->temporaryIndex;
    memory->temporaryIndex += size;
    return location;
}

void freeTMemory(int size, MemoryBlock * memory = 0)  {
    if(!memory) memory = globalMemory;
    memory->temporaryIndex -= size;
}

void clearTMemory(MemoryBlock * memory = 0) {
    if(!memory) memory = globalMemory;
    memory->temporaryIndex = 0;
    // zeroMemory(memory->temporary, memory->temporarySize);
}

void pushMarkerTMemory(MemoryBlock * memory = 0)  {
    if(!memory) memory = globalMemory;
    memory->markerStack[memory->markerStackIndex] = memory->temporaryIndex;
    memory->markerStackIndex++;
}

void popMarkerTMemory(MemoryBlock * memory = 0)  {
    if(!memory) memory = globalMemory;
    int size = memory->temporaryIndex - memory->markerStack[memory->markerStackIndex];
    memory->markerStackIndex--;
    freeTMemory(size, memory);
}

void * getDMemory(int size, MemoryBlock * memory = 0) {
    if(!memory) memory = globalMemory;
    char * dynamic = (char*)memory->dynamic;
    char * chunks = memory->chunks;
    int chunkSize = memory->chunkSize;

    int availableChunks = 0;
    bool foundMemorySlot = false;
    void * location = 0;
    for(int chunkIndex = 0; chunkIndex < memory->chunkCount; chunkIndex++) {

        if(chunks[chunkIndex] == 0) {
			availableChunks++;

			if(availableChunks*chunkSize >= size+sizeof(int)) {
				int firstChunk = chunkIndex-availableChunks+1;
				memSet(chunks + firstChunk, 255, sizeof(char)*availableChunks);
				location = dynamic + firstChunk*chunkSize;
				*((uint*)location) = availableChunks;
				location = (char*)location + sizeof(uint);
				foundMemorySlot = true;
				break;
			}
        } else  {
            availableChunks = 0;
            // chunkIndex += *((int*)(dynamic + (chunkIndex * chunkSize)));
        }
	}

	assert(foundMemorySlot);

    return location;
}

void freeDMemory(void * data, MemoryBlock * memory = 0) {
    if(!memory) memory = globalMemory;
	if(data == 0) return;

    char* dataStart = (char*)data - sizeof(int);

    int byteOffset = (char*)dataStart - (char*)memory->dynamic;
    int chunkIndex = byteOffset / memory->chunkSize;

    int chunkCount = *((int*)dataStart);

    zeroMemory(memory->chunks+chunkIndex, chunkCount);
}

//
//
//

uint flagSet(uint flags, int flagType)
{
	uint newFlags = flags | flagType;
	return newFlags;
}

bool flagCheck(uint flags, int flagType)
{
	bool result = (flags | flagType) == flags;
	return result;
}

//
//
//

enum TimerType {
	TIMER_TYPE_BEGIN,
	TIMER_TYPE_END,
};

struct TimerInfo {
	int initialised;

	const char* file;
	const char* function;
	int line, line2;
	uint type;
};

#define CYCLEBUFFERSIZE 120
struct TimerSlot {
	int timerIndex;
	u64 cycles;
	uint type;
	uint threadId;
};

struct Timings {
	u64 cycles;
	int hits;
};

struct DebugState {
	bool isInitialised;
	int timerInfoCount;
	TimerInfo timerInfos[256];
	Timings timings[256];
	TimerSlot* timerBuffer;
	int bufferSize;
	u64 bufferIndex;

	char* stringMemory;
	int stringMemorySize;
	int stringMemoryIndex;
};

// extern const int globalTimingsCount;
DebugState* globalDebugState;

inline uint getThreadID() {
	char *threadLocalStorage = (char *)__readgsqword(0x30);
	uint threadID = *(uint *)(threadLocalStorage + 0x48);

	return threadID;
}

void addTimerSlot(int timerIndex, int type) {
	// uint id = atomicAdd64(&globalDebugState->bufferIndex, 1);
	uint id = globalDebugState->bufferIndex++;
	TimerSlot* slot = globalDebugState->timerBuffer + id;
	slot->cycles = __rdtsc();
	slot->type = type;
	slot->threadId = getThreadID();
	slot->timerIndex = timerIndex;
}

void addTimerSlotAndInfo(int timerIndex, int type, const char* file, const char* function, int line) {

	TimerInfo* timerInfo = globalDebugState->timerInfos + timerIndex;

	if(!timerInfo->initialised) {
		timerInfo->initialised = true;
		timerInfo->file = file;
		timerInfo->function = function;
		timerInfo->line = line;
		timerInfo->type = type;
	}

	addTimerSlot(timerIndex, type);
}

struct TimerBlock {
	int counter;

	TimerBlock(int counter, const char* file, const char* function, int line) {

		this->counter = counter;
		addTimerSlotAndInfo(counter, TIMER_TYPE_BEGIN, file, function, line);
	}

	~TimerBlock() {
		addTimerSlot(counter, TIMER_TYPE_END);
	}
};

#define TIMER_BLOCK() \
	TimerBlock timerBlock##__LINE__(__COUNTER__, __FILE__, __FUNCTION__, __LINE__);

#define TIMER_BEGIN(ID) \
	const int timerCounter##ID = __COUNTER__; \
	addTimerSlotAndInfo(timerCounter##ID, TIMER_TYPE_BEGIN, __FILE__, __FUNCTION__, __LINE__); 

#define TIMER_END(ID) \
	addTimerSlot(timerCounter##ID, TIMER_TYPE_END);




struct Statistic {
	r64 min;
	r64 max;
	r64 avg;
	int count;
};

void beginStatistic(Statistic* stat) {
	stat->min = DBL_MAX; 
	stat->max = -DBL_MAX; 
	stat->avg = 0; 
	stat->count = 0; 
}

void updateStatistic(Statistic* stat, r64 value) {
	if(value < stat->min) stat->min = value;
	if(value > stat->max) stat->max = value;
	stat->avg += value;
	++stat->count;
}

void endStatistic(Statistic* stat) {
	stat->avg /= stat->count;
}

struct TimerStatistic {
	u64 cycles;
	int timerIndex;
};
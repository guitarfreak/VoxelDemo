
enum StructType {
	STRUCTTYPE_INT = 0,
	STRUCTTYPE_FLOAT,
	STRUCTTYPE_CHAR,
	STRUCTTYPE_BOOL,

	STRUCTTYPE_VEC3,
	STRUCTTYPE_ENTITY,

	STRUCTTYPE_SIZE,
};

enum ArrayType {
	ARRAYTYPE_CONSTANT, 
	ARRAYTYPE_DYNAMIC, 
	ARRAYTYPE_STRING, 

	ARRAYTYPE_SIZE, 
};

struct ArrayInfo {
	int type;
	bool isOffset;

	union {
		int size;
		int offset;
	};
};

struct StructMemberInfo {
	char* name;
	int type;
	int offset;
	int arrayCount;
	ArrayInfo arrays[2];
};

StructMemberInfo initMemberInfo(char* name, int type, int offset) {
	StructMemberInfo info;
	info.name = name;
	info.type = type;
	info.offset = offset;
	info.arrayCount = 0;

	return info;
}

StructMemberInfo initMemberInfo(char* name, int type, int offset, ArrayInfo aInfo1, ArrayInfo aInfo2 = {-1}) {
	StructMemberInfo info = initMemberInfo(name, type, offset);
	info.arrayCount = 1;
	if(aInfo2.type != -1) info.arrayCount = 2;

	info.arrays[0] = aInfo1;
	info.arrays[1] = aInfo2;

	return info;
}

StructMemberInfo vec3StructMemberInfos[] = {
	initMemberInfo("x", STRUCTTYPE_FLOAT, offsetof(Vec3, x)),
	initMemberInfo("y", STRUCTTYPE_FLOAT, offsetof(Vec3, y)),
	initMemberInfo("z", STRUCTTYPE_FLOAT, offsetof(Vec3, z)),
};

StructMemberInfo entityStructMemberInfos[] = {
	initMemberInfo("init", STRUCTTYPE_INT, offsetof(Entity, init)),
	initMemberInfo("type", STRUCTTYPE_INT, offsetof(Entity, type)),
	initMemberInfo("id", STRUCTTYPE_INT, offsetof(Entity, id)),
	initMemberInfo("name", STRUCTTYPE_CHAR, offsetof(Entity, name), {ARRAYTYPE_CONSTANT, 0, memberSize(Entity, name)}),
	initMemberInfo("pos", STRUCTTYPE_VEC3, offsetof(Entity, pos)),
	initMemberInfo("dir", STRUCTTYPE_VEC3, offsetof(Entity, dir)),
	initMemberInfo("rot", STRUCTTYPE_VEC3, offsetof(Entity, rot)),
	initMemberInfo("rotAngle", STRUCTTYPE_FLOAT, offsetof(Entity, rotAngle)),
	initMemberInfo("dim", STRUCTTYPE_VEC3, offsetof(Entity, dim)),
	initMemberInfo("camOff", STRUCTTYPE_VEC3, offsetof(Entity, camOff)),
	initMemberInfo("vel", STRUCTTYPE_VEC3, offsetof(Entity, vel)),
	initMemberInfo("acc", STRUCTTYPE_VEC3, offsetof(Entity, acc)),
	initMemberInfo("movementType", STRUCTTYPE_INT, offsetof(Entity, movementType)),
	initMemberInfo("spatial", STRUCTTYPE_INT, offsetof(Entity, spatial)),
	initMemberInfo("deleted", STRUCTTYPE_BOOL, offsetof(Entity, deleted)),
	initMemberInfo("isMoving", STRUCTTYPE_BOOL, offsetof(Entity, isMoving)),
	initMemberInfo("isColliding", STRUCTTYPE_BOOL, offsetof(Entity, isColliding)),
	initMemberInfo("exploded", STRUCTTYPE_BOOL, offsetof(Entity, exploded)),
	initMemberInfo("playerOnGround", STRUCTTYPE_BOOL, offsetof(Entity, playerOnGround)),
};

struct StructInfo {
	int size;
	int memberCount;
	StructMemberInfo* list;
};

StructInfo structInfos[] = {
	{ sizeof(int), 0 },
	{ sizeof(float), 0 },
	{ sizeof(char), 0 },
	{ sizeof(bool), 0 },
	{ sizeof(Vec3), arrayCount(vec3StructMemberInfos), vec3StructMemberInfos }, 
	{ sizeof(Entity), arrayCount(entityStructMemberInfos), entityStructMemberInfos }, 
};

bool typeIsPrimitive(int type) {
	return structInfos[type].memberCount == 0;
}

void printMember(int type, char* data) {
	switch(type) {
		case STRUCTTYPE_INT: printf("%i", *(int*)data); break;
		case STRUCTTYPE_FLOAT: printf("%f", *(float*)data); break;
		case STRUCTTYPE_CHAR: printf("%c", *data); break;
		case STRUCTTYPE_BOOL: printf((*((bool*)data)) ? "true" : "false"); break;
		default: break;
	};
}

void printType(int structType, char* data, int depth = 0) {
	StructInfo* info = structInfos + structType;

	if(depth == 0) printf("{\n");

	int arrayPrintLimit = 5;
	int indent = 2;

	if(typeIsPrimitive(structType)) {
		printMember(structType, data);
		printf(", ");
		if(depth == 0) printf("\n");
		return;
	}

	for(int i = 0; i < info->memberCount; i++) {
		StructMemberInfo member = info->list[i];

		bool isPrimitive = typeIsPrimitive(member.type);

		if(depth == 0) printf("%*s", indent*(depth+1), "");

		if(member.arrayCount == 0) {
			if(!isPrimitive) printf("{ ");
			printType(member.type, data + member.offset, depth + 1);
			if(!isPrimitive) printf("}, ");
			if(depth == 0) printf("\n");
		} else {
			ArrayInfo aInfo = member.arrays[0];

			if(aInfo.type == ARRAYTYPE_STRING) {
				printf("\"%s\"", *((char**)(data + member.offset)));
			} else {
				if(member.arrayCount == 1) {
					int typeSize = structInfos[member.type].size;
					int arraySize = aInfo.isOffset ? *(int*)(data + aInfo.offset) : aInfo.size;
					char* arrayBase = (aInfo.type == ARRAYTYPE_CONSTANT) ? data + member.offset :*((char**)(data + member.offset));

					printf("[ ");
					int size = min(arraySize, arrayPrintLimit);
					for(int i = 0; i < size; i++) {
						char* value = arrayBase + (typeSize*i);
						if(!isPrimitive) printf("{ ");
						printType(member.type, value, depth + 1);
						if(!isPrimitive) printf("}, ");
					}
					if(arraySize > arrayPrintLimit) printf("...");
					printf("]");
				} else {
					int typeSize = structInfos[member.type].size;
					int arraySize = aInfo.isOffset ? *(int*)(data + aInfo.offset) : aInfo.size;
					char* arrayBase = (aInfo.type == ARRAYTYPE_CONSTANT) ? data + member.offset :*((char**)(data + member.offset));

					ArrayInfo aInfo2 = member.arrays[1];
					int arraySize2 = aInfo2.isOffset ? *(int*)(data + aInfo2.offset) : aInfo2.size;

					printf("[ ");
					for(int arrayIndex = 0; arrayIndex < arraySize; arrayIndex++) {
						char* x;
						if(aInfo2.type == ARRAYTYPE_CONSTANT) x = (arrayBase + (arrayIndex*arraySize2*typeSize));
						else x = (arrayBase + (arrayIndex*sizeof(int*)));
						char* arrayBase2 = (aInfo2.type == ARRAYTYPE_CONSTANT) ? x : *((char**)(x));

						printf("[ ");
						int size = min(arraySize2, arrayPrintLimit);
						for(int i = 0; i < size; i++) {
							char* value = arrayBase2 + (typeSize*i);
							if(!isPrimitive) printf("{ ");
							printType(member.type, value, depth + 1);
							if(!isPrimitive) printf("}, ");
						}
						if(arraySize2 > arrayPrintLimit) printf("...");
						printf("], ");
					}
					printf("]");

				}

			}

			if(depth == 0) printf("\n");
		}
	}

	if(depth == 0) printf("}\n");

}



#define DEBUG_NOTE_LENGTH 50
#define DEBUG_NOTE_DURATION 3

struct Asset;
struct DebugState {
	Asset* assets;
	int assetCount;

	i64 lastTimeStamp;
	f64 dt;
	f64 time;

	f64 debugTime;
	f64 debugRenderTime;

	bool showMenu;
	bool showStats;
	bool showConsole;
	bool showHud;

	DrawCommandList commandListDebug;
	Input* input;

	Timer* timer;
	Timings timings[120][32];
	Statistic statistics[120][32];
	int cycleIndex;
	bool stopCollating;

	bool setPause;
	bool setPlay;
	bool noCollating;
	int lastCycleIndex;
	GraphSlot* savedBuffer;
	int savedBufferIndex;
	int savedBufferMax;
	int savedBufferCount;
	int lastBufferIndex;

	GraphSlot graphSlots[16][8]; // threads, stackIndex
	int graphSlotCount[16];

	//

	int mode;
	int graphSortingIndex;
	
	double timelineCamPos;
	double timelineCamSize;

	float lineGraphCamPos;
	float lineGraphCamSize;
	float lineGraphHeight;
	int lineGraphHighlight;

	//

	GuiInput gInput;
	Gui* gui;
	Gui* gui2;
	float guiAlpha;

	Input* recordedInput;
	int inputCapacity;
	bool recordingInput;
	int inputIndex;

	char* snapShotMemory[8];
	int snapShotCount;
	int snapShotMemoryIndex;

	bool playbackInput;
	int playbackIndex;
	bool playbackSwapMemory;
	bool playbackPause;
	bool playbackBreak;
	int playbackBreakIndex;

	Console console;

	char* notificationStack[10];
	float notificationTimes[10];
	int notificationCount;

	char* infoStack[10];
	int infoStackCount;
};

void addDebugNote(char* string, float duration = DEBUG_NOTE_DURATION) {
	DebugState* ds = globalDebugState;

	assert(strLen(string) < DEBUG_NOTE_LENGTH);
	if(ds->notificationCount >= arrayCount(ds->notificationStack)) return;

	int count = ds->notificationCount;
	strClear(ds->notificationStack[count]);
	ds->notificationTimes[count] = duration;
	strCpy(ds->notificationStack[count], string);
	ds->notificationCount++;
}

void addDebugInfo(char* string) {
	DebugState* ds = globalDebugState;

	if(ds->infoStackCount >= arrayCount(ds->infoStack)) return;
	ds->infoStack[ds->infoStackCount++] = string;
}



struct Asset {
	int index;
	int folderIndex;
	char* filePath;
	FILETIME lastWriteTime;
};

void initWatchFolders(HANDLE* folderHandles, Asset* assets, int* assetCount) {
	for(int i = 0; i < 3; i++) {
		HANDLE fileChangeHandle = FindFirstChangeNotification(watchFolders[i], false, FILE_NOTIFY_CHANGE_LAST_WRITE);

		if(fileChangeHandle == INVALID_HANDLE_VALUE) {
			printf("Could not set folder change notification.\n");
		}

		folderHandles[i] = fileChangeHandle;
	}

	int fileCounts[] = {TEXTURE_SIZE, CUBEMAP_SIZE, BX_Size};
	char** paths[] = {texturePaths, cubeMapPaths, (char**)textureFilePaths};
	for(int folderIndex = 0; folderIndex < arrayCount(watchFolders); folderIndex++) {

		int fileCount = fileCounts[folderIndex];

		for(int i = 0; i < fileCount; i++) {
			char* path = paths[folderIndex][i];
			Asset* asset = assets + *assetCount; 
			*assetCount = (*assetCount) + 1;
			asset->lastWriteTime = getLastWriteTime(path);
			asset->index = i;
			asset->filePath = getPArray(char, strLen(path) + 1);
			strCpy(asset->filePath, path);
			asset->folderIndex = folderIndex;
		}
	}

}

void reloadChangedFiles(HANDLE* folderHandles, Asset* assets, int assetCount) {
	// Todo: Get ReadDirectoryChangesW to work instead of FindNextChangeNotification.

	/*
	// HANDLE directory = CreateFile("..\\data\\Textures", FILE_LIST_DIRECTORY, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	// HANDLE directory = CreateFile("C:\\Projects", FILE_LIST_DIRECTORY, FILE_SHARE_WRITE | FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,  OPEN_EXISTING,  FILE_FLAG_BACKUP_SEMANTICS, NULL);
	// HANDLE directory = CreateFile("..\\data\\Textures", FILE_LIST_DIRECTORY, FILE_SHARE_WRITE | FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,  OPEN_EXISTING,  FILE_FLAG_BACKUP_SEMANTICS, NULL);
	HANDLE directory = CreateFile("..\\data\\Textures", FILE_LIST_DIRECTORY, FILE_SHARE_WRITE | FILE_SHARE_READ | FILE_SHARE_DELETE, NULL,  OPEN_EXISTING,  FILE_FLAG_BACKUP_SEMANTICS, NULL);
	// HANDLE directory = CreateFile("C:\\Projects\\", FILE_LIST_DIRECTORY, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if(directory == INVALID_HANDLE_VALUE) {
		printf("Could not open directory.\n");
	}

	FILE_NOTIFY_INFORMATION notifyInformation[1024];
	DWORD bytesReturned = 0;
	bool result = ReadDirectoryChangesW(directory, (LPVOID)&notifyInformation, sizeof(notifyInformation), false, FILE_NOTIFY_CHANGE_LAST_WRITE, &bytesReturned, 0, 0);
	CloseHandle(directory);
	*/

	DWORD fileStatus = WaitForMultipleObjects(arrayCount(watchFolders), folderHandles, false, 0);
	if(valueBetweenInt(fileStatus, WAIT_OBJECT_0, WAIT_OBJECT_0 + 9)) {
		// Note: WAIT_OBJECT_0 is defined as 0, so we can use it as an index.
		int folderIndex = fileStatus;

		FindNextChangeNotification(folderHandles[folderIndex]);


		for(int i = 0; i < assetCount; i++) {
			Asset* asset = assets + i;
			if(asset->folderIndex != folderIndex) continue;

			FILETIME newWriteTime = getLastWriteTime(asset->filePath);
			if(CompareFileTime(&asset->lastWriteTime, &newWriteTime) != 0) {

				if(folderIndex == 0) {
					loadTextureFromFile(globalGraphicsState->textures + asset->index, texturePaths[asset->index], -1, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE, true);

				} else if(folderIndex == 1) {
						loadCubeMapFromFile(globalGraphicsState->cubeMaps + asset->index, (char*)cubeMapPaths[asset->index], 5, INTERNAL_TEXTURE_FORMAT, GL_RGBA, GL_UNSIGNED_BYTE, true);

				} else if(folderIndex == 2) {
					loadVoxelTextures((char*)minecraftTextureFolderPath, INTERNAL_TEXTURE_FORMAT, true, asset->index);

				}

				asset->lastWriteTime = newWriteTime;
			}
		}
	}

}

struct Asset;
struct DebugState {
	Asset* assets;
	int assetCount;

	LONGLONG lastTimeStamp;
	float dt;
	float time;

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
	TimerSlot* savedBuffer[120];
	int savedBufferCounts[120];
	int savedBufferMax;

	u64 mainThreadSlotCycleRange[120][2];
	int graphSortingIndex;

	GuiInput gInput;
	Gui* gui;
	Gui* gui2;
	float guiAlpha;

	Input* recordedInput;
	int inputCapacity;
	bool recordingInput;
	int inputIndex;

	bool playbackStart;
	bool playbackInput;
	int playbackIndex;
	bool playbackSwapMemory;

	// Console.

	Console console;
};



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
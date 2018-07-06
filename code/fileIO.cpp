
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

inline int fileSize(FILE* file) {
	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	fseek(file, 0, SEEK_SET);

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

	myAssert(offsetInBytes+sizeInBytes < size);

	fseek(file, offsetInBytes, SEEK_SET);
	fread(buffer, sizeInBytes, 1, file);

	fclose(file);
	return size;
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

	myAssert(offsetInBytes+sizeInBytes < size);

	fseek(file, offsetInBytes, SEEK_SET);
	fwrite(buffer, sizeInBytes, 1, file);

	fclose(file);
	return size;
}

void writeDataToFile(char* data, int size, char* fileName) {
	FILE* file = fopen(fileName, "wb");
	fwrite(data, size, 1, file);
	fclose(file);
}

void readDataFromFile(char* data, char* fileName) {
	FILE* file = fopen(fileName, "rb");

	int size = fileSize(file);
	fread(data, size, 1, file);
	fclose(file);
}

char* getFileExtension(char* file) {
	char* ext = strrchr(file, '.');
	if(!ext) return 0;
	else ext++;

	return ext;
}


// WAVE_FORMAT_PCM
// WAVE_FORMAT_IEEE_FLOAT

struct WaveFileHeader {
	char chunkId[4];
	int chunkSize;
	char format[4];

	char subchunk1Id[4];
	int subchunk1Size;
	short audioFormat;
	short numChannels;
	int sampleRate;
	int byteRate;
	short blockAlign;
	short bitsPerSample;
	// short cbSize;
	// short validBitsPerSample;
	// int channelMask;
	// char subFormat[16];

	char subchunk2Id[4];
	int subchunk2Size;
	short data;
};

struct Audio {
	char* name;

	char* file;

	int channels;
	int sampleRate;
	int bitsPerSample;

	short* data;
	int frameCount;
	int totalLength;
};

struct Track {
	Audio* audio;

	float volume;

	bool used;

	bool playing;
	bool paused;
	int index;
};

struct AudioState {
	IMMDeviceEnumerator* deviceEnumerator;
	IMMDevice* immDevice;
	IAudioClient* audioClient;
	IAudioRenderClient* renderClient;
	float latencyInSeconds;

	WAVEFORMATEX* waveFormat;
	uint bufferFrameCount;

	//

	Audio* files;
	int fileCount;
	int fileCountMax;

	// Mixer.

	float masterVolume;

	Track tracks[10];
	// int activeTracks;
};

void addTrack(AudioState* as, Audio* audio) {
	Track* track = 0;
	for(int i = 0; i < arrayCount(as->tracks); i++) {
		if(!as->tracks[i].used) {
			track = as->tracks + i;
		}
	}

	if(!track) return;

	track->used = true;
	track->audio = audio;
	track->playing = true;
	track->paused = false;
	track->index = 0;
	
	track->volume = 1.0f;
}

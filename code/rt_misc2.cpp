
//

char* fillString(char* text, ...) {
	va_list vl;
	va_start(vl, text);

	int length = strLen(text);
	char* buffer = getTStringX(length+1);

	char valueBuffer[20] = {};

	int ti = 0;
	int bi = 0;
	while(true) {
		char t = text[ti];

		if(text[ti] == '%' && text[ti+1] == '.') {
			float v = va_arg(vl, double);
			floatToStr(valueBuffer, v, charToInt(text[ti+2]));
			int sLen = strLen(valueBuffer);
			memCpy(buffer + bi, valueBuffer, sLen);

			ti += 4;
			bi += sLen;
			getTStringX(sLen);
		} else if(text[ti] == '%' && text[ti+1] == 'f') {
			float v = va_arg(vl, double);
			floatToStr(valueBuffer, v, 2);
			int sLen = strLen(valueBuffer);
			memCpy(buffer + bi, valueBuffer, sLen);

			ti += 2;
			bi += sLen;
			getTStringX(sLen);
		} else if(text[ti] == '%' && text[ti+1] == 'i') {
			if(text[ti+2] == '6') {
				// 64 bit signed integer.

				// assert(text[ti+3] == '4');

				i64 v = va_arg(vl, i64);
				intToStr(valueBuffer, v);
				int sLen = strLen(valueBuffer);

				if(text[ti+4] == '.') {
					ti += 1;

					int digitCount = intDigits(v);
					int commaCount = digitCount/3;
					if(digitCount%3 == 0) commaCount--;
					for(int i = 0; i < commaCount; i++) {
						strInsert(valueBuffer, sLen - (i+1)*3 - i, ',');
						sLen++;
					}
				}

				memCpy(buffer + bi, valueBuffer, sLen);
				ti += 4;
				bi += sLen;
				getTStringX(sLen);
			} else {
				// 32 bit signed integer.
				int v = va_arg(vl, int);
				intToStr(valueBuffer, v);
				int sLen = strLen(valueBuffer);

				if(text[ti+2] == '.') {
					ti += 1;

					int digitCount = intDigits(v);
					int commaCount = digitCount/3;
					if(digitCount%3 == 0) commaCount--;
					for(int i = 0; i < commaCount; i++) {
						strInsert(valueBuffer, sLen - (i+1)*3 - i, ',');
						sLen++;
					}
				}

				memCpy(buffer + bi, valueBuffer, sLen);

				ti += 2;
				bi += sLen;
				getTStringX(sLen);
			}
		} else if(text[ti] == '%' && text[ti+1] == 's') {
			char* str = va_arg(vl, char*);
			int sLen = strLen(str);
			memCpy(buffer + bi, str, sLen);

			ti += 2;
			bi += sLen;
			getTStringX(sLen);
		} else if(text[ti] == '%' && text[ti+1] == 'b') {
			bool str = va_arg(vl, bool);
			char* s = str == 1 ? "true" : "false";
			int sLen = strLen(s);
			memCpy(buffer + bi, s, sLen);

			ti += 2;
			bi += sLen;
			getTStringX(sLen);
		} else if(text[ti] == '%' && text[ti+1] == '%') {
			buffer[bi++] = '%';
			ti += 2;
			getTStringX(1);
		} else {
			buffer[bi++] = text[ti++];
			getTStringX(1);

			if(buffer[bi-1] == '\0') break;
		}
	}

	return buffer;
}

//
// Noise.
//

void whiteNoise(Rect region, int sampleCount, Vec2* samples) {
	for(int i = 0; i < sampleCount; ++i) {
		Vec2 randomPos = vec2(randomInt(region.min.x, region.max.x), 
		                      randomInt(region.min.y, region.max.y));
		samples[i] = randomPos;
	}
}

// Wraps automatically.
int blueNoise(Rect region, float radius, Vec2** noiseSamples) {
	bool wrapAround = true;

	Vec2 dim = rectDim(region);
	float cs = radius/M_SQRT2;

	Vec2i gdim = vec2i(roundUpf(dim.w/cs), roundUpf(dim.h/cs));
	int gridSize = gdim.w*gdim.h;

	*noiseSamples = mallocArray(Vec2, gridSize);
	Vec2* samples = *noiseSamples;
	int sampleCount = 0;

	int testCount = 30;

	int* grid = mallocArray(int, gridSize);
	for(int i = 0; i < gridSize; i++) grid[i] = -1;

	int* activeList = mallocArray(int, gridSize);
	int activeListCount = 0;

	// Setup first sample randomly.
	samples[sampleCount++] = vec2(randomFloat(0, dim.w), randomFloat(0, dim.h));
	activeList[activeListCount++] = 0;

	Vec2 pos = samples[0];
	Vec2i gridPos = vec2i(pos/cs);
	grid[gridPos.y*gdim.w + gridPos.x] = 0;

	Rect regionOrigin = rectTrans(region, region.min*-1);
	Vec2 regionDim = rectDim(regionOrigin);
	while(activeListCount > 0) {

		int activeIndex = randomInt(0, activeListCount-1);
		int sampleIndex = activeList[activeIndex];
		Vec2 activeSample = samples[sampleIndex];

		for(int i = 0; i < testCount; ++i) {
			float angle = randomFloat(0, M_2PI);
			float distance = randomFloat(radius, radius*2);
			Vec2 newSample = activeSample + angleToDir(angle)*distance;

			if(!pointInRect(newSample, regionOrigin)) continue;

			// Search around sample point.

			int minx = roundDownf((newSample.x - radius*2.0f)/cs);
			int miny = roundDownf((newSample.y - radius*2.0f)/cs);
			int maxx = roundDownf((newSample.x + radius*2.0f)/cs);
			int maxy = roundDownf((newSample.y + radius*2.0f)/cs);

			bool validPosition = true;
			for(int y = miny; y <= maxy; ++y) {
				for(int x = minx; x <= maxx; ++x) {
					int mx = mod(x, gdim.w);
					int my = mod(y, gdim.h);
					int index = grid[my*gdim.w+mx];

					if(index > -1) {
						Vec2 s = samples[index];

						// Wrap sample position if we check outside boundaries.
						if(mx != x) s.x += mx > x ? -regionDim.w : regionDim.w;
						if(my != y) s.y += my > y ? -regionDim.h : regionDim.h;

						float distance = len(s - newSample);
						if(distance < radius) {
							validPosition = false;
							break;
						}
					}
				}
				if(!validPosition) break;
			}

			if(validPosition) {
				samples[sampleCount] = newSample;
				activeList[activeListCount++] = sampleCount;

				Vec2i gridPos = vec2i(newSample/cs);
				grid[gridPos.y*gdim.w + gridPos.x] = sampleCount;
				sampleCount++;
			}
		}

		// delete active sample after testCount times
		activeList[activeIndex] = activeList[activeListCount-1];
		activeListCount--;
	}

	for(int i = 0; i < sampleCount; ++i) samples[i] += region.min;

	free(grid);
	free(activeList);

	return sampleCount;
}


static int SEED = 0;

static int hash[] = {208,34,231,213,32,248,233,56,161,78,24,140,71,48,140,254,245,255,247,247,40,
	185,248,251,245,28,124,204,204,76,36,1,107,28,234,163,202,224,245,128,167,204,
	9,92,217,54,239,174,173,102,193,189,190,121,100,108,167,44,43,77,180,204,8,81,
	70,223,11,38,24,254,210,210,177,32,81,195,243,125,8,169,112,32,97,53,195,13,
	203,9,47,104,125,117,114,124,165,203,181,235,193,206,70,180,174,0,167,181,41,
	164,30,116,127,198,245,146,87,224,149,206,57,4,192,210,65,210,129,240,178,105,
	228,108,245,148,140,40,35,195,38,58,65,207,215,253,65,85,208,76,62,3,237,55,89,
	232,50,217,64,244,157,199,121,252,90,17,212,203,149,152,140,187,234,177,73,174,
	193,100,192,143,97,53,145,135,19,103,13,90,135,151,199,91,239,247,33,39,145,
	101,120,99,3,186,86,99,41,237,203,111,79,220,135,158,42,30,154,120,67,87,167,
	135,176,183,191,253,115,184,21,233,58,129,233,142,39,128,211,118,137,139,255,
	114,20,218,113,154,27,127,246,250,1,8,198,250,209,92,222,173,21,88,102,219};

int noise2(int x, int y)
{
	int tmp = hash[(y + SEED) % 256];
	return hash[(tmp + x) % 256];
}

float lin_inter(float x, float y, float s)
{
	return x + s * (y-x);
}

float smooth_inter(float x, float y, float s)
{
	return lin_inter(x, y, s * s * (3-2*s));
}

float noise2d(float x, float y)
{
	int x_int = x;
	int y_int = y;
	float x_frac = x - x_int;
	float y_frac = y - y_int;
	int s = noise2(x_int, y_int);
	int t = noise2(x_int+1, y_int);
	int u = noise2(x_int, y_int+1);
	int v = noise2(x_int+1, y_int+1);
	float low = smooth_inter(s, t, x_frac);
	float high = smooth_inter(u, v, x_frac);
	return smooth_inter(low, high, y_frac);
}

float perlin2d(float x, float y, float freq, int depth)
{
	float xa = x*freq;
	float ya = y*freq;
	float amp = 1.0;
	float fin = 0;
	float div = 0.0;

	int i;
	for(i=0; i<depth; i++)
	{
		div += 256 * amp;
		fin += noise2d(xa, ya) * amp;
		amp /= 2;
		xa *= 2;
		ya *= 2;
	}

	return fin/div;
}

//
// Sort.
//

struct SortPair {
	float key;
	int index;
};

void bubbleSort(int* list, int size) {
	for(int off = 0; off < size-2; off++) {
		bool sw = false;

		for(int i = 0; i < size-1 - off; i++) {
			if(list[i+1] < list[i]) {
				swap(&off, &size);
				sw = true;
			}
		}

		if(!sw) break;
	}
}

void bubbleSort(SortPair* list, int size, bool sortDirection = false) {
	for(int off = 0; off < size-1; off++) {
		bool sw = false;

		for(int i = 0; i < size-1 - off; i++) {
			bool result = sortDirection ? (list[i+1].key > list[i].key) : 
										  (list[i+1].key < list[i].key);
			if(result) {
				swap(list + i, list + (i+1));
				sw = true;
			}
		}

		if(!sw) break;
	}
}

void mergeSort(int* list, int size) {
	// int* buffer = getTArray(int, size);
	int* buffer = (int*)malloc(sizeof(int)*size);
	int stage = 0;

	for(;;) {
		stage++;
		int stageSize = 1 << stage;
		int splitSize = 1 << stage-1;

		int* src = stage%2 == 0 ? list : buffer;
		int* dest = stage%2 == 0 ? buffer : list;

		int count = ceil(size/(float)splitSize);
		if(count <= 1) {
			if(stage%2 == 0) memCpy(list, buffer, size);
			break;
		}

		for(int i = 0; i < size; i += stageSize) {
			int* fbuf = src + i;
			int fi = 0;
			int remainder = size - i;
			int as0 = min(splitSize, remainder);
			int as1 = min(splitSize, remainder-splitSize, 0);
			int ai0 = 0; 
			int ai1 = 0;
			int* buf0 = dest + i;
			int* buf1 = buf0 + as0;

			for(;;) {
				if(ai0 < as0 && ai1 < as1) {
					if(buf0[ai0] < buf1[ai1]) fbuf[fi++] = buf0[ai0++];
					else 					  fbuf[fi++] = buf1[ai1++];
				} 
				else if(ai0 < as0) fbuf[fi++] = buf0[ai0++];
				else if(ai1 < as1) fbuf[fi++] = buf1[ai1++];
				else break;
			}
		}
	}

	free(buffer);
}


// sorts in bytes
void radixSort(int* list, int size) {
	// int* buffer = getTArray(int, size);
	int* buffer = (int*)malloc(sizeof(int)*size);
	int stageCount = 4;

	for(int stage = 0; stage < stageCount; stage++) {
		int* src = stage%2 == 0 ? list : buffer;
		int* dst = stage%2 == 0 ? buffer : list;
		int bucket[257] = {};
		int offset = 8*stage;

		// count 
		for(int i = 0; i < size; i++) {
			uchar byte = src[i] >> offset;
			bucket[byte+1]++;
		}

		// turn sizes into offsets
		for(int i = 0; i < 256-1; i++) {
			bucket[i+1] += bucket[i];
		}

		for(int i = 0; i < size; i++) {
			uchar byte = src[i] >> offset;
			dst[bucket[byte]] = src[i];
			bucket[byte]++;
		}
	}

	free(buffer);
}

void radixSortSimd(int* list, int size) {
	// int* buffer = getTArray(int, size);
	int* buffer = (int*)malloc(sizeof(int)*size);

	int stageCount = 4;

	for(int stage = 0; stage < stageCount; stage++) {
		int* src = stage%2 == 0 ? list : buffer;
		int* dst = stage%2 == 0 ? buffer : list;
		int bucket[257] = {};
		int offset = 8*stage;

		__m128i stageS = _mm_set1_epi32(stage*8);
		__m128i one = _mm_set1_epi32(1);

		if(size % 4 != 0) {
			int rest = size % 4;
			for(int i = 0; i < rest; i++) {
				uchar byte = src[i] >> offset;
				bucket[byte+1]++;
			}
		}

		for(int i = 0; i < size; i += 4) {
			__m128i byte = _mm_set_epi32(src[i], src[i+1], src[i+2], src[i+3]);
			byte = _mm_srl_epi32(byte, stageS);
			byte = _mm_add_epi32(byte, one);
			bucket[byte.m128i_u8[0]] = bucket[byte.m128i_u8[0]] + 1;
			bucket[byte.m128i_u8[1]] = bucket[byte.m128i_u8[1]] + 1;
			bucket[byte.m128i_u8[2]] = bucket[byte.m128i_u8[2]] + 1;
			bucket[byte.m128i_u8[3]] = bucket[byte.m128i_u8[3]] + 1;
		}

		// turn sizes into offsets
		for(int i = 0; i < 256-1; i++) {
			bucket[i+1] += bucket[i];
		}

		for(int i = 0; i < size; i++) {
			uchar byte = src[i] >> offset;
			dst[bucket[byte]] = src[i];
			bucket[byte]++;
		}
	}

	free(buffer);
}

void radixSortPair(SortPair* list, int size) {
	// SortPair* buffer = getTArray(SortPair, size);
	SortPair* buffer = (SortPair*)malloc(sizeof(SortPair)*size);

	int stageCount = 4;

	for(int stage = 0; stage < stageCount; stage++) {
		SortPair* src = stage%2 == 0 ? list : buffer;
		SortPair* dst = stage%2 == 0 ? buffer : list;
		int bucket[257] = {};

		// count 
		for(int i = 0; i < size; i++) {
			uchar byte = *((int*)&src[i].key) >> (8*stage);
			bucket[byte+1]++;
		}

		// turn sizes into offsets
		for(int i = 0; i < 256-1; i++) {
			bucket[i+1] += bucket[i];
		}

		for(int i = 0; i < size; i++) {
			uchar byte = *((int*)&src[i].key) >> (8*stage);
			dst[bucket[byte]] = src[i];
			bucket[byte]++;
		}
	}

	free(buffer);
}

//
//
//

struct GraphCam {
	double x, y, w, h;
	double left, bottom, right, top;
	double xMin, xMax, yMin, yMax;
	Rect viewPort;
};

void graphCamSetBoundaries(GraphCam* cam, double xMin, double xMax, double yMin, double yMax) {
	cam->xMin = xMin;
	cam->xMax = xMax;
	cam->yMin = yMin;
	cam->yMax = yMax;	
}

void graphCamInit(GraphCam* cam, double x, double y, double w, double h, double xMin, double xMax, double yMin, double yMax) {
	cam->x = x;
	cam->y = y;
	cam->w = w;
	cam->h = h;
	graphCamSetBoundaries(cam, xMin, xMax, yMin, yMax);
}

void graphCamInit(GraphCam* cam, double xMin, double xMax, double yMin, double yMax) {
	double w = xMax - xMin;
	double h = yMax - yMin;
	graphCamInit(cam, xMin+w/2, yMin+h/2, w, h, xMin, xMax, yMin, yMax);
}

void graphCamSetViewPort(GraphCam* cam, Rect viewPort) {
	cam->viewPort = viewPort;
}

inline double graphCamLeft(GraphCam* cam) { return cam->x - cam->w/2; }
inline double graphCamRight(GraphCam* cam) { return cam->x + cam->w/2; }
inline double graphCamBottom(GraphCam* cam) { return cam->y - cam->h/2; }
inline double graphCamTop(GraphCam* cam) { return cam->y + cam->h/2; }

void graphCamUpdateSides(GraphCam* cam) {
	cam->left = graphCamLeft(cam);
	cam->bottom = graphCamBottom(cam);
	cam->right = graphCamRight(cam);
	cam->top = graphCamTop(cam);
}

void graphCamSizeClamp(GraphCam* cam, double wMin, double hMin, double wMax = -1, double hMax = -1) {
	if(wMax == -1) wMax = cam->xMax - cam->xMin;
	if(hMax == -1) hMax = cam->yMax - cam->yMin;

	clamp(&cam->w, wMin, wMax);
	clamp(&cam->h, hMin, hMax);
}

void graphCamScaleToPos(GraphCam* cam, int xAmount, double xScale, double xClamp, int yAmount, double yScale, double yClamp, Vec2 pos) {
	double diff, offset;
	float posOffset;
	double mod;

	diff = cam->w;
	mod = pow(xScale, abs(xAmount));
	offset = xAmount > 0 ? mod : 1/mod;
	cam->w *= offset;
	clampMin(&cam->w, xClamp);
	diff -= cam->w;
	posOffset = mapRange(pos.x, cam->viewPort.left, cam->viewPort.right, -0.5f, 0.5f);
	cam->x += diff * posOffset;

	diff = cam->h;
	mod = pow(yScale, abs(yAmount));
	offset = yAmount > 0 ? mod : 1/mod;
	cam->h *= offset;
	clampMin(&cam->h, yClamp);
	diff -= cam->h;
	posOffset = mapRange(pos.y, cam->viewPort.bottom, cam->viewPort.top, -0.5f, 0.5f);
	cam->y += diff * posOffset;

	graphCamUpdateSides(cam);
}

void graphCamCalcScale(GraphCam* cam, int xAmount, double xScale, float pos, double* newPos, double* newSize) {
	double diff, offset;
	float posOffset;
	double mod;

	diff = cam->w;
	mod = pow(xScale, abs(xAmount));
	offset = xAmount > 0 ? mod : 1/mod;
	*newSize = cam->w * offset;

	diff -= (*newSize);
	posOffset = mapRange(pos, cam->viewPort.left, cam->viewPort.right, -0.5f, 0.5f);
	*newPos = cam->x + (diff * posOffset);
}

void graphCamPosClamp(GraphCam* cam, double xMin, double xMax, double yMin, double yMax) {
	clamp(&cam->x, xMin, xMax);
	clamp(&cam->y, yMin, yMax);

	graphCamUpdateSides(cam);
}

void graphCamPosClamp(GraphCam* cam) {
	graphCamPosClamp(cam, cam->xMin + cam->w/2, cam->xMax - cam->w/2, cam->yMin + cam->h/2, cam->yMax - cam->h/2);
}

void graphCamTrans(GraphCam* cam, double xTrans, double yTrans) {
	cam->x += xTrans * (cam->w/(rectW(cam->viewPort)));
	cam->y += yTrans * (cam->h/(rectH(cam->viewPort)));

	graphCamUpdateSides(cam);
}

double graphCamScreenToCamSpaceX(GraphCam* cam, float v) {
	return v * (cam->w/(rectW(cam->viewPort)));
}

double graphCamScreenToCamSpaceY(GraphCam* cam, float v) {
	return v * (cam->h/(rectH(cam->viewPort)));
}

float graphCamCamToScreenSpaceX(GraphCam* cam, double v) {
	return v * ((rectW(cam->viewPort))/cam->w);
}

float graphCamCamToScreenSpaceY(GraphCam* cam, double v) {
	return v * ((rectH(cam->viewPort))/cam->h);
}

// Maps camera space to view/screenspace.
inline float graphCamMapX(GraphCam* cam, double v) {
	float x = mapRange(v, cam->left, cam->right, 
	                   (double)cam->viewPort.left, (double)cam->viewPort.right);
	return x;
}

inline float graphCamMapY(GraphCam* cam, double v) {
	float y = mapRange(v, cam->bottom, cam->top, 
	                   (double)cam->viewPort.bottom, (double)cam->viewPort.top);
	return y;
}

Vec2 graphCamMap(GraphCam* cam, double x, double y) {
	Vec2 vec;
	vec.x = graphCamMapX(cam, x);
	vec.y = graphCamMapY(cam, y);

	return vec;
}

// Maps view/screenspace to camera.
inline double graphCamMapXReverse(GraphCam* cam, float v) {
	double x = mapRange((double)v, (double)cam->viewPort.left, (double)cam->viewPort.right, 
	                    cam->left, cam->right);
	return x;
}

inline double graphCamMapYReverse(GraphCam* cam, float v) {
	double y = mapRange((double)v, (double)cam->viewPort.bottom, (double)cam->viewPort.top, 
	                    cam->bottom, cam->top);
	return y;
}

Rect graphCamMiniMap(GraphCam* cam, Rect viewPort) {
	Rect r;
	r.left = mapRange(cam->left,     cam->xMin, cam->xMax, 
	                                 (double)viewPort.left, (double)viewPort.right);
	r.bottom = mapRange(cam->bottom, cam->yMin, cam->yMax, 
	                                 (double)viewPort.bottom, (double)viewPort.top);
	r.right = mapRange(cam->right,   cam->xMin, cam->xMax, 
	                                 (double)viewPort.left, (double)viewPort.right);
	r.top = mapRange(cam->top,       cam->yMin, cam->yMax, 
	                                 (double)viewPort.bottom, (double)viewPort.top);
	return r;
}
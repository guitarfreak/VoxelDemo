extern ThreadQueue* globalThreadQueue;

const char* minecraftTextureFolderPath = DATA_FOLDER("Textures\\Minecraft\\");


Vec3 voxelFogColor = colorSRGB(vec3(0.43f,0.38f,0.44f));

#define SELECTION_RADIUS 5

// #define VIEW_DISTANCE 4096 // 64
// #define VIEW_DISTANCE 3072 // 32

// #define VIEW_DISTANCE 2500 // 32
// #define VIEW_DISTANCE 2048 // 32
// #define VIEW_DISTANCE 1024 // 16
// #define VIEW_DISTANCE 512  // 8
#define VIEW_DISTANCE 256 // 4
// #define VIEW_DISTANCE 128 // 2
// #define VIEW_DISTANCE 64 // 1


#define USE_MALLOC 1

#define VOXEL_X 64
#define VOXEL_Y 64
#define VOXEL_Z 254
#define VOXEL_SIZE VOXEL_X*VOXEL_Y*VOXEL_Z
#define VC_X 66
#define VC_Y 66
#define VC_Z 256
#define VOXEL_CACHE_SIZE VC_X*VC_Y*VC_Z

uchar* voxelCache[8];
uchar* voxelLightingCache[8];

#define voxelArray(x, y, z) (x)*VOXEL_Y*VOXEL_Z + (y)*VOXEL_Z + (z)
#define getVoxelCache(x, y, z) (x)*VC_Y*VC_Z + (y)*VC_Z + (z)

// int startX = 37800;
// int startY = 48000;

// float reflectionAlpha = 0.75f;
float reflectionAlpha = 0.5f;
float waterAlpha = 0.75f;
int globalLumen = 210;

int startX = 37750;
int startY = 47850;
int startXMod = 58000;
int startYMod = 68000;

int WORLD_MIN = 60;
int WORLD_MAX = 255;
// const int WATER_LEVEL_HEIGHT = WORLD_MIN*1.06f;
float waterLevelValue = 0.017f;
int WATER_LEVEL_HEIGHT = lerp(waterLevelValue, WORLD_MIN, WORLD_MAX);
// #define WATER_LEVEL_HEIGHT 62

float worldFreq = 0.004f;
int worldDepth = 6;
float modFreq = 0.02f;
int modDepth = 4;
float modOffset = 0.1f;
float heightLevels[4] = {0.4, 0.6, 0.8, 1.0f};
float worldPowCurve = 4;

#define THREADING 1

bool* treeNoise;


struct MakeMeshThreadedData;
MakeMeshThreadedData* voxelThreadData;



struct VoxelMesh {
	bool generated;
	bool upToDate;
	bool meshUploaded;

	volatile uint activeGeneration;
	volatile uint activeMaking;

	bool modifiedByUser;

	Vec2i coord;
	uchar* voxels;
	uchar* lighting;

	float transform[3][3];
	int quadCount;

	int quadCountTrans;

	char* meshBuffer;
	int meshBufferSize;
	int meshBufferCapacity;
	uint bufferId;

	char* texBuffer;
	int texBufferSize;
	int texBufferCapacity;
	uint textureId;
	uint texBufferId;

	char* meshBufferTrans;
	int meshBufferTransCapacity;
	uint bufferTransId;
	char* texBufferTrans;
	int texBufferTransCapacity;
	uint textureTransId;
	uint texBufferTransId;

	int bufferSizePerQuad;
	int textureBufferSizePerQuad;
};

struct VoxelData {
	DArray<VoxelMesh> voxels;
	DArray<int>* voxelHash;
	int voxelHashSize;
};

struct MakeMeshThreadedData {
	VoxelMesh* m;
	VoxelData* voxelData;

	int inProgress;
};

enum BlockTypes {
	BT_None = 0,
	BT_Water,
	BT_Sand,
	BT_Grass,
	BT_Stone,
	BT_Snow,
	BT_TreeLog,
	BT_Leaves,
	BT_Glass,
	BT_GlowStone,
	BT_Pumpkin,

	BT_Size,
};

enum BlockTextures {
	BX_None = 0,
	BX_Water,
	BX_Sand,
	BX_GrassTop, BX_GrassSide, BX_GrassBottom,
	BX_Stone,
	BX_Snow,
	BX_TreeLogTop, BX_TreeLogSide,
	BX_Leaves,
	BX_Glass,
	BX_GlowStone,
	BX_PumpkinTop, BX_PumpkinSide, BX_PumpkinBottom,

	BX_Size,
};

enum BlockTextures2 {
	BX2_None = 0,
	BX2_Leaves,

	BX2_Size,
};

const char* textureFilePaths[BX_Size] = {
	DATA_FOLDER("Textures\\Minecraft\\none.png"),
	DATA_FOLDER("Textures\\Minecraft\\water.png"),
	DATA_FOLDER("Textures\\Minecraft\\sand.png"),
	DATA_FOLDER("Textures\\Minecraft\\grass_top.png"),
	DATA_FOLDER("Textures\\Minecraft\\grass_side.png"),
	DATA_FOLDER("Textures\\Minecraft\\grass_bottom.png"),
	DATA_FOLDER("Textures\\Minecraft\\stone.png"),
	DATA_FOLDER("Textures\\Minecraft\\snow.png"),
	DATA_FOLDER("Textures\\Minecraft\\tree_log_top.png"),
	DATA_FOLDER("Textures\\Minecraft\\tree_log_side.png"),
	DATA_FOLDER("Textures\\Minecraft\\leaves.png"),
	DATA_FOLDER("Textures\\Minecraft\\glass.png"),
	DATA_FOLDER("Textures\\Minecraft\\glowstone.png"),
	DATA_FOLDER("Textures\\Minecraft\\pumpkin_top.png"),
	DATA_FOLDER("Textures\\Minecraft\\pumpkin_side.png"),
	DATA_FOLDER("Textures\\Minecraft\\pumpkin_bottom.png"),
};

const char* textureFilePaths2[BX2_Size] = {
	DATA_FOLDER("Textures\\Minecraft\\none.png"),
	DATA_FOLDER("Textures\\Minecraft\\leaves.png"),
};


// uchar blockColor[BT_Size] = {0,0,0,0,0,0,0,47,0,0,0};
uchar blockColor[BT_Size] = {0,17,0,0,0,0,0,16,0,0,0};
uchar texture2[BT_Size] = {0,1,1,1,1,1,1,BX_Leaves,1,1,1};
uchar textureLerp[BT_Size] = {0,0,0,0,0,0,0,0,0,0,0};


#define allTexSame(t) t,t,t,t,t,t
uchar texture1Faces[BT_Size][6] = {
	{0,0,0,0,0,0},
	{allTexSame(BX_Water)},
	{allTexSame(BX_Sand)},
	{BX_GrassSide, BX_GrassSide, BX_GrassSide, BX_GrassSide, BX_GrassTop, BX_GrassBottom},
	{allTexSame(BX_Stone)},
	{allTexSame(BX_Snow)},
	{BX_TreeLogSide, BX_TreeLogSide, BX_TreeLogSide, BX_TreeLogSide, BX_TreeLogTop, BX_TreeLogTop},
	{allTexSame(BX_Leaves)},
	{allTexSame(BX_Glass)},
	{allTexSame(BX_GlowStone)},
	{BX_PumpkinSide, BX_PumpkinSide, BX_PumpkinSide, BX_PumpkinSide, BX_PumpkinTop, BX_PumpkinBottom},
};

uchar geometry[BT_Size] = {
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_empty,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_transp,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_force,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_force,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
};

uchar meshSelection[BT_Size] = {0,1,0,0,0,0,0,1,1,0,0};

// uint blockMainTexture[BT_Size] = {
// };

static unsigned char colorPaletteCompact[64][3] =
{
   { 255,255,255 }, { 238,238,238 }, { 221,221,221 }, { 204,204,204 },
   { 187,187,187 }, { 170,170,170 }, { 153,153,153 }, { 136,136,136 },
   { 119,119,119 }, { 102,102,102 }, {  85, 85, 85 }, {  68, 68, 68 },
   {  51, 51, 51 }, {  34, 34, 34 }, {  17, 17, 17 }, {   0,  0,  0 },

   // { 220,100,30 }, { 0,100,220 }, { 255,160,160 }, { 255, 32, 32 },
   // { 220,100,30 }, { 0,100,220 }, { 255,160,160 }, { 255, 32, 32 },
   { 200,20,0 }, { 0,70,180 }, { 255,160,160 }, { 255, 32, 32 },
   { 200,120,160 }, { 200, 60,150 }, { 220,100,130 }, { 255,  0,128 },
   { 240,240,255 }, { 220,220,255 }, { 160,160,255 }, {  32, 32,255 },
   { 120,160,200 }, {  60,150,200 }, { 100,130,220 }, {   0,128,255 },
   { 240,255,240 }, { 220,255,220 }, { 160,255,160 }, {  32,255, 32 },
   { 160,200,120 }, { 150,200, 60 }, { 130,220,100 }, { 128,255,  0 },
   { 255,255,240 }, { 255,255,220 }, { 220,220,180 }, { 255,255, 32 },
   { 200,160,120 }, { 200,150, 60 }, { 220,130,100 }, { 255,128,  0 },
   { 255,240,255 }, { 255,220,255 }, { 220,180,220 }, { 255, 32,255 },
   { 160,120,200 }, { 150, 60,200 }, { 130,100,220 }, { 128,  0,255 },
   { 240,255,255 }, { 220,255,255 }, { 180,220,220 }, {  32,255,255 },
   { 120,200,160 }, {  60,200,150 }, { 100,220,130 }, {   0,255,128 },
};

static float colorPalette[64][4];

void buildColorPalette() {
   int i;
   for (i=0; i < 64; ++i) {
      colorPalette[i][0] = colorPaletteCompact[i][0] / 255.0f;
      colorPalette[i][1] = colorPaletteCompact[i][1] / 255.0f;
      colorPalette[i][2] = colorPaletteCompact[i][2] / 255.0f;
      colorPalette[i][3] = 1.0f;
   }
}


void initVoxelMesh(VoxelMesh* m, Vec2i coord) {
	TIMER_BLOCK();

	*m = {};
	m->coord = coord;

	if(USE_MALLOC) {
		m->voxels = (uchar*)malloc(VOXEL_SIZE);
		m->lighting = (uchar*)malloc(VOXEL_SIZE);
	} else {
		// m->meshBufferCapacity = kiloBytes(200);
		// m->meshBuffer = (char*)getPMemory(m->meshBufferCapacity);
		// m->texBufferCapacity = m->meshBufferCapacity/4;
		// m->texBuffer = (char*)getPMemory(m->texBufferCapacity);

		// m->meshBufferTransCapacity = kiloBytes(200);
		// m->meshBufferTrans = (char*)getPMemory(m->meshBufferTransCapacity);
		// m->texBufferTransCapacity = m->meshBufferTransCapacity/4;
		// m->texBufferTrans = (char*)getPMemory(m->texBufferTransCapacity);

		m->voxels = (uchar*)getPMemory(VOXEL_SIZE);
		m->lighting = (uchar*)getPMemory(VOXEL_SIZE);
	}

	glCreateBuffers(1, &m->bufferId);
	glCreateBuffers(1, &m->bufferTransId);

	if(STBVOX_CONFIG_MODE == 1) {
		glCreateBuffers(1, &m->texBufferId);
		glCreateTextures(GL_TEXTURE_BUFFER, 1, &m->textureId);

		glCreateBuffers(1, &m->texBufferTransId);
		glCreateTextures(GL_TEXTURE_BUFFER, 1, &m->textureTransId);
	}
}

void freeVoxelMesh(VoxelMesh* m) {
	if(!m->voxels) return;

	if(USE_MALLOC) {
		free(m->voxels);
		free(m->lighting);
	}

	glDeleteBuffers(1, &m->bufferId);
	glDeleteBuffers(1, &m->bufferTransId);

	if(STBVOX_CONFIG_MODE == 1) {
		glDeleteBuffers(1, &m->texBufferId);
		glDeleteBuffers(1, &m->texBufferTransId);
	}
}

void addVoxelMesh(VoxelData* voxelData, Vec2i coord, int index) {
	int hashIndex = mod(coord.x*9 + coord.y*23, voxelData->voxelHashSize);
	
	DArray<int>* voxelList = voxelData->voxelHash + hashIndex;
	voxelList->push(index);
}

VoxelMesh* getVoxelMesh(VoxelData* voxelData, Vec2i coord) {
	int hashIndex = mod(coord.x*9 + coord.y*23, voxelData->voxelHashSize);

	VoxelMesh* m = 0;
	DArray<int>* voxelList = voxelData->voxelHash + hashIndex;
	for(int i = 0; i < voxelList->count; i++) {
		VoxelMesh* mesh = voxelData->voxels.data + voxelList->data[i];
		if(mesh->coord == coord) {
			m = mesh;
			break;
		}
	}

	if(!m) {
		VoxelMesh mesh;
		initVoxelMesh(&mesh, coord);
		voxelData->voxels.push(mesh);
		m = voxelData->voxels.data + voxelData->voxels.count-1;
		int index = voxelData->voxels.count-1;
		voxelList->push(index);
	}

	return m;
}

void generateVoxelMeshThreaded(void* data) {
	TIMER_BLOCK();

	VoxelMesh* m = (VoxelMesh*)data;
	Vec2i coord = m->coord;

	// float worldHeightOffset = -0.1f;
	float worldHeightOffset = -0.1f;

	if(!m->generated) {
		Vec3i min = vec3i(0,0,0);
		Vec3i max = vec3i(VOXEL_X,VOXEL_Y,VOXEL_Z);

		Vec3i treePositions[100];
		int treePositionsSize = 0;

		for(int y = min.y; y < max.y; y++) {
			for(int x = min.x; x < max.x; x++) {
				int gx = (coord.x*VOXEL_X)+x;
				int gy = (coord.y*VOXEL_Y)+y;

				float height = perlin2d(gx+4000+startX, gy+4000+startY, worldFreq, worldDepth);
				height += worldHeightOffset; 

				// float mod = perlin2d(gx+startXMod, gy+startYMod, 0.008f, 4);
				float perlinMod = perlin2d(gx+startXMod, gy+startYMod, modFreq, modDepth);
				float mod = lerp(perlinMod, -modOffset, modOffset);

				float modHeight = height+mod;
				int blockType;
	    			 if(modHeight <  heightLevels[0]) blockType = BT_Sand; // sand
	    		else if(modHeight <  heightLevels[1]) blockType = BT_Grass; // grass
	    		else if(modHeight <  heightLevels[2]) blockType = BT_Stone; // stone
	    		else if(modHeight <= heightLevels[3]) blockType = BT_Snow; // snow

	    		height = clamp(height, 0, 1);
	    		// height = pow(height,3.5f);
	    		height = pow(height,worldPowCurve);
	    		int blockHeight = lerp(height, WORLD_MIN, WORLD_MAX);

	    		for(int z = 0; z < blockHeight; z++) {
	    			m->voxels[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = blockType;
	    			m->lighting[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = 0;
	    		}

	    		for(int z = blockHeight; z < VOXEL_Z; z++) {
	    			m->voxels[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = 0;
	    			m->lighting[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = globalLumen;
	    		}

	    		if(blockType == BT_Grass && treeNoise[y*VOXEL_Y + x] == 1 && 
	    			valueBetween(y, min.y+3, max.y-3) && valueBetween(x, min.x+3, max.x-3) && 
	    			valueBetween(perlinMod, 0.2f, 0.4f)) {
	    			treePositions[treePositionsSize++] = vec3i(x,y,blockHeight);
		    	}

		    	if(blockHeight < WATER_LEVEL_HEIGHT) {
		    		for(int z = blockHeight; z < WATER_LEVEL_HEIGHT; z++) {
		    			m->voxels[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = BT_Water;

		    			Vec2i waterLightRange = vec2i(0,globalLumen);
		    			int lightValue = mapRange(blockHeight, WORLD_MIN, WATER_LEVEL_HEIGHT, waterLightRange.x, waterLightRange.y);
		    			m->lighting[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = lightValue;
		    		}
		    	}
		    }
		}

		for(int i = 0; i < treePositionsSize; i++) {
			Vec3i p = treePositions[i];
			int treeHeight = randomIntPCG(3,6);
			int crownHeight = randomIntPCG(1,3);

			Vec3i tp = p + vec3i(0,0,treeHeight);
			Vec3i offset = vec3i(2,2,1);
			Vec3i min = tp - offset;
			Vec3i max = tp + offset;

			if(crownHeight == 2) max.z += 1;
			else if (crownHeight == 3) {
				max.z += 1;
				min.z -= 1;
			}

			for(int x = min.x; x <= max.x; x++) {
				for(int y = min.y; y <= max.y; y++) {
					for(int z = min.z; z <= max.z; z++) {
						m->voxels[voxelArray(x,y,z)] = BT_Leaves;    			
						m->lighting[voxelArray(x,y,z)] = 0;    			
					}
				}
			}

			m->voxels[voxelArray(min.x, min.y, max.z)] = 0;
			m->voxels[voxelArray(min.x, min.y, min.z)] = 0;
			m->voxels[voxelArray(min.x, max.y, max.z)] = 0;
			m->voxels[voxelArray(min.x, max.y, min.z)] = 0;
			m->voxels[voxelArray(max.x, min.y, max.z)] = 0;
			m->voxels[voxelArray(max.x, min.y, min.z)] = 0;
			m->voxels[voxelArray(max.x, max.y, max.z)] = 0;
			m->voxels[voxelArray(max.x, max.y, min.z)] = 0;
			m->lighting[voxelArray(min.x, min.y, max.z)] = globalLumen;
			m->lighting[voxelArray(min.x, min.y, min.z)] = globalLumen;
			m->lighting[voxelArray(min.x, max.y, max.z)] = globalLumen;
			m->lighting[voxelArray(min.x, max.y, min.z)] = globalLumen;
			m->lighting[voxelArray(max.x, min.y, max.z)] = globalLumen;
			m->lighting[voxelArray(max.x, min.y, min.z)] = globalLumen;
			m->lighting[voxelArray(max.x, max.y, max.z)] = globalLumen;
			m->lighting[voxelArray(max.x, max.y, min.z)] = globalLumen;

			for(int i = 0; i < treeHeight; i++) {
				m->voxels[voxelArray(p.x,p.y,p.z+i)] = BT_TreeLog;
				m->lighting[voxelArray(p.x,p.y,p.z+i)] = 0;
			}
		}
	}

	if(THREADING) atomicSub(&m->activeGeneration);
	m->generated = true;
}

void makeMeshThreaded(void* data) {	
	TIMER_BLOCK();
	
	MakeMeshThreadedData* d = (MakeMeshThreadedData*)data;
	VoxelMesh* m = d->m;

	int cacheId = THREADING ? getThreadQueueId(globalThreadQueue) : 1;

	// gather voxel data in radius and copy to cache
	Vec2i coord = m->coord;
	for(int y = -1; y < 2; y++) {
		for(int x = -1; x < 2; x++) {
			Vec2i c = coord + vec2i(x,y);
			VoxelMesh* lm = getVoxelMesh(d->voxelData, c);

			assert(lm->generated);

			int w = x == 0 ? VOXEL_X : 1;
			int h = y == 0 ? VOXEL_Y : 1;

			Vec2i mPos;
			mPos.x = x == -1 ? VOXEL_X-1 : 0;
			mPos.y = y == -1 ? VOXEL_Y-1 : 0;

			Vec2i lPos;
			if(x == -1) lPos.x = 0;
			else if(x ==  0) lPos.x = 1;
			else if(x ==  1) lPos.x = VOXEL_X+1;
			if(y == -1) lPos.y = 0;
			else if(y ==  0) lPos.y = 1;
			else if(y ==  1) lPos.y = VOXEL_Y+1;

			for(int z = 0; z < VOXEL_Z; z++) {
				for(int y = 0; y < h; y++) {
					for(int x = 0; x < w; x++) {
						voxelCache[cacheId][getVoxelCache(x+lPos.x, y+lPos.y, z+1)] = lm->voxels[(x+mPos.x)*VOXEL_Y*VOXEL_Z + (y+mPos.y)*VOXEL_Z + z];
						voxelLightingCache[cacheId][getVoxelCache(x+lPos.x, y+lPos.y, z+1)] = lm->lighting[(x+mPos.x)*VOXEL_Y*VOXEL_Z + (y+mPos.y)*VOXEL_Z + z];
					}
				}
			}

			// make floor solid
			for(int y = 0; y < VC_Y; y++) {
				for(int x = 0; x < VC_X; x++) {
					voxelCache[cacheId][getVoxelCache(x, y, 0)] = BT_Sand; // change
				}
			}
		}
	}

	stbvox_mesh_maker mm;
	stbvox_init_mesh_maker(&mm);
	stbvox_input_description* inputDesc = stbvox_get_input_description(&mm);
	*inputDesc = {};

	if(USE_MALLOC) {
		m->meshBufferCapacity = kiloBytes(500);
		m->meshBuffer = (char*)malloc(m->meshBufferCapacity);
		m->texBufferCapacity = m->meshBufferCapacity/4;
		m->texBuffer = (char*)malloc(m->texBufferCapacity);

		m->meshBufferTransCapacity = kiloBytes(500);
		m->meshBufferTrans = (char*)malloc(m->meshBufferTransCapacity);
		m->texBufferTransCapacity = m->meshBufferTransCapacity/4;
		m->texBufferTrans = (char*)malloc(m->texBufferTransCapacity);
	} else {
		// m->meshBufferCapacity = kiloBytes(500);
		// m->meshBuffer = (char*)getDMemory(m->meshBufferCapacity);
		// m->texBufferCapacity = m->meshBufferCapacity/4;
		// m->texBuffer = (char*)getDMemory(m->texBufferCapacity);

		// m->meshBufferTransCapacity = kiloBytes(500);
		// m->meshBufferTrans = (char*)getDMemory(m->meshBufferTransCapacity);
		// m->texBufferTransCapacity = m->meshBufferTransCapacity/4;
		// m->texBufferTrans = (char*)getDMemory(m->texBufferTransCapacity);

		m->meshBufferCapacity = kiloBytes(500);
		m->meshBuffer = (char*)malloc(m->meshBufferCapacity);
		m->texBufferCapacity = m->meshBufferCapacity/4;
		m->texBuffer = (char*)malloc(m->texBufferCapacity);

		m->meshBufferTransCapacity = kiloBytes(500);
		m->meshBufferTrans = (char*)malloc(m->meshBufferTransCapacity);
		m->texBufferTransCapacity = m->meshBufferTransCapacity/4;
		m->texBufferTrans = (char*)malloc(m->texBufferTransCapacity);
	}

	stbvox_set_buffer(&mm, 0, 0, m->meshBuffer, m->meshBufferCapacity);
	if(STBVOX_CONFIG_MODE == 1) {
		stbvox_set_buffer(&mm, 0, 1, m->texBuffer, m->texBufferCapacity);
	}

	stbvox_set_buffer(&mm, 1, 0, m->meshBufferTrans, m->meshBufferTransCapacity);
	if(STBVOX_CONFIG_MODE == 1) {
		stbvox_set_buffer(&mm, 1, 1, m->texBufferTrans, m->texBufferTransCapacity);
	}

	// int count = stbvox_get_buffer_count(&mm);




	inputDesc->block_tex2 = texture2;
	inputDesc->block_tex1_face = texture1Faces;
	inputDesc->block_geometry = geometry;
	inputDesc->block_selector = meshSelection;
	inputDesc->block_texlerp = textureLerp;

	uchar color[BT_Size];
	for(int i = 0; i < BT_Size; i++) color[i] = STBVOX_MAKE_COLOR(blockColor[i], 1, 0);
		inputDesc->block_color = color;




	stbvox_set_input_stride(&mm, VC_Y*VC_Z,VC_Z);
	stbvox_set_input_range(&mm, 0,0,0, VOXEL_X, VOXEL_Y, VOXEL_Z);

	inputDesc->blocktype = &voxelCache[cacheId][getVoxelCache(1,1,1)];
	inputDesc->lighting = &voxelLightingCache[cacheId][getVoxelCache(1,1,1)];

	// stbvox_set_default_mesh(&mm, 0);
	int success = stbvox_make_mesh(&mm);

	stbvox_set_mesh_coordinates(&mm, coord.x*VOXEL_X, coord.y*VOXEL_Y,0);

	stbvox_get_transform(&mm, m->transform);
	float bounds [2][3]; stbvox_get_bounds(&mm, bounds);

	m->quadCount = stbvox_get_quad_count(&mm, 0);
	m->quadCountTrans = stbvox_get_quad_count(&mm, 1);

	m->bufferSizePerQuad = stbvox_get_buffer_size_per_quad(&mm, 0);
	m->textureBufferSizePerQuad = stbvox_get_buffer_size_per_quad(&mm, 1);



	if(THREADING) atomicSub(&m->activeMaking);
	m->upToDate = true;
	d->inProgress = false;
}

void makeMesh(VoxelMesh* m, VoxelData* voxelData) {
	TIMER_BLOCK();

	// int threadJobsMax = 20;

	bool notAllMeshsAreReady = false;
	Vec2i coord = m->coord;
	for(int y = -1; y < 2; y++) {
		for(int x = -1; x < 2; x++) {
			Vec2i c = coord + vec2i(x,y);
			VoxelMesh* lm = getVoxelMesh(voxelData, c);

			if(!lm->generated) {
				if(THREADING) {
					// if(!lm->activeGeneration) {
					if(!lm->activeGeneration && !threadQueueFull(globalThreadQueue)) {
						// if(!lm->activeGeneration && threadQueueOpenJobs(globalThreadQueue) < threadJobsMax) {
						atomicAdd(&lm->activeGeneration);
						threadQueueAdd(globalThreadQueue, generateVoxelMeshThreaded, lm);
					}
					notAllMeshsAreReady = true;
				} else {
					// generateVoxelMeshThreaded(lm);

					threadQueueAdd(globalThreadQueue, generateVoxelMeshThreaded, lm);
				}
			} 
		}
	}

	if(!THREADING) threadQueueComplete(globalThreadQueue);

	if(notAllMeshsAreReady) return;

	if(!m->upToDate) {
		if(!m->activeMaking) {
			if(THREADING) {
				if(threadQueueFull(globalThreadQueue)) return;
				// if(threadQueueOpenJobs(globalThreadQueue) < threadJobsMax) return;

				MakeMeshThreadedData* data;
				for(int i = 0; i < 256; i++) {
					if(!voxelThreadData[i].inProgress) {
						voxelThreadData[i] = {m, voxelData, true};
						// data = voxelThreadData + i;
						
						atomicAdd(&m->activeMaking);
						threadQueueAdd(globalThreadQueue, makeMeshThreaded, &voxelThreadData[i]);

						break;
					}
				}
			} else {
				voxelThreadData[1] = {m, voxelData, true};
				makeMeshThreaded(&voxelThreadData[1]);
			}
		}

		if(THREADING) return;
	} 

	glNamedBufferData(m->bufferId, m->bufferSizePerQuad*m->quadCount, m->meshBuffer, GL_STATIC_DRAW);

	glNamedBufferData(m->texBufferId, m->textureBufferSizePerQuad*m->quadCount, m->texBuffer, GL_STATIC_DRAW);
	glTextureBuffer(m->textureId, GL_RGBA8UI, m->texBufferId);

	glNamedBufferData(m->bufferTransId, m->bufferSizePerQuad*m->quadCountTrans, m->meshBufferTrans, GL_STATIC_DRAW);

	glNamedBufferData(m->texBufferTransId, m->textureBufferSizePerQuad*m->quadCountTrans, m->texBufferTrans, GL_STATIC_DRAW);
	glTextureBuffer(m->textureTransId, GL_RGBA8UI, m->texBufferTransId);

	if(USE_MALLOC) {
		free(m->meshBuffer);
		free(m->texBuffer);
		free(m->meshBufferTrans);
		free(m->texBufferTrans);
	} else {
		// freeDMemory(m->meshBuffer);
		// freeDMemory(m->texBuffer);
		// freeDMemory(m->meshBufferTrans);
		// freeDMemory(m->texBufferTrans);

		free(m->meshBuffer);
		free(m->texBuffer);
		free(m->meshBufferTrans);
		free(m->texBufferTrans);
	}

	m->meshUploaded = true;
}

// coord 		Vec3
// voxel 		Vec3i -> based on voxel size
// mesh			Vec2i -> based on mesh size
// localVoxel 	Vec3i -> mod(voxel)

// voxelCoord 	Vec3 -> voxel + voxelSize/2
// meshCoord 	Vec3 -> mesh + meshSize/2

// meshPointer 	-> mesh
// voxelPointer -> meshPointer,

// coord -> voxel
Vec3i coordToVoxel(Vec3 coord) {
	if(coord.x < 0) coord.x -= 1;
	if(coord.y < 0) coord.y -= 1;
	Vec3i result = vec3i(coord);

	return result;
}

// voxel -> mesh
Vec2i voxelToMesh(Vec3i voxel) {
	Vec2i result = vec2i(floor(voxel.x/(float)VOXEL_X), floor(voxel.y/(float)VOXEL_Y));

	return result;
}

// coord -> mesh
Vec2i coordToMesh(Vec3 coord) {
	Vec3i mc = coordToVoxel(coord);
	Vec2i result = voxelToMesh(mc);

	return result;
}

// voxel -> localVoxel
Vec3i voxelToLocalVoxel(Vec3i voxel) {
	Vec3i result = voxel;
	result.x = mod(voxel.x, VOXEL_X);
	result.y = mod(voxel.y, VOXEL_Y);

	return result;
}


// voxel -> voxelCoord
Vec3 voxelToVoxelCoord(Vec3i voxel) {
	Vec3 result;
	result = vec3(voxel) + vec3(0.5f, 0.5f, 0.5f);
	return result;
}

// coord -> voxelCoord
Vec3 coordToVoxelCoord(Vec3 coord) {
	Vec3 result = voxelToVoxelCoord(coordToVoxel(coord));
	return result;
}

// mesh -> meshCoord
Vec3 meshToMeshCoord(Vec2i mesh) {
	Vec3 result = vec3(mesh.x*VOXEL_X + VOXEL_X*0.5f, mesh.y*VOXEL_Y + VOXEL_Y*0.5f, VOXEL_Z*0.5f);
	return result;
}

// coord -> meshCoord
Vec3 coordToMeshCoord(Vec3 coord) {
	Vec3 result = meshToMeshCoord(coordToMesh(coord));
	return result;
}


// voxel -> block
uchar* getBlockFromVoxel(VoxelData* voxelData, Vec3i voxel) {
	VoxelMesh* vm = getVoxelMesh(voxelData, voxelToMesh(voxel));
	Vec3i localCoord = voxelToLocalVoxel(voxel);
	uchar* block = &vm->voxels[voxelArray(localCoord.x, localCoord.y, localCoord.z)];

	return block;
}

// coord -> block
uchar* getBlockFromCoord(VoxelData* voxelData, Vec3 coord) {
	return getBlockFromVoxel(voxelData, coordToVoxel(coord));
}

// voxel -> lighting
uchar* getLightingFromVoxel(VoxelData* voxelData, Vec3i voxel) {
	VoxelMesh* vm = getVoxelMesh(voxelData, voxelToMesh(voxel));
	Vec3i localCoord = voxelToLocalVoxel(voxel);
	uchar* block = &vm->lighting[voxelArray(localCoord.x, localCoord.y, localCoord.z)];

	return block;
}

// coord -> lighting
uchar* getLightingFromCoord(VoxelData* voxelData, Vec3 coord) {
	return getLightingFromVoxel(voxelData, coordToVoxel(coord));
}

void setupVoxelUniforms(Vec4 camera, uint texUnit1, uint texUnit2, uint faceUnit, Mat4 view, Mat4 proj, Vec3 fogColor, Vec3 trans = vec3(0,0,0), Vec3 scale = vec3(1,1,1), Vec3 rotation = vec3(0,0,0)) {
	TIMER_BLOCK();

	buildColorPalette();

	Vec3 li = normVec3(vec3(0,0.5f,0.5f));
	Mat4 ambientLighting = {
		li.x, li.y, li.z ,0, // reversed lighting direction
		0.5,0.5,0.5,0, // directional color
		0.5,0.5,0.5,0, // constant color
		0.5,0.5,0.5,1.0f/1000.0f/1000.0f, // fog data for simple_fog
	};

	Mat4 al;

	float bright = 1.0f;
	float amb[3][3];

	#ifdef STBVOX_CONFIG_LIGHTING_SIMPLE
	bright = 0.35f;  // when demoing lighting

	static float dtl = 0;
	dtl += 0.008f;
	float start = 40;
	float amp = 30;

	Vec3 lColor = vec3(0.7f,0.7f,0.5f);
	Vec3 lColorBrightness = lColor*50;
	Vec3 light[2] = { vec3(0,0,(amp/2)+start + sin(dtl)*amp), lColorBrightness };
	// int loc = glGetUniformLocation(globalGraphicsState->pipelineIds.voxelFragment, "light_source");
	// glProgramUniform3fv(globalGraphicsState->pipelineIds.voxelFragment, loc, 2, (GLfloat*)light);
	pushUniform(SHADER_VOXEL, 1, VOXEL_UNIFORM_LIGHT_SOURCE, light, 2);
	dcCube({light[0], vec3(3,3,3), vec4(lColor, 1), 0, vec3(0,0,0)});
	#endif

	// ambient direction is sky-colored upwards
	// "ambient" lighting is from above
	Vec3 li2 = normVec3(vec3(-1,1,1));
	al.e2[0][0] = li2.x;
	al.e2[0][1] = li2.y;
	al.e2[0][2] = li2.z;
	al.e2[0][3] = 0;

	// al.e2[0][0] =  0.3f;
	// al.e2[0][1] = -0.5f;
	// al.e2[0][2] =  0.9f;

	amb[1][0] = 0.3f; amb[1][1] = 0.3f; amb[1][2] = 0.3f; // dark-grey
	amb[2][0] = 1.0; amb[2][1] = 1.0; amb[2][2] = 1.0; // white

	// convert so (table[1]*dot+table[2]) gives
	// above interpolation
	//     lerp((dot+1)/2, amb[1], amb[2])
	//     amb[1] + (amb[2] - amb[1]) * (dot+1)/2
	//     amb[1] + (amb[2] - amb[1]) * dot/2 + (amb[2]-amb[1])/2

	for (int j=0; j < 3; ++j) {
		al.e2[1][j] = (amb[2][j] - amb[1][j])/2 * bright;
		al.e2[2][j] = (amb[1][j] + amb[2][j])/2 * bright;
	}
	al.e2[1][3] = 0;
	al.e2[2][3] = 0;

	// fog color
	al.e2[3][0] = fogColor.x, al.e2[3][1] = fogColor.y, al.e2[3][2] = fogColor.z;
	// al.e2[3][3] = 1.0f / (view_distance - MESH_CHUNK_SIZE_X);
	// al.e2[3][3] *= al.e2[3][3];
	al.e2[3][3] = (float)1.0f/(VIEW_DISTANCE - VOXEL_X);
	al.e2[3][3] *= al.e2[3][3];

	ambientLighting = al;

	int texUnit[2] = {texUnit1, texUnit2};

	for(int i = 0; i < STBVOX_UNIFORM_count; ++i) {
		stbvox_uniform_info sui;
		int success = stbvox_get_uniform_info(&sui, i);
		if(success == false) continue;
		if(i == VOXEL_UNIFORM_TRANSFORM) continue;

		int count = sui.array_length;
		void* data = sui.default_value;

		if(i == VOXEL_UNIFORM_FACE_DATA) data = &faceUnit;
		else if(i == VOXEL_UNIFORM_TEX_ARRAY) data = texUnit;
		else if(i == VOXEL_UNIFORM_COLOR_TABLE) data = colorPalette;
		else if(i == VOXEL_UNIFORM_AMBIENT) data = ambientLighting.e;
		else if(i == VOXEL_UNIFORM_CAMERA_POS) data = camera.e;

		pushUniform(SHADER_VOXEL, 2, i, data, count);
	}	


	Mat4 model = modelMatrix(trans, scale, 0, rotation);
	Mat4 finalMat = proj*view*model;

	pushUniform(SHADER_VOXEL, 1, VOXEL_UNIFORM_ALPHATEST, 0.0f);
	pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CLIPPLANE, false);
	pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CPLANE1, 0,0,0,0);
	pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CPLANE2, 0,0,0,0);

	pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_MODEL, model.e);
	pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_MODEL_VIEW, finalMat.e);
}

void drawVoxelMesh(VoxelMesh* m, int drawMode = 0) {
	TIMER_BLOCK();

	globalGraphicsState->textureUnits[0] = globalGraphicsState->textures3d[0].id;
	globalGraphicsState->textureUnits[1] = globalGraphicsState->textures3d[1].id;
	globalGraphicsState->samplerUnits[0] = globalGraphicsState->samplers[SAMPLER_VOXEL_1];
	globalGraphicsState->samplerUnits[1] = globalGraphicsState->samplers[SAMPLER_VOXEL_2];
	globalGraphicsState->samplerUnits[2] = globalGraphicsState->samplers[SAMPLER_VOXEL_3];



	glBindSamplers(0,16,globalGraphicsState->samplerUnits);

	bindShader(SHADER_VOXEL);
	pushUniform(SHADER_VOXEL, 2, VOXEL_UNIFORM_TRANSFORM, m->transform[0], 3);

	if(drawMode == 0 || drawMode == 2) {
		glBindBuffer(GL_ARRAY_BUFFER, m->bufferId);
		int vaLoc = glGetAttribLocation(getShader(SHADER_VOXEL)->vertex, "attr_vertex");
		glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 0, (void*)0);
		glEnableVertexAttribArray(vaLoc);

		globalGraphicsState->textureUnits[2] = m->textureId;
		glBindTextures(0,16,globalGraphicsState->textureUnits);

		glDrawArrays(GL_QUADS, 0, m->quadCount*4);
	}

	if(drawMode == 0 || drawMode == 1) {
		glBindBuffer(GL_ARRAY_BUFFER, m->bufferTransId);
		int vaLoc = glGetAttribLocation(getShader(SHADER_VOXEL)->vertex, "attr_vertex");
		glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 0, (void*)0);
		glEnableVertexAttribArray(vaLoc);

		globalGraphicsState->textureUnits[2] = m->textureTransId;
		glBindTextures(0,16,globalGraphicsState->textureUnits);

		glDrawArrays(GL_QUADS, 0, m->quadCountTrans*4);
	}
}


void loadVoxelTextures(char* folderPath, int internalFormat, bool reload = false, int levelToReload = 0) {

	const int mipMapCount = 5;
	char* p = getTString(34);
	strClear(p);
	strAppend(p, (char*)folderPath);

	Texture* texture = globalGraphicsState->textures3d + 0;

	if(!reload) {
		glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, &texture->id);
		glTextureStorage3D(texture->id, mipMapCount, internalFormat, 32, 32, BX_Size);
	}

	for(int layerIndex = 0; layerIndex < BX_Size; layerIndex++) {
		if(reload && layerIndex != levelToReload) continue;

		int x,y,n;
		unsigned char* stbData = stbi_load(textureFilePaths[layerIndex], &x, &y, &n, 4);

		if(layerIndex == BX_Water) {
			uint* data = (uint*)stbData;
			for(int x = 0; x < 32; x++) {
				for(int y = 0; y < 32; y++) {
					Vec4 c;
					colorGetRGBA(data[y*32 + x], c.e);
					c.r = waterAlpha;
					data[y*32 + x] = mapRGBA(c.e);
				}
			}
		}

		glTextureSubImage3D(texture->id, 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);

		stbi_image_free(stbData);
	}

	glGenerateTextureMipmap(texture->id);


	// Adjust mipmap alpha levels for tree textures.

	if(!reload || (reload && levelToReload == BX_Leaves))
	{
		int textureId = globalGraphicsState->textures3d[0].id;

		float alphaCoverage[mipMapCount] = {};
		int size = 32;
		Vec4* pixels = (Vec4*)getTMemory(sizeof(Vec4)*size*size);
		for(int i = 0; i < mipMapCount; i++) {
			glGetTextureSubImage(textureId, i, 0,0,BX_Leaves, size, size, 1, GL_RGBA, GL_FLOAT, size*size*sizeof(Vec4), &pixels[0]);

			for(int y = 0; y < size; y++) {
				for(int x = 1; x < size; x++) {
					alphaCoverage[i] += pixels[y*size + x].a;
				}
			}
		
			alphaCoverage[i] = alphaCoverage[i] / (size*size);
			size /= 2;
		}

		float alphaCoverage2[mipMapCount] = {};
		size = 16;
		for(int i = 1; i < mipMapCount; i++) {
			glGetTextureSubImage(textureId, i, 0,0,BX_Leaves, size, size, 1, GL_RGBA, GL_FLOAT, size*size*sizeof(Vec4), &pixels[0]);

			// float alphaScale = (size*size*alphaCoverage[0]) / (alphaCoverage[i]*size*size);
			float alphaScale = (alphaCoverage[0]) / (alphaCoverage[i]);

			for(int y = 0; y < size; y++) {
				for(int x = 1; x < size; x++) {
					pixels[y*size + x].a *= alphaScale;
					alphaCoverage2[i] += pixels[y*size + x].a;
				}
			}
		
			glTextureSubImage3D(textureId, i, 0, 0, BX_Leaves, size, size, 1, GL_RGBA, GL_FLOAT, pixels);

			alphaCoverage2[i] = alphaCoverage2[i] / (size*size);
			size /= 2;
		}
	}


	texture = globalGraphicsState->textures3d + 1;

	if(!reload) {
		glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, &texture->id);
		glTextureStorage3D(texture->id, 1, internalFormat, 32, 32, BX2_Size);
	}

	for(int layerIndex = 0; layerIndex < BX2_Size; layerIndex++) {
		if(reload && layerIndex != levelToReload) continue;

		int x,y,n;
		unsigned char* stbData = stbi_load(textureFilePaths2[layerIndex], &x, &y, &n, 4);
		
		glTextureSubImage3D(texture->id, 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);
		stbi_image_free(stbData);
	}

	glGenerateTextureMipmap(texture->id);

}


bool collisionVoxelWidthBox(VoxelData* voxelData, Vec3 boxPos, Vec3 boxSize, float* minDistance = 0, Vec3* collisionVoxel = 0) {

	// First get the mesh coords that touch the player box.

	Rect3 box = rect3CenDim(boxPos, boxSize);
	Vec3i voxelMin = coordToVoxel(box.min);
	Vec3i voxelMax = coordToVoxel(box.max+1);

	bool checkDistance = minDistance != 0 && collisionVoxel != 0;

	bool collision = false;
	if(checkDistance) *minDistance = 100000; // @Cleanup: Replace with FLT_MAX or something.

	// Check collision with the voxel that's closest.

	for(int x = voxelMin.x; x < voxelMax.x; x++) {
		for(int y = voxelMin.y; y < voxelMax.y; y++) {
			for(int z = voxelMin.z; z < voxelMax.z; z++) {
				Vec3i coord = vec3i(x,y,z);
				uchar* block = getBlockFromVoxel(voxelData, coord);

				if(*block > 0) {
					if(checkDistance) {
						Vec3 cBox = voxelToVoxelCoord(coord);
						float distance = lenVec3(boxPos - cBox);
						if(*minDistance == 100000 || distance > *minDistance) {
							*minDistance = distance;
							*collisionVoxel = cBox;
						}
					}

					collision = true;
				}
			}
		}
	}

	return collision;
}
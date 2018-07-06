
#define VOXEL_X 64
#define VOXEL_Y 64
#define VOXEL_Z 254

#define VC_X (VOXEL_X + 2)
#define VC_Y (VOXEL_Y + 2)
#define VC_Z (VOXEL_Z + 2)
#define VOXEL_CACHE_SIZE (VC_X*VC_Y*VC_Z)

#define voxelCacheArray(x, y, z) (x)*VC_Y*VC_Z + (y)*VC_Z + (z)

// #define TEXTURE_CACHE_DISTANCE (VIEW_DISTANCE + 2)
// #define DATA_CACHE_DISTANCE (VIEW_DISTANCE / 2)
// #define STORE_DISTANCE (DATA_CACHE_DISTANCE + 2)

#define chunkOffsetToVoxelOffset(c) vec3i(c * vec2i(VOXEL_X, VOXEL_Y), 0)

struct VoxelWorldSettings {
	int viewDistance = 5;
	int storeDistanceOffset = 3;
	int storeSize = 2;

	//

	Vec4 fogColor = linearToGamma(vec4(hslToRgbf(0.55f,0.5f,0.8f),1));

	float reflectionAlpha = 0.5f;
	float waterAlpha = 0.75f;
	int globalLumen = 210;

	int startX = 37750;
	int startY = 47850;
	int startXMod = 58000;
	int startYMod = 68000;

	int worldMin = 60;
	int worldMax = 255;

	float waterLevelValue = 0.017f;
	int waterLevelHeight = lerp(waterLevelValue, worldMin, worldMax);

	float worldFreq = 0.004f;
	int worldDepth = 6;
	float modFreq = 0.02f;
	int modDepth = 4;
	float modOffset = 0.1f;
	float heightLevels[4];
	float worldPowCurve = 4;

	bool* treeNoise;
};

struct VoxelMesh {
	Vec2i coord;
	uchar* data;
	uchar* voxels;
	uchar* lighting;

	uchar* compressedData;
	int compressedVoxelsSize;
	int compressedLightingSize;

	//

	bool compressionStep;
	bool stored;

	bool generated;
	bool upToDate;
	bool uploaded;

	bool modified;

	volatile uint activeGeneration;
	volatile uint activeMaking;
	volatile uint activeStoring;

	//

	char* meshBuffer;
	uint bufferId;

	char* texBuffer;
	uint textureId;
	uint texBufferId;

	char* meshBufferTrans;
	uint bufferTransId;

	char* texBufferTrans;
	uint textureTransId;
	uint texBufferTransId;

	//

	float transform[3][3];
	int quadCount;
	int quadCountTrans;
	int bufferSizePerQuad;
	int textureBufferSizePerQuad;
};

struct VoxelArray {
	int data[10];
	int count;
};

struct VoxelData {
	int size;
	DArray<VoxelMesh> voxels;
	VoxelArray* voxelHash;
	int voxelHashSize;
};

struct MakeMeshThreadedData {
	VoxelMesh* m;
	VoxelData* voxelData;
};

struct GenerateMeshThreadedData {
	VoxelMesh* m;
	VoxelWorldSettings* vs;
};

void voxelWorldSettingsInit(VoxelWorldSettings* vs) {
	float heightLevels[] = {0.4, 0.6, 0.8, 1.0f};
	for(int i = 0; i < arrayCount(vs->heightLevels); i++) 
		vs->heightLevels[i] = heightLevels[i];

}

enum BlockTypes {
	BT_None = 0,
	BT_Water,
	BT_Sand,
	BT_Ground,
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
	BX2_Grass,
	BX2_Leaves,

	BX2_Size,
};

int blockTypesInventory[] = {
	BX_None, BX_Water, BX_Sand, BX_GrassBottom, BX_GrassTop, BX_Stone, BX_Snow, BX_TreeLogTop, BX_Leaves, BX_Glass, BX_GlowStone, BX_PumpkinTop,
};

const char* voxelTextureFiles[BX_Size] = {
	"minecraft\\none.png",
	"minecraft\\water.png",
	"minecraft\\sand.png",
	"minecraft\\grass_top.png",
	"minecraft\\grass_side.png",
	"minecraft\\grass_bottom.png",
	"minecraft\\stone.png",
	"minecraft\\snow.png",
	"minecraft\\tree_log_top.png",
	"minecraft\\tree_log_side.png",
	"minecraft\\leaves.png",
	"minecraft\\glass.png",
	"minecraft\\glowstone.png",
	"minecraft\\pumpkin_top.png",
	"minecraft\\pumpkin_side.png",
	"minecraft\\pumpkin_bottom.png",
};

const char* voxelTextureFiles2[BX2_Size] = {
	"minecraft\\none.png",
	"minecraft\\grass_bottom.png",
	"minecraft\\leaves.png",
};

char* footstepsDirt[] = {
	"footsteps\\dirt1.ogg",
	"footsteps\\dirt2.ogg",
	"footsteps\\dirt3.ogg",
	"footsteps\\dirt4.ogg",
};
char* footstepsWater[] = {
	"footsteps\\water1.ogg",
	"footsteps\\water2.ogg",
	"footsteps\\water3.ogg",
};
char* footstepsGrass[] = {
	"footsteps\\grass1.ogg",
	"footsteps\\grass2.ogg",
	"footsteps\\grass3.ogg",
	"footsteps\\grass4.ogg",
	"footsteps\\grass5.ogg",
	"footsteps\\grass6.ogg",
};
char* footstepsSand[] = {
	"footsteps\\sand1.ogg",
	"footsteps\\sand2.ogg",
	"footsteps\\sand3.ogg",
};
char* footstepsSnow[] = {
	"footsteps\\snow1.ogg",
	"footsteps\\snow2.ogg",
	"footsteps\\snow3.ogg",
	"footsteps\\snow4.ogg",
};

struct FootstepArray {
	char** files;
	int count;
};

enum FootstepType {
	FOOTSTEP_DIRT = 0,
	FOOTSTEP_WATER,
	FOOTSTEP_GRASS,
	FOOTSTEP_SAND,
	FOOTSTEP_SNOW,

	FOOTSTEP_SIZE,
};

FootstepArray footstepFiles[] = {
	{ footstepsDirt,  arrayCount(footstepsDirt) },
	{ footstepsWater, arrayCount(footstepsWater) },
	{ footstepsGrass, arrayCount(footstepsGrass) },
	{ footstepsSand,  arrayCount(footstepsSand) },
	{ footstepsSnow,  arrayCount(footstepsSnow) },
};

int blockTypeFootsteps[BT_Size] = {
	FOOTSTEP_DIRT,
	FOOTSTEP_WATER,
	FOOTSTEP_SAND,
	FOOTSTEP_GRASS,
	FOOTSTEP_GRASS,
	FOOTSTEP_DIRT,
	FOOTSTEP_SNOW,
	FOOTSTEP_DIRT,
	FOOTSTEP_GRASS,
	FOOTSTEP_DIRT,
	FOOTSTEP_DIRT,
	FOOTSTEP_DIRT,
};



// uchar blockColor[BT_Size] = {0,0,0,0,0,0,0,47,0,0,0};
uchar blockColor[BT_Size] = {0,17,0,0,0,0,0,0,16,0,0,0};
uchar texture2[BT_Size] = {0,1,1,BX2_Grass,1,1,1,1,BX_Leaves,1,1,1};
uchar textureLerp[BT_Size] = {0,0,0,0,0,0,0,0,0,0,0,0};


#define allTexSame(t) t,t,t,t,t,t
uchar texture1Faces[BT_Size][6] = {
	{0,0,0,0,0,0},
	{allTexSame(BX_Water)},
	{allTexSame(BX_Sand)},
	{allTexSame(BX_GrassBottom)},
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
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_force,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_force,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
};

uchar meshSelection[BT_Size] = {0,1,0,0,0,0,0,0,1,1,0,0};

Vec4 blockTypeParticleColor[] = {
	vec4(0,0,0,1),
	vec4(0.561, 0.682, 0.812,0.75f), // BT_Water,
	vec4(0.859, 0.722, 0.49,1), // BT_Sand,
	vec4(0.643, 0.518, 0.337,1), // BT_Ground,
	vec4(0.431, 0.475, 0.196,1), // BT_Grass,
	vec4(0.451, 0.451, 0.451,1), // BT_Stone,
	vec4(0.945, 0.961, 0.969,1), // BT_Snow,
	vec4(0.698, 0.62, 0.478,1), // BT_TreeLog,
	vec4(200/255.0f,20/255.0f,0,1), // BT_Leaves,
	vec4(1,1,1,0.2f), // BT_Glass,
	vec4(0.918f,0.882f,0.51f,1), // BT_GlowStone,
	vec4(0.98f,0.6f,0.165f,1), // BT_Pumpkin,
};

float blockHardnessType[] = {
	0, // Air. 
	1, // Leaves.
	2, // Sand.
	3, // Ground.
	4, // Wood.
	5, // Stone.
};

float blockTypeHardness[] = {
	blockHardnessType[0], // BT_None,
	blockHardnessType[1], // BT_Water,
	blockHardnessType[2], // BT_Sand,
	blockHardnessType[3], // BT_Ground,
	blockHardnessType[3], // BT_Grass,
	blockHardnessType[5], // BT_Stone,
	blockHardnessType[3], // BT_Snow,
	blockHardnessType[4], // BT_TreeLog,
	blockHardnessType[1], // BT_Leaves,
	blockHardnessType[1], // BT_Glass,
	blockHardnessType[4], // BT_GlowStone,
	blockHardnessType[4], // BT_Pumpkin,
};

Vec3 voxelVertices[] = {
	vec3( 1, 1, 1), vec3( 1, 1,-1), vec3( 1,-1,-1), vec3( 1,-1, 1), // +x
	vec3(-1,-1, 1), vec3(-1,-1,-1), vec3(-1, 1,-1), vec3(-1, 1, 1), // -x
	vec3(-1, 1, 1), vec3(-1, 1,-1), vec3( 1, 1,-1), vec3( 1, 1, 1), // +y
	vec3( 1,-1, 1), vec3( 1,-1,-1), vec3(-1,-1,-1), vec3(-1,-1, 1), // -y
	vec3(-1, 1, 1), vec3( 1, 1, 1), vec3( 1,-1, 1), vec3(-1,-1, 1), // +z
	vec3( 1, 1,-1), vec3(-1, 1,-1), vec3(-1,-1,-1), vec3( 1,-1,-1), // -z
};

static unsigned char colorPaletteCompact[64][3] =
{
   { 255,255,255 }, { 238,238,238 }, { 221,221,221 }, { 204,204,204 },
   { 187,187,187 }, { 170,170,170 }, { 153,153,153 }, { 136,136,136 },
   { 119,119,119 }, { 102,102,102 }, {  85, 85, 85 }, {  68, 68, 68 },
   {  51, 51, 51 }, {  34, 34, 34 }, {  17, 17, 17 }, {   0,  0,  0 },
   { 200, 20,  0 }, {   0, 70,180 }, { 255,160,160 }, { 255, 32, 32 },
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

void allocVoxelGPUData(VoxelMesh* m) {
	glCreateBuffers(1, &m->bufferId);
	glCreateBuffers(1, &m->bufferTransId);

	if(STBVOX_CONFIG_MODE == 1) {
		glCreateBuffers(1, &m->texBufferId);
		glCreateTextures(GL_TEXTURE_BUFFER, 1, &m->textureId);

		glCreateBuffers(1, &m->texBufferTransId);
		glCreateTextures(GL_TEXTURE_BUFFER, 1, &m->textureTransId);
	}
}

void allocVoxelMesh(VoxelMesh* m) {
	m->data = mallocArray(uchar, VOXEL_CACHE_SIZE*2);
	m->voxels = m->data;
	m->lighting = m->voxels + VOXEL_CACHE_SIZE;

	allocVoxelGPUData(m);
}

void freeVoxelData(VoxelMesh* m) {
	if(m->generated) {
		free(m->data);
		m->voxels = 0;
		m->lighting = 0;
	}
}

void freeVoxelCompressedData(VoxelMesh* m) {
	if(m->generated) {
		free(m->compressedData);
		m->compressedVoxelsSize = 0;
		m->compressedLightingSize = 0;
	}
}

void freeVoxelGPUData(VoxelMesh* m) {
	if(m->uploaded) {
		glDeleteBuffers(1, &m->bufferId); m->bufferId = 0;
		glDeleteBuffers(1, &m->bufferTransId); m->bufferTransId = 0;

		if(STBVOX_CONFIG_MODE == 1) {
			glDeleteBuffers(1, &m->texBufferId); m->texBufferId = 0;
			glDeleteBuffers(1, &m->texBufferTransId); m->texBufferTransId = 0;
			glDeleteTextures(1, &m->textureId); m->textureId = 0;
			glDeleteTextures(1, &m->textureTransId); m->textureTransId = 0;
		}
	}	
}

void freeVoxelMesh(VoxelMesh* m) {
	if(m->generated) {
		if(!m->stored) freeVoxelData(m);
		else freeVoxelCompressedData(m);

		freeVoxelGPUData(m);
	}
}

void initVoxelMesh(VoxelMesh* m, Vec2i coord) {
	TIMER_BLOCK();

	*m = {};
	m->coord = coord;

	allocVoxelMesh(m);
}

void addVoxelMesh(VoxelData* voxelData, int hashIndex, int index) {
	VoxelArray* voxelList = voxelData->voxelHash + hashIndex;

	assert(voxelList->count < arrayCount(voxelList->data));

	voxelList->data[voxelList->count++] = index;
}

void addVoxelMesh(VoxelData* voxelData, Vec2i coord, int index) {
	int hashIndex = mod(coord.x*9 + coord.y*23, voxelData->voxelHashSize);

	return addVoxelMesh(voxelData, hashIndex, index);
}

VoxelMesh* getVoxelMesh(VoxelData* voxelData, Vec2i coord, bool create = true) {
	int hashIndex = mod(coord.x*9 + coord.y*23, voxelData->voxelHashSize);

	VoxelMesh* m = 0;
	// DArray<int>* voxelList = voxelData->voxelHash + hashIndex;
	VoxelArray* voxelList = voxelData->voxelHash + hashIndex;
	for(int i = 0; i < voxelList->count; i++) {
		VoxelMesh* mesh = voxelData->voxels.data + voxelList->data[i];
		if(mesh->coord == coord) {
			m = mesh;
			break;
		}
	}

	// Initialise new mesh.

	if(!m && create) {
		VoxelMesh mesh;
		initVoxelMesh(&mesh, coord);
		voxelData->voxels.push(mesh);
		m = voxelData->voxels.data + voxelData->voxels.count-1;

		int index = voxelData->voxels.count-1;
		addVoxelMesh(voxelData, hashIndex, index);
	}

	return m;
}

void generateVoxelMeshThreaded(void* data) {
	TIMER_BLOCK();

	GenerateMeshThreadedData* d = (GenerateMeshThreadedData*)data;
	VoxelWorldSettings* vs = d->vs;
	VoxelMesh* m = d->m;
	Vec2i coord = m->coord;

	// Start of at 1,1,1 because cache is 2 wider.
	uchar* voxels   = &m->voxels[voxelCacheArray(1,1,1)];
	uchar* lighting = &m->lighting[voxelCacheArray(1,1,1)];

	if(!m->generated) {
		// float worldHeightOffset = -0.1f;
		float worldHeightOffset = -0.1f;

		Vec3i min = vec3i(0,0,0);
		Vec3i max = vec3i(VOXEL_X,VOXEL_Y,VOXEL_Z);

		Vec3i treePositions[100];
		int treePositionsSize = 0;

		for(int y = min.y; y < max.y; y++) {
			for(int x = min.x; x < max.x; x++) {
				int gx = (coord.x*VOXEL_X)+x;
				int gy = (coord.y*VOXEL_Y)+y;

				float height = perlin2d(gx+4000+vs->startX, gy+4000+vs->startY, vs->worldFreq, vs->worldDepth);
				height += worldHeightOffset; 

				// float mod = perlin2d(gx+startXMod, gy+startYMod, 0.008f, 4);
				float perlinMod = perlin2d(gx+vs->startXMod, gy+vs->startYMod, vs->modFreq, vs->modDepth);
				float mod = lerp(perlinMod, -vs->modOffset, vs->modOffset);

				float modHeight = height+mod;
				int blockType;
	    			 if(modHeight <  vs->heightLevels[0]) blockType = BT_Sand; // sand
	    		else if(modHeight <  vs->heightLevels[1]) blockType = BT_Ground; // grass
	    		else if(modHeight <  vs->heightLevels[2]) blockType = BT_Stone; // stone
	    		else if(modHeight <= vs->heightLevels[3]) blockType = BT_Snow; // snow

	    		height = clamp01(height);
	    		// height = pow(height,3.5f);
	    		height = pow(height,vs->worldPowCurve);
	    		int blockHeight = lerp(height, vs->worldMin, vs->worldMax);

	    		// Blocktype.
	    		for(int z = 0; z < blockHeight; z++) {
	    			voxels[voxelCacheArray(x,y,z)] = blockType;
	    			lighting[voxelCacheArray(x,y,z)] = 0;
	    		}

	    		if(blockType == BT_Ground) {
	    			voxels[voxelCacheArray(x,y,blockHeight-1)] = BT_Grass;
	    		}

	    		// Air.
	    		for(int z = blockHeight; z < VOXEL_Z; z++) {
	    			voxels[voxelCacheArray(x,y,z)] = 0;
	    			lighting[voxelCacheArray(x,y,z)] = vs->globalLumen;
	    		}

	    		// Trees.
	    		if(blockType == BT_Ground && vs->treeNoise[y*VOXEL_Y + x] == 1 && 
	    			between(y, min.y+3, max.y-3) && between(x, min.x+3, max.x-3) && 
	    			between(perlinMod, 0.2f, 0.4f)) {
	    			treePositions[treePositionsSize++] = vec3i(x,y,blockHeight);
		    	}

		    	// Water.
		    	int waterLevelHeight = vs->waterLevelHeight;
		    	if(blockHeight < waterLevelHeight) {
		    		for(int z = blockHeight; z < waterLevelHeight; z++) {
		    			voxels[voxelCacheArray(x,y,z)] = BT_Water;

		    			Vec2i waterLightRange = vec2i(30, vs->globalLumen);
		    			int lightValue = mapRange(blockHeight, vs->worldMin, waterLevelHeight, waterLightRange.x, waterLightRange.y);

		    			lighting[voxelCacheArray(x,y,z)] = lightValue;
		    		}
		    	}
		    }
		}

		for(int i = 0; i < treePositionsSize; i++) {
			Vec3i p = treePositions[i];
			int treeHeight = randomInt(3,6);
			int crownHeight = randomInt(1,3);

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
						voxels[voxelCacheArray(x,y,z)] = BT_Leaves;    			
						lighting[voxelCacheArray(x,y,z)] = 0;    			
					}
				}
			}

			voxels[voxelCacheArray(min.x, min.y, max.z)] = 0;
			voxels[voxelCacheArray(min.x, min.y, min.z)] = 0;
			voxels[voxelCacheArray(min.x, max.y, max.z)] = 0;
			voxels[voxelCacheArray(min.x, max.y, min.z)] = 0;
			voxels[voxelCacheArray(max.x, min.y, max.z)] = 0;
			voxels[voxelCacheArray(max.x, min.y, min.z)] = 0;
			voxels[voxelCacheArray(max.x, max.y, max.z)] = 0;
			voxels[voxelCacheArray(max.x, max.y, min.z)] = 0;
			lighting[voxelCacheArray(min.x, min.y, max.z)] = vs->globalLumen;
			lighting[voxelCacheArray(min.x, min.y, min.z)] = vs->globalLumen;
			lighting[voxelCacheArray(min.x, max.y, max.z)] = vs->globalLumen;
			lighting[voxelCacheArray(min.x, max.y, min.z)] = vs->globalLumen;
			lighting[voxelCacheArray(max.x, min.y, max.z)] = vs->globalLumen;
			lighting[voxelCacheArray(max.x, min.y, min.z)] = vs->globalLumen;
			lighting[voxelCacheArray(max.x, max.y, max.z)] = vs->globalLumen;
			lighting[voxelCacheArray(max.x, max.y, min.z)] = vs->globalLumen;

			for(int i = 0; i < treeHeight; i++) {
				voxels[voxelCacheArray(p.x,p.y,p.z+i)] = BT_TreeLog;
				lighting[voxelCacheArray(p.x,p.y,p.z+i)] = 0;
			}
		}
	}

	atomicSub(&m->activeGeneration);
	m->generated = true;
}

void makeMeshThreaded(void* data) {	
	TIMER_BLOCK();
	
	MakeMeshThreadedData* d = (MakeMeshThreadedData*)data;
	VoxelMesh* m = d->m;

	uchar* cache = m->voxels;
	uchar* lightingCache = m->lighting;

	// gather voxel data in radius and copy to cache
	Vec2i coord = m->coord;
	for(int y = -1; y < 2; y++) {
		for(int x = -1; x < 2; x++) {
			if(x == 0 && y == 0) continue;

			Vec2i c = coord + vec2i(x,y);
			VoxelMesh* lm = getVoxelMesh(d->voxelData, c);
			uchar* voxels = &lm->voxels[voxelCacheArray(1,1,1)];
			uchar* lighting = &lm->lighting[voxelCacheArray(1,1,1)];

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

						int index = voxelCacheArray(x + mPos.x, y + mPos.y, z);
						int cacheIndex = voxelCacheArray(x + lPos.x, y + lPos.y, z+1);

						cache[cacheIndex] = voxels[index];
						lightingCache[cacheIndex] = lighting[index];
					}
				}
			}

			// make floor solid
			for(int y = 0; y < VC_Y; y++) {
				for(int x = 0; x < VC_X; x++) {
					cache[voxelCacheArray(x, y, 0)] = BT_Sand; // change
				}
			}
		}
	}

	stbvox_mesh_maker mm;
	stbvox_init_mesh_maker(&mm);
	stbvox_input_description* inputDesc = stbvox_get_input_description(&mm);
	*inputDesc = {};

	int meshBufferCapacity;
	int texBufferCapacity;
	int meshBufferTransCapacity;
	int texBufferTransCapacity;

	{
		meshBufferCapacity = kiloBytes(500);
		m->meshBuffer = (char*)malloc(meshBufferCapacity);
		texBufferCapacity = meshBufferCapacity/4;
		m->texBuffer = (char*)malloc(texBufferCapacity);

		meshBufferTransCapacity = kiloBytes(500);
		m->meshBufferTrans = (char*)malloc(meshBufferTransCapacity);
		texBufferTransCapacity = meshBufferTransCapacity/4;
		m->texBufferTrans = (char*)malloc(texBufferTransCapacity);
	} 

	stbvox_set_buffer(&mm, 0, 0, m->meshBuffer, meshBufferCapacity);
	if(STBVOX_CONFIG_MODE == 1) {
		stbvox_set_buffer(&mm, 0, 1, m->texBuffer, texBufferCapacity);
	}

	stbvox_set_buffer(&mm, 1, 0, m->meshBufferTrans, meshBufferTransCapacity);
	if(STBVOX_CONFIG_MODE == 1) {
		stbvox_set_buffer(&mm, 1, 1, m->texBufferTrans, texBufferTransCapacity);
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

	inputDesc->blocktype = &cache[voxelCacheArray(1,1,1)];
	inputDesc->lighting = &lightingCache[voxelCacheArray(1,1,1)];

	// stbvox_set_default_mesh(&mm, 0);
	int success = stbvox_make_mesh(&mm);



	stbvox_set_mesh_coordinates(&mm, coord.x*VOXEL_X, coord.y*VOXEL_Y,0);

	stbvox_get_transform(&mm, m->transform);
	float bounds [2][3]; stbvox_get_bounds(&mm, bounds);

	m->quadCount = stbvox_get_quad_count(&mm, 0);
	m->quadCountTrans = stbvox_get_quad_count(&mm, 1);

	m->bufferSizePerQuad = stbvox_get_buffer_size_per_quad(&mm, 0);
	m->textureBufferSizePerQuad = stbvox_get_buffer_size_per_quad(&mm, 1);



	atomicSub(&m->activeMaking);
	m->upToDate = true;
}

void restoreMeshThreaded(void* data);
void makeMesh(VoxelMesh* m, VoxelData* voxelData, VoxelWorldSettings* vs) {
	TIMER_BLOCK();

	bool notAllMeshsAreReady = false;
	Vec2i coord = m->coord;
	for(int y = -1; y < 2; y++) {
		for(int x = -1; x < 2; x++) {
			Vec2i c = coord + vec2i(x,y);
			VoxelMesh* lm = getVoxelMesh(voxelData, c);

			if(!lm->generated) {
				if(!threadQueueFull(theThreadQueue) && !lm->activeGeneration) {
					atomicAdd(&lm->activeGeneration);

					GenerateMeshThreadedData data = {lm, vs};
					threadQueueAdd(theThreadQueue, generateVoxelMeshThreaded, &data, sizeof(data));
				}
				notAllMeshsAreReady = true;

			} 

			if(lm->stored) {
				if(!threadQueueFull(theThreadQueue) && !lm->activeStoring) {
					atomicAdd(&lm->activeStoring);
					allocVoxelMesh(lm);
					threadQueueAdd(theThreadQueue, restoreMeshThreaded, lm);
				}

				notAllMeshsAreReady = true;
			}

		}
	}

	if(notAllMeshsAreReady) return;

	if(!m->upToDate) {
		if(!threadQueueFull(theThreadQueue) && !m->activeMaking) {

			atomicAdd(&m->activeMaking);
			MakeMeshThreadedData data = {m, voxelData};
			threadQueueAdd(theThreadQueue, makeMeshThreaded, &data, sizeof(data));
		}

		return;
	} 

	glNamedBufferData(m->bufferId, m->bufferSizePerQuad*m->quadCount, m->meshBuffer, GL_STATIC_DRAW);

	glNamedBufferData(m->texBufferId, m->textureBufferSizePerQuad*m->quadCount, m->texBuffer, GL_STATIC_DRAW);
	glTextureBuffer(m->textureId, GL_RGBA8UI, m->texBufferId);

	glNamedBufferData(m->bufferTransId, m->bufferSizePerQuad*m->quadCountTrans, m->meshBufferTrans, GL_STATIC_DRAW);

	glNamedBufferData(m->texBufferTransId, m->textureBufferSizePerQuad*m->quadCountTrans, m->texBufferTrans, GL_STATIC_DRAW);
	glTextureBuffer(m->textureTransId, GL_RGBA8UI, m->texBufferTransId);

	free(m->meshBuffer);
	free(m->texBuffer);
	free(m->meshBufferTrans);
	free(m->texBufferTrans);

	m->uploaded = true;
}

void compressVoxelData(VoxelMesh* mesh, uchar* buffer);
void storeMeshThreaded(void* data) {

	VoxelMesh* m = (VoxelMesh*)data;

	uchar* buffer = mallocArray(uchar, VOXEL_CACHE_SIZE);
	compressVoxelData(m, buffer);
	free(buffer);

	m->upToDate = false;
	m->uploaded = false;

	atomicSub(&m->activeStoring);
	m->compressionStep = true;
}

void decompressVoxelData(VoxelMesh* mesh);
void restoreMeshThreaded(void* data) {

	VoxelMesh* m = (VoxelMesh*)data;

	decompressVoxelData(m);
	freeVoxelCompressedData(m);

	atomicSub(&m->activeStoring);
	m->stored = false;
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
	if(!vm->generated || (vm->compressionStep || vm->stored)) return 0;

	Vec3i localCoord = voxelToLocalVoxel(voxel);
	uchar* block = &vm->voxels[voxelCacheArray(1 + localCoord.x, 1 + localCoord.y, 1 + localCoord.z)];

	return block;
}

// coord -> block
uchar* getBlockFromCoord(VoxelData* voxelData, Vec3 coord) {
	return getBlockFromVoxel(voxelData, coordToVoxel(coord));
}

// voxel -> lighting
uchar* getLightingFromVoxel(VoxelData* voxelData, Vec3i voxel) {
	VoxelMesh* vm = getVoxelMesh(voxelData, voxelToMesh(voxel));
	if(!vm->generated || (vm->compressionStep || vm->stored)) return 0;

	Vec3i localCoord = voxelToLocalVoxel(voxel);
	uchar* block = &vm->lighting[voxelCacheArray(1 + localCoord.x, 1 + localCoord.y, 1 + localCoord.z)];

	return block;
}

// coord -> lighting
uchar* getLightingFromCoord(VoxelData* voxelData, Vec3 coord) {
	return getLightingFromVoxel(voxelData, coordToVoxel(coord));
}

Vec3i getVoxelOffsetFromChunk(Vec2i chunk) {
	return vec3i(chunk * vec2i(VOXEL_X, VOXEL_Y), 0);
}

//

void getVoxelQuadFromFaceDir(Vec3 p, Vec3 faceDir, Vec3 vs[4], float size);
void drawVoxelCube(Vec3 pos, float size, Vec4 color, int type) {

	int colorPalleteIndex = blockColor[type];
	Vec3 blockColor;
	blockColor.r = colorPaletteCompact[colorPalleteIndex][0] / 255.0f;
	blockColor.g = colorPaletteCompact[colorPalleteIndex][1] / 255.0f;
	blockColor.b = colorPaletteCompact[colorPalleteIndex][2] / 255.0f;
	color.rgb = color.rgb * blockColor;
	color = gammaToLinear(color);

	int dirIndex = 1;
	for(int faceIndex = 0; faceIndex < 6; faceIndex++) {

		Vec3 faceDir = vec3(0,0,0);
		faceDir.e[abs(dirIndex)-1] = dirIndex > 0 ? 1 : -1;
		if(dirIndex > 0) dirIndex *= -1;
		else dirIndex = abs(dirIndex) + 1;

		Vec3 vs[4];
		getVoxelQuadFromFaceDir(pos, faceDir, vs, size);
		for(int i = 0; i < arrayCount(vs); i++) vs[i] = vs[i];

		int faceTextureId = texture1Faces[type][faceIndex];

		dcQuad(vs[0], vs[1], vs[2], vs[3], color, "voxelTextures", rect(0,0,1,1), faceTextureId);
	}
}

void setupVoxelUniforms(Vec3 cameraPos, Mat4 view, Mat4 proj, Vec3 fogColor, int viewDistance, Vec3 trans = vec3(0,0,0), Vec3 scale = vec3(1,1,1), Vec3 rotation = vec3(0,0,0)) {
	TIMER_BLOCK();

	uint texUnit1 = 0; 
	uint texUnit2 = 1; 
	uint faceUnit = 2; 

	buildColorPalette();

	Vec3 li = norm(vec3(0,0.5f,0.5f));
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
	// int loc = glGetUniformLocation(theGraphicsState->pipelineIds.voxelFragment, "light_source");
	// glProgramUniform3fv(theGraphicsState->pipelineIds.voxelFragment, loc, 2, (GLfloat*)light);
	pushUniform(SHADER_Voxel, 1, VOXEL_UNIFORM_LIGHT_SOURCE, light, 2);
	dcCube({light[0], vec3(3,3,3), vec4(lColor, 1), 0, vec3(0,0,0)});
	#endif

	// ambient direction is sky-colored upwards
	// "ambient" lighting is from above
	Vec3 li2 = norm(vec3(-1,1,1));
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
	
	// al.e2[3][3] = (float)1.0f/((VIEW_DISTANCE*VOXEL_X) - VOXEL_X);
	// al.e2[3][3] *= al.e2[3][3];

	al.e2[3][3] = (float)1.0f/((viewDistance*VOXEL_X) - VOXEL_X);
	al.e2[3][3] *= al.e2[3][3];

	ambientLighting = al;

	int texUnit[2] = {texUnit1, texUnit2};

	Vec4 camera = vec4(cameraPos, 1);

	for(int i = 0; i < STBVOX_UNIFORM_count; ++i) {
		stbvox_uniform_info sui;
		int success = stbvox_get_uniform_info(&sui, i);
		if(!success) continue;

		int count = sui.array_length;
		void* data = sui.default_value;

		     if(strCompare(sui.name, "facearray")) data = &faceUnit;
		else if(strCompare(sui.name, "tex_array")) data = texUnit;
		else if(strCompare(sui.name, "color_table")) data = colorPalette;
		else if(strCompare(sui.name, "ambient")) data = ambientLighting.e;
		else if(strCompare(sui.name, "camera_pos")) data = camera.e;
		else if(strCompare(sui.name, "transform")) continue;

		int type;
		     if(sui.type == STBVOX_UNIFORM_TYPE_sampler) type = UNIFORM_TYPE_INT;
		else if(sui.type == STBVOX_UNIFORM_TYPE_vec3) type = UNIFORM_TYPE_VEC3;
		else if(sui.type == STBVOX_UNIFORM_TYPE_vec4) type = UNIFORM_TYPE_VEC4;

		pushUniform(SHADER_Voxel, 2, sui.name, type, data, count);
	}	


	Mat4 model = modelMatrix(trans, scale, 0, rotation);
	Mat4 finalMat = proj*view*model;

	pushUniform(SHADER_Voxel, 1, "alphaTest", 0.0f);

	pushUniform(SHADER_Voxel, 0, "clipPlane", false);
	pushUniform(SHADER_Voxel, 0, "cPlane", 0,0,0,0);
	pushUniform(SHADER_Voxel, 0, "cPlane2", 0,0,0,0);
	pushUniform(SHADER_Voxel, 0, "model", &model);
	pushUniform(SHADER_Voxel, 0, "model_view", &finalMat);
}

void drawVoxelMesh(VoxelMesh* m, Vec2i chunkOffset, int drawMode = 0) {
	TIMER_BLOCK();

	glBindSamplers(0,3,theGraphicsState->samplers + SAMPLER_VOXEL_1);

	GLuint textureUnits[3];
	textureUnits[0] = getTexture("voxelTextures")->id;
	textureUnits[1] = getTexture("voxelTextures2")->id;

	bindShader(SHADER_Voxel);

	stbvox_mesh_maker mm;
	stbvox_set_mesh_coordinates(&mm, (-chunkOffset.x + m->coord.x)*VOXEL_X, (-chunkOffset.y + m->coord.y)*VOXEL_Y, 0);
	stbvox_get_transform(&mm, m->transform);

	pushUniform(SHADER_Voxel, 2, "transform", UNIFORM_TYPE_VEC3, &m->transform[0], 3);

	if(drawMode == 0 || drawMode == 2) {
		glBindBuffer(GL_ARRAY_BUFFER, m->bufferId);
		int vaLoc = glGetAttribLocation(getShader(SHADER_Voxel)->vertex, "attr_vertex");
		glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 0, (void*)0);
		glEnableVertexAttribArray(vaLoc);

		textureUnits[2] = m->textureId;
		glBindTextures(0,3,textureUnits);

		glDrawArrays(GL_QUADS, 0, m->quadCount*4);
	}

	if(drawMode == 0 || drawMode == 1) {
		glBindBuffer(GL_ARRAY_BUFFER, m->bufferTransId);
		int vaLoc = glGetAttribLocation(getShader(SHADER_Voxel)->vertex, "attr_vertex");
		glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 0, (void*)0);
		glEnableVertexAttribArray(vaLoc);

		textureUnits[2] = m->textureTransId;
		glBindTextures(0,3,textureUnits);

		glDrawArrays(GL_QUADS, 0, m->quadCountTrans*4);
	}
}


void loadVoxelTextures(Texture* texture, Texture* texture2, char* folderPath, float waterAlpha, int internalFormat, bool reload = false, int levelToReload = 0) {

	const int mipMapCount = 5;
	char* p = getTString(34);
	strClear(p);
	strAppend(p, (char*)folderPath);

	if(!reload) {
		texture->dim = vec2i(32,32);
		texture->channels = 4;
		texture->levels = 0;
		texture->type = TEXTURE_TYPE_3D;

		glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, &texture->id);
		glTextureStorage3D(texture->id, mipMapCount, internalFormat, 32, 32, BX_Size);
	}

	for(int layerIndex = 0; layerIndex < BX_Size; layerIndex++) {
		if(reload && layerIndex != levelToReload) continue;

		char* textureName = (char*)voxelTextureFiles[layerIndex];
		char* path = getTexture(textureName)->file;
		int x,y,n;
		unsigned char* stbData = stbi_load(path, &x, &y, &n, 4);

		if(layerIndex == BX_Water) {
			uchar* data = stbData;
			int size = x*y;
			for(int i = 0; i < size; i++) {
				data[i*4 + 3] = waterAlpha * 255;
			}
		}

		glTextureSubImage3D(texture->id, 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);

		stbi_image_free(stbData);
	}

	glGenerateTextureMipmap(texture->id);


	// Adjust mipmap alpha levels for tree textures.

	if(!reload || (reload && levelToReload == BX_Leaves))
	{
		int textureId = texture->id;

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

	{
		texture = texture2;

		if(!reload) {
			texture2->dim = vec2i(32,32);
			texture2->channels = 4;
			texture2->levels = 0;
			texture2->type = TEXTURE_TYPE_3D;

			glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, &texture->id);
			glTextureStorage3D(texture->id, 1, internalFormat, 32, 32, BX2_Size);
		}

		for(int layerIndex = 0; layerIndex < BX2_Size; layerIndex++) {
			if(reload && layerIndex != levelToReload) continue;

			char* textureName = (char*)voxelTextureFiles2[layerIndex];
			char* path = getTexture(textureName)->file;
			int x,y,n;
			unsigned char* stbData = stbi_load(path, &x, &y, &n, 4);
			
			glTextureSubImage3D(texture->id, 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);
			stbi_image_free(stbData);
		}
	}

	glGenerateTextureMipmap(texture->id);
}

bool collisionVoxelBox(VoxelData* voxelData, Vec3 boxPos, Vec3 boxSize, Vec2i chunkOffset, float* minDistance = 0, Vec3* collisionVoxel = 0, bool* inWater = 0) {

	// First get the mesh coords that touch the player box.

	Rect3 box = rect3CenDim(boxPos, boxSize);
	Vec3i voxelMin = coordToVoxel(box.min);
	Vec3i voxelMax = coordToVoxel(box.max+1);

	bool checkDistance = minDistance != 0 && collisionVoxel != 0;

	bool collision = false;
	if(checkDistance) *minDistance = FLT_MAX;

	// Check collision with the voxel that's closest.

	Vec3i voxelOffset = vec3i(chunkOffset * vec2i(VOXEL_X, VOXEL_Y), 0);

	for(int x = voxelMin.x; x < voxelMax.x; x++) {
		for(int y = voxelMin.y; y < voxelMax.y; y++) {
			for(int z = voxelMin.z; z < voxelMax.z; z++) {
				Vec3i coord = vec3i(x,y,z);
				uchar* block = getBlockFromVoxel(voxelData, coord + voxelOffset);

				if(block) {					
					if(*block > 1) {
						if(checkDistance) {
							Vec3 cBox = voxelToVoxelCoord(coord);
							float distance = len(boxPos - cBox);
							
							if(distance < *minDistance) {
								*minDistance = distance;
								*collisionVoxel = cBox;
							}
						}

						collision = true;
					}

					if(*block == BT_Water && inWater) *inWater = true;
				}
			}
		}
	}

	return collision;
}

bool collisionVoxelBoxDistance(VoxelData* voxelData, Vec3 pos, Vec3 size, Vec2i chunkOffset, Vec3* newPos, Vec3* collisionNormal) {

	*collisionNormal = vec3(0,0,0);

	bool result = true;

	int collisionCount = 0;
	bool collision = true;
	while(collision) {

		float md;
		Vec3 collisionBox;
		collision = collisionVoxelBox(voxelData, pos, size, chunkOffset, &md, &collisionBox);

		if(collision) {
			collisionCount++;

			float maxDistance = -FLT_MAX;
			Vec3 dir = vec3(0,0,0);

			// check all the 6 planes and take the one with the shortest distance
			Vec3 directions[] = {vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0), 
							 	 vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1),};
			for(int i = 0; i < 6; i++) {
				Vec3 n = directions[i];

				// assuming voxel size is 1
				// this could be simpler because the voxels are axis aligned
				Vec3 p = collisionBox + (n * ((size/2) + 0.5));
				float d = -dot(p, n);
				float d2 = dot(pos, n);

				// distances are lower then zero in this case where the point is 
				// not on the same side as the normal
				float distance = d + d2;

				if(i == 0 || distance > maxDistance) {
					maxDistance = distance;
					dir = n;
				}
			}

			// float error = 0;
			float error = 0.0001f;
			pos += dir*(-maxDistance + error);

			(*collisionNormal) += dir;
		}

		if(collisionCount > 5) {
			result = false;
			break;
		}
	}

	*newPos = pos;

	return result;
}

bool raycastGroundVoxelBox(VoxelData* voxelData, Vec3 pos, Vec3 size, Vec2i chunkOffset, uchar* collisionBlockType) {
	float raycastThreshold = 0.01f;

	bool groundCollision = false;

	Vec3 bottomCenter = pos - vec3(0,0,size.z/2);
	Vec2 offsets[] = {vec2(0,0), 
					  vec2(0.5f, 0.5f), vec2(-0.5f, 0.5f),
					  vec2(0.5f,-0.5f), vec2(-0.5f,-0.5f),};

	Vec3i voxelOffset = vec3i(chunkOffset * vec2i(VOXEL_X, VOXEL_Y), 0);

	for(int i = 0; i < 5; i++) {
		Vec3 gp = bottomCenter + vec3(offsets[i],0)*size;
		gp.z -= raycastThreshold;

		Vec3i voxel = coordToVoxel(gp);
		uchar* blockType = getBlockFromVoxel(voxelData, voxel + voxelOffset);

		if(blockType) {
			if(*blockType > 1) {
				groundCollision = true;
				if(collisionBlockType) *collisionBlockType = *blockType;
				break;
			}
		}
	}

	return groundCollision;
}


void compressVoxelData(VoxelMesh* mesh, uchar* buffer) {
	uchar* buf = buffer;
	int bufferCount = 0;

	for(int index = 0; index < 2; index++) {

		uchar* data = index == 0 ? mesh->voxels : mesh->lighting;

		uchar count = 1;
		uchar blockType = data[0];
		for(int i = 1; i < VOXEL_CACHE_SIZE; i++) {
			if(data[i] != blockType || count == UCHAR_MAX || i == VOXEL_CACHE_SIZE-1) {
				writeTypeAndAdvance(buf, count, uchar);
				writeTypeAndAdvance(buf, blockType, uchar);

				count = 1;
				blockType = data[i];

				bufferCount++;
			} else {
				count++;
			}
		}
		// Handle last element.
		writeTypeAndAdvance(buf, (uchar)1, uchar);
		writeTypeAndAdvance(buf, data[VOXEL_CACHE_SIZE-1], uchar);
		bufferCount++;

		if(index == 0) mesh->compressedVoxelsSize = bufferCount;
		else mesh->compressedLightingSize = bufferCount - mesh->compressedVoxelsSize;
	}

	mesh->compressedData = mallocArray(uchar, bufferCount * (sizeof(uchar)*2));
	memcpy(mesh->compressedData, buffer, bufferCount * (sizeof(uchar)*2));
}

void decompressVoxelData(VoxelMesh* mesh) {

	uchar* buf = mesh->compressedData;
	for(int i = 0; i < 2; i++) {
		uchar* data = i == 0 ? mesh->voxels : mesh->lighting;
		int count = i == 0 ? mesh->compressedVoxelsSize : mesh->compressedLightingSize;

		int pos = 0;
		for(int i = 0; i < count; i++) {
			uchar count = readTypeAndAdvance(buf, uchar);
			uchar type = readTypeAndAdvance(buf, uchar);

			for(int i = 0; i < count; i++) data[pos+i] = type;
			pos += count;
		}

	}

}

void resetVoxelHashAndMeshes(VoxelData* vd) {

	VoxelArray* voxelHash = vd->voxelHash;
	for(int i = 0; i < vd->voxelHashSize; i++) {
		voxelHash[i].count = 0;
	}

	DArray<VoxelMesh>* voxels = &vd->voxels;
	for(int i = 0; i < voxels->count; i++) {
		freeVoxelMesh(voxels->data + i);
	}

	voxels->clear();
}

void drawCubeMap(char* skybox, Entity* player, Entity* camera, bool playerMode, int fieldOfView, float aspectRatio, VoxelWorldSettings* voxelSettings, bool reflection) {

	bindShader(SHADER_CubeMap);
	glBindTextures(0, 1, &getTexture(skybox)->id);
	glBindSampler(0, 0);

	Vec3 skyBoxRot = playerMode ? player->rot : camera->rot;
	skyBoxRot.x += M_PI;

	Camera skyBoxCam = getCamData(vec3(0,0,0), skyBoxRot, vec3(0,0,0), vec3(0,1,0), vec3(0,0,1));
	pushUniform(SHADER_CubeMap, 0, "view", viewMatrix(skyBoxCam.pos, -skyBoxCam.look, skyBoxCam.up, skyBoxCam.right));
	pushUniform(SHADER_CubeMap, 0, "proj", projMatrix(degreeToRadian(fieldOfView), aspectRatio, 0.001f, 2));

	pushUniform(SHADER_CubeMap, 2, "clipPlane", reflection);
	if(reflection) {
		pushUniform(SHADER_CubeMap, 0, "cPlane", 0,0,-1,	voxelSettings->waterLevelHeight);
	}

	pushUniform(SHADER_CubeMap, 2, "fogColor", voxelSettings->fogColor);

	glDisable(GL_DEPTH_TEST);
	glFrontFace(GL_CCW);
		glDrawArrays(GL_TRIANGLES, 0, 6*6);
	glFrontFace(GL_CW);
	glEnable(GL_DEPTH_TEST);
}

void getVoxelQuadFromFaceDir(Vec3 p, Vec3 faceDir, Vec3 vs[4], float size) {

	Vec3* voxelVerts;
	     if(faceDir.x > 0) voxelVerts = voxelVertices + 0*4;
	else if(faceDir.x < 0) voxelVerts = voxelVertices + 1*4;
	else if(faceDir.y > 0) voxelVerts = voxelVertices + 2*4;
	else if(faceDir.y < 0) voxelVerts = voxelVertices + 3*4;
	else if(faceDir.z > 0) voxelVerts = voxelVertices + 4*4;
	else if(faceDir.z < 0) voxelVerts = voxelVertices + 5*4;
	else assert(false);

	for(int i = 0; i < 4; i++) vs[i] = p + voxelVerts[i]*(size * 0.5f);
}

void getVoxelQuadFromFaceDir(Vec3 p, int faceIndex, Vec3 vs[4], float size) {

	Vec3 dirs[] = {vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0), 
	               vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1), };

	return getVoxelQuadFromFaceDir(p, dirs[faceIndex], vs, size);
}

void getVoxelShowingVoxelFaceDirs(Vec3 dirFromVoxelToCam, Vec3 faceDirs[3]) {
	for(int i = 0; i < 3; i++) {
		faceDirs[i] = vec3(0,0,0);
		faceDirs[i].e[i] = dirFromVoxelToCam.e[i] < 0 ? -1 : 1;
	}
}

void entityWorldCollision(Vec3* pos, Vec3 dim, Vec3* vel, Vec2i chunk, VoxelData* vd, float dt) {

	float friction = 0.01f;
	float stillnessThreshold = 0.0001f;

	Vec3 newPos = *pos;
	Vec3 newVel = *vel;
	Vec3 cNormal;
	bool result = collisionVoxelBoxDistance(vd, newPos, dim, chunk, &newPos, &cNormal);

	bool groundRaycast = raycastGroundVoxelBox(vd, newPos, dim, chunk, 0);

	if(between(newVel.z, -stillnessThreshold, stillnessThreshold)) {
		newVel.z = 0;
	}

	// Wall bounce.
	if(cNormal.xy != vec2(0,0)) {
		newVel = reflectVector(newVel, cNormal);
		newVel *= 0.5f;
	}

	if(cNormal.xy != vec2(0,0) || groundRaycast) {
		newVel.x *= pow(friction,dt);
		newVel.y *= pow(friction,dt);
	} 

	if(cNormal.z != 0) {
		newVel.z = 0;
	}

	*pos = newPos;
	*vel = newVel;
}

void entityWorldCollision(Entity* e, VoxelData* vd, float dt) {
	return entityWorldCollision(&e->pos, e->dim, &e->vel, e->chunk, vd, dt);
}

#if 0
struct VoxelRaycastData { 
	Vec3 voxel;
	Vec3 pos;
	Vec3 dir;

	float t;

	Vec3 offset;
	Vec3 maxOffset;
};

VoxelRaycastData voxelRaycastInit(Vec3 pos, Vec3 dir) {
	VoxelRaycastData vd;
	vd->pos = pos;
	vd->dir = dir;

	vd->t = 0;
	vd->offset = 

	return vd;
}
#endif
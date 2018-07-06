
#define APP_NAME "VoxelDemo"

#define HOTRELOAD_SHADERS 1
#define USE_SRGB 1
const int INTERNAL_TEXTURE_FORMAT = USE_SRGB ? GL_SRGB8_ALPHA8 : GL_RGBA8;

#define COLOR_SRGB(color) \
	(theGraphicsState->useSRGB ? linearToGamma(color) : color);

#define Editor_Executable_Path "C:\\Program Files\\Sublime Text 3\\sublime_text.exe"
#define Windows_Font_Folder "\\Fonts\\"
#define Windows_Font_Path_Variable "windir"
	
#define App_Session_File   ".\\session.tmp"
#define Game_Settings_File ".\\settings.tmp"
#define Saves_Folder       ".\\saves\\"
#define Save_State1        "saveState1.sav"

#ifdef SHIPPING_MODE
#define DATA_FOLDER(str) ".\\data\\" str
#else 
#define DATA_FOLDER(str) "..\\data\\" str
#endif

#define Gui_Session_File   DATA_FOLDER("guiSettings.tmp")
#define App_Font_Folder    DATA_FOLDER("Fonts\\")
#define App_Audio_Folder   DATA_FOLDER("Audio\\")
#define App_Texture_Folder DATA_FOLDER("Textures\\")

#define CubeMap_Texture_Folder "skyboxes\\"
#define Voxel_Texture_Folder "minecraft\\"

//

struct AppSessionSettings {
	Rect windowRect;
};

void appWriteSessionSettings(char* filePath, AppSessionSettings* at) {
	writeDataToFile((char*)at, sizeof(AppSessionSettings), filePath);
}

void appReadSessionSettings(char* filePath, AppSessionSettings* at) {
	readDataFromFile((char*)at, filePath);
}

void saveAppSettings(AppSessionSettings at) {
	if(fileExists(App_Session_File)) {
		appWriteSessionSettings(App_Session_File, &at);
	}
}

// Mesh.

struct Vertex {
	Vec3 pos;
	Vec2 uv;
	Vec3 normal;
};

struct MeshMap {
	Vertex* vertexArray;
	int size;
};

const Vertex meshCube[] = {
	{vec3(-0.5f,-0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f,-0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f,-0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f,-0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f, 0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f, 0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f, 0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f,-0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f, 0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f, 0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f,-0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3( 0.5f,-0.5f, 0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3( 0.5f, 0.5f,-0.5f), vec2(1,0), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f,-0.5f), vec2(0,0), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f,-0.5f), vec2(0,1), vec3(0,0,1)},
	{vec3(-0.5f, 0.5f, 0.5f), vec2(1,1), vec3(0,0,1)},
	{vec3(-0.5f,-0.5f, 0.5f), vec2(1,0), vec3(0,0,1)},
};

const Vertex meshQuad[] = {
	{vec3(-0.5f,-0.5f, 0), vec2(0,1), vec3(1,1,1)},
	{vec3(-0.5f, 0.5f, 0), vec2(0,0), vec3(1,1,1)},
	{vec3( 0.5f, 0.5f, 0), vec2(1,0), vec3(1,1,1)},
	{vec3( 0.5f,-0.5f, 0), vec2(1,1), vec3(1,1,1)},
};

#define MESHLIST \
	MESHFUNC(Quad) \
	MESHFUNC(Cube)

//

#define MESHFUNC(name) MESH_##name,
enum MeshId {
	MESH_START = -1,
	MESHLIST
	MESH_SIZE,
};
#undef MESHFUNC

#define MESHFUNC(name) {(Vertex*)mesh##name, sizeof(mesh##name)},
MeshMap meshArrays[] = {
	MESHLIST
};
#undef MESHFUNC

// Sampler.

enum SamplerType {
	SAMPLER_NORMAL = 0,
	SAMPLER_VOXEL_1,
	SAMPLER_VOXEL_2,
	SAMPLER_VOXEL_3,
	SAMPLER_SIZE,
};

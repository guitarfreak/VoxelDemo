
#define USE_SRGB 1
const int INTERNAL_TEXTURE_FORMAT = USE_SRGB ? GL_SRGB8_ALPHA8 : GL_RGBA8;

#define COLOR_SRGB(color) \
	(globalGraphicsState->useSRGB ? colorSRGB(color) : color);

#define APP_NAME "VoxelGame"


#define editor_executable_path "C:\\Program Files\\Sublime Text 3\\sublime_text.exe"

#define HOTRELOAD_SHADERS 1

#define App_Session_File ".\\session.tmp"

#ifdef SHIPPING_MODE
#define DATA_FOLDER(str) ".\\data\\" str
#else 
#define DATA_FOLDER(str) "..\\data\\" str
#endif

#define SAVES_FOLDER ".\\saves\\"
#define SAVE_STATE1 "saveState1.sav"

#define GUI_SETTINGS_FILE DATA_FOLDER("guiSettings.txt")

#define App_Font_Folder DATA_FOLDER("Fonts\\")
// #define App_Image_Folder DATA_FOLDER("Images\\")
#define App_Audio_Folder DATA_FOLDER("Audio\\")

#define FONT_SOURCESANS_PRO "LiberationSans-Regular.ttf"
#define FONT_CONSOLAS "consola.ttf"
#define FONT_CALIBRI "LiberationSans-Regular.ttf"

// #define FONT_SOURCESANS_PRO "SourceSansPro-Regular.ttf"
// #define FONT_CONSOLAS "consola.ttf"
// #define FONT_CALIBRI "calibri.ttf"

#define Windows_Font_Folder "\\Fonts\\"
#define Windows_Font_Path_Variable "windir"

//

const char* watchFolders[] = {
	DATA_FOLDER("Textures\\Misc\\"),
	DATA_FOLDER("Textures\\Skyboxes\\"),
	DATA_FOLDER("Textures\\Minecraft\\"),
};

struct AppSessionSettings {
	Rect windowRect;
};

void appWriteSessionSettings(char* filePath, AppSessionSettings* at) {
	writeDataToFile((char*)at, sizeof(AppSessionSettings), filePath);
}

void appReadSessionSettings(char* filePath, AppSessionSettings* at) {
	readDataFile((char*)at, filePath);
}

void saveAppSettings(AppSessionSettings at) {
	if(fileExists(App_Session_File)) {
		appWriteSessionSettings(App_Session_File, &at);
	}
}


//

enum TextureId {
	TEXTURE_WHITE = 0,
	TEXTURE_RECT,
	TEXTURE_CIRCLE,
	TEXTURE_TEST,
	TEXTURE_SIZE,
};

char* texturePaths[] = {
	DATA_FOLDER("Textures\\Misc\\white.png"),
	DATA_FOLDER("Textures\\Misc\\rect.png"),
	DATA_FOLDER("Textures\\Misc\\circle.png"),
	DATA_FOLDER("Textures\\Misc\\test.png"),
};

//

enum CubeMapIds {
	// CUBEMAP_1 = 0,
	// CUBEMAP_2,
	// CUBEMAP_3,
	// CUBEMAP_4,
	// CUBEMAP_5,
	CUBEMAP_5 = 0,
	CUBEMAP_SIZE,
};

char* cubeMapPaths[] = {
	// "..\\data\\Textures\\Skyboxes\\sb1.png",
	// "..\\data\\Textures\\Skyboxes\\sb2.png", 
	// "..\\data\\Textures\\Skyboxes\\sb3.jpg", 
	// "..\\data\\Textures\\Skyboxes\\sb4.png", 
	DATA_FOLDER("Textures\\Skyboxes\\xoGVD3X.jpg"),
};

//
// Shaders.
//

struct Vertex {
	Vec3 pos;
	Vec2 uv;
	Vec3 normal;
};

struct MeshMap {
	Vertex* vertexArray;
	int size;
};

enum MeshId {
	MESH_CUBE = 0,
	MESH_QUAD,
	MESH_SIZE,
};

const Vertex cubeArray[] = {
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

const Vertex quadArray[] = {
	{vec3(-0.5f,-0.5f, 0), vec2(0,1), vec3(1,1,1)},
	{vec3(-0.5f, 0.5f, 0), vec2(0,0), vec3(1,1,1)},
	{vec3( 0.5f, 0.5f, 0), vec2(1,0), vec3(1,1,1)},
	{vec3( 0.5f,-0.5f, 0), vec2(1,1), vec3(1,1,1)},
};

MeshMap meshArrays[] = {
	{(Vertex*)cubeArray, sizeof(cubeArray)},
	{(Vertex*)quadArray, sizeof(quadArray)},
};

//

enum SamplerType {
	SAMPLER_NORMAL = 0,
	SAMPLER_VOXEL_1,
	SAMPLER_VOXEL_2,
	SAMPLER_VOXEL_3,
	SAMPLER_SIZE,
};

//

enum FrameBufferType {
	FRAMEBUFFER_3dMsaa = 0,
	FRAMEBUFFER_3dNoMsaa,
	FRAMEBUFFER_Reflection,
	FRAMEBUFFER_2d,

	FRAMEBUFFER_DebugMsaa,
	FRAMEBUFFER_DebugNoMsaa,

	FRAMEBUFFER_SIZE,
};

//


struct ShaderUniformType {
	uint type;
	char* name;
};

struct MakeShaderInfo {
	char* vertexString;
	char* fragmentString;

	int uniformCount;
	ShaderUniformType* uniformNameMap;
};

enum UniformType {
	UNIFORM_TYPE_VEC4 = 0,
	UNIFORM_TYPE_VEC3,
	UNIFORM_TYPE_VEC2,
	UNIFORM_TYPE_MAT4,
	UNIFORM_TYPE_INT,
	UNIFORM_TYPE_FLOAT,

	UNIFORM_TYPE_SIZE,
};

#define GLSL(src) "#version 430\n \
	#extension GL_ARB_bindless_texture 			: enable\n \
	#extension GL_ARB_shading_language_include 	: enable\n \
	#extension GL_ARB_uniform_buffer_object 	: enable\n \
	#extension GL_ARB_gpu_shader5               : enable\n \
	#extension GL_ARB_gpu_shader_fp64           : enable\n \
	#extension GL_ARB_shader_precision          : enable\n \
	#extension GL_ARB_conservative_depth        : enable\n \
	#extension GL_ARB_texture_cube_map_array    : enable\n \
	#extension GL_ARB_separate_shader_objects   : enable\n \
	#extension GL_ARB_shading_language_420pack  : enable\n \
	#extension GL_ARB_shading_language_packing  : enable\n \
	#extension GL_ARB_explicit_uniform_location : enable\n" #src

//

enum CubeUniforms {
	CUBE_UNIFORM_MODEL = 0,
	CUBE_UNIFORM_VIEW,
	CUBE_UNIFORM_PROJ,
	CUBE_UNIFORM_COLOR,
	CUBE_UNIFORM_MODE,
	CUBE_UNIFORM_VERTICES,
	CUBE_UNIFORM_CPLANE,
	CUBE_UNIFORM_UV,
	CUBE_UNIFORM_ALPHA_TEST,
	CUBE_UNIFORM_ALPHA,
	CUBE_UNIFORM_SIZE,
};

ShaderUniformType cubeShaderUniformType[] = {
	{UNIFORM_TYPE_MAT4, "model"},
	{UNIFORM_TYPE_MAT4, "view"},
	{UNIFORM_TYPE_MAT4, "proj"},
	{UNIFORM_TYPE_VEC4, "setColor"},
	{UNIFORM_TYPE_INT,  "mode"},
	{UNIFORM_TYPE_VEC3, "vertices"},
	{UNIFORM_TYPE_VEC4, "cPlane"},
	{UNIFORM_TYPE_VEC2, "setUV"},
	{UNIFORM_TYPE_FLOAT, "alpha"},
	{UNIFORM_TYPE_INT, "alphaTest"},
};

const char* vertexShaderCube = GLSL (
	out gl_PerVertex { vec4 gl_Position; };
	out vec4 Color;
	smooth out vec3 uv;

	layout(location = 0) in vec3 attr_verts;
	layout(location = 1) in vec2 attr_texUV;
	layout(location = 2) in vec3 attr_normal;

	uniform mat4x4 model;
	uniform mat4x4 view;
	uniform mat4x4 proj;

	uniform bool mode;

	uniform vec3 vertices[24];
	uniform vec2 setUV[24];
	uniform vec4 setColor;

	void main() {
		Color = setColor;

		vec4 pos;
		if(mode == true) {
			pos = vec4(vertices[gl_VertexID], 1);
			gl_Position = proj*view*pos;
			uv = vec3(setUV[gl_VertexID],0);
		} else {
			pos = vec4(attr_verts, 1);
			gl_Position = proj*view*model*pos;
			uv = vec3(attr_texUV, 0);
		}
	}
);

const char* fragmentShaderCube = GLSL (
	layout(binding = 0) uniform sampler2D s;

	smooth in vec3 uv;
	in vec4 Color;

	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	uniform bool alphaTest = false;
	uniform float alpha = 0.5f;

	void main() {
		color = texture(s, uv.xy) * Color;

		if(alphaTest) {
			if(color.a <= alpha) discard;
		}
	}
);

//

enum QuadUniforms {
	QUAD_UNIFORM_UV = 0,
	QUAD_UNIFORM_TEXZ,
	QUAD_UNIFORM_MOD,
	QUAD_UNIFORM_COLOR,
	QUAD_UNIFORM_CAMERA,
	
	QUAD_UNIFORM_PRIMITIVE_MODE,
	QUAD_UNIFORM_VERTS,

	QUAD_UNIFORM_SIZE,
};

ShaderUniformType quadShaderUniformType[] = {
	{UNIFORM_TYPE_VEC4, "setUV"},
	{UNIFORM_TYPE_FLOAT, "texZ"},
	{UNIFORM_TYPE_VEC4, "mod"},
	{UNIFORM_TYPE_VEC4, "setColor"},
	{UNIFORM_TYPE_VEC4, "camera"},

	{UNIFORM_TYPE_INT, "primitiveMode"},
	{UNIFORM_TYPE_VEC2, "verts"},
};

const char* vertexShaderQuad = GLSL (
	const vec2 quad[] = vec2[] (
		vec2( -0.5f, -0.5f ),
		vec2( -0.5f,  0.5f ),
		vec2(  0.5f, -0.5f ),
		vec2(  0.5f,  0.5f )
		);

	const ivec2 quad_uv[] = ivec2[] (
		ivec2(  0.0,  0.0 ),
		ivec2(  0.0,  1.0 ),
		ivec2(  1.0,  0.0 ),
		ivec2(  1.0,  1.0 )
		);

	uniform vec4 setUV;
	uniform float texZ;
	uniform vec4 mod;
	uniform vec4 setColor;
	uniform vec4 camera; // left bottom right top

	uniform bool primitiveMode = false;
	uniform vec2 verts[32];

	out gl_PerVertex { vec4 gl_Position; };
	smooth out vec3 uv;
	out vec4 Color;

	void main() {

		if(primitiveMode) {
			uv = vec3(0,0,-1);
			Color = setColor;

			vec2 model = verts[gl_VertexID];
			vec2 view = model/(camera.zw*0.5f) - camera.xy/(camera.zw*0.5f);
			gl_Position = vec4(view, 0, 1);

		} else {
			
			ivec2 pos = quad_uv[gl_VertexID];
			uv = vec3(setUV[pos.x], setUV[2 + pos.y], texZ);
			vec2 v = quad[gl_VertexID];

			Color = setColor;

			vec2 model = v*mod.zw + mod.xy;
			vec2 view = model/(camera.zw*0.5f) - camera.xy/(camera.zw*0.5f);
			gl_Position = vec4(view, 0, 1);
		}

	}
);

const char* fragmentShaderQuad = GLSL (
	layout(binding = 0) uniform sampler2D s;
	layout(binding = 1) uniform sampler2DArray sArray[2];

// smooth in vec2 uv;
	smooth in vec3 uv;
	in vec4 Color;

	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	void main() {
		vec4 texColor;
		if(uv.z > -1) texColor = texture(sArray[0], vec3(uv.xy, floor(uv.z)));
		else texColor = texture(s, uv.xy);

		color = texColor * Color;
	}
);

//

enum TestUniforms {
	TEST_UNIFORM_SIZE = 0,
};

// struct {
//   ResourceGLuint  
//     box_vbo,
//     box_ibo,
//     sphere_vbo,
//     sphere_ibo,

//     scene_ubo,
//     objects_ubo;
// };

// struct {
//   GLuint64  
//     box_vbo,
//     box_ibo,
//     sphere_vbo,
//     sphere_ibo,

//     scene_ubo,
//     objects_ubo;
// };

struct SceneData {
  // mat4  viewProjMatrix;
  // mat4  viewProjMatrixI;
  // mat4  viewMatrix;
  // mat4  viewMatrixI;
  // mat4  viewMatrixIT;
  
  // vec4  wLightPos;
  
  // uvec2 viewport;
  // float shrinkFactor;
  // float time;
	Vec4 color;
};

const char* vertexShaderTest = GLSL (
	out gl_PerVertex { vec4 gl_Position; };

	// layout(commandBindableNV) uniform;

	struct SceneData {
	  // mat4  viewProjMatrix;
	  // mat4  viewProjMatrixI;
	  // mat4  viewMatrix;
	  // mat4  viewMatrixI;
	  // mat4  viewMatrixIT;
	  
	  // vec4  wLightPos;
	  
	  // uvec2 viewport;
	  // float shrinkFactor;
	  // float time;
		vec4 color;
	};

	// const vec4 vertices[] = vec4[](
	// 		vec4(-0.5f, -0.5f, 0.0f, 1.0f),
	// 		vec4(-0.5f,  0.5f, 0.0f, 1.0f),
	// 		vec4( 0.5f, -0.5f, 0.0f, 1.0f),
	// 		vec4( 0.5f,  0.5f, 0.0f, 1.0f));

	const vec4 vertices[] = vec4[](
			vec4(-0.1f, -0.1f, 0.0f, 1.0f),
			vec4(-0.1f,  0.1f, 0.0f, 1.0f),
			vec4( 0.1f, -0.1f, 0.0f, 1.0f),
			vec4( 0.1f,  0.1f, 0.0f, 1.0f));

	// layout(std140, binding = 1) uniform vec2 pos;
	// layout(binding = 1) uniform vec2 pos;

	// layout(binding = 1) uniform float sArray;
	// layout(binding = 1) uniform sampler2D sArray;

	layout(std140, binding = 0) uniform sceneBuffer {
	  SceneData scene;
		// vec2 pos;
		// vec4 color;
	};


	// layout(std140, binding = 0) uniform vec4 color;

	// layout(std140,binding=UBO_OBJECT) uniform objectBuffer {
	  // ObjectData  object;
	// };

	out vec4 Color;

	void main() {
		gl_Position = vertices[gl_VertexID];
		// Color = vec4(1,0,0,1);
		// Color = color;
		Color = scene.color;
	}
);

const char* fragmentShaderTest = GLSL (
	in vec4 Color;
	out vec4 color;

	void main() {
		// color = vec4(1,1,1,1);
		color = Color;
		// color = scene.color;
	}
);

//

enum ParticleUniforms {
	PARTICLE_UNIFORM_MODEL = 0,
	PARTICLE_UNIFORM_VIEW,
	PARTICLE_UNIFORM_PROJ,

	PARTICLE_UNIFORM_SIZE,
};

ShaderUniformType particleShaderUniformType[] = {
	{UNIFORM_TYPE_MAT4, "model"},
	{UNIFORM_TYPE_MAT4, "view"},
	{UNIFORM_TYPE_MAT4, "proj"},
};

struct ParticleVertex {
	Mat4 m;
	Vec4 c;
};

const char* vertexShaderParticle = GLSL (
	out gl_PerVertex { vec4 gl_Position; };
	out vec4 Color;
	out vec2 uv;

	// layout(location = 0) in vec3 attr_verts;
	// layout(location = 1) in vec2 attr_texUV;
	// // layout(location = 2) in vec3 attr_normal;
	// layout(location = 3) in vec4 attr_color;

	layout(location = 0) uniform vec3 vertices[4];
	layout(location = 4) uniform vec2 attr_texUV[4];

	layout(location = 8) in vec4 attr_color;
	layout(location = 9) in mat4x4 model;

	uniform mat4x4 view;
	uniform mat4x4 proj;

	void main() {
		gl_Position = proj*view*model*vec4(vertices[gl_VertexID], 1);
		// gl_Position = proj*view*vec4(vertices[gl_VertexID], 1);

		// gl_Position = proj*view*model*vec4(attr_verts, 1);
		uv = attr_texUV[gl_VertexID];
		Color = attr_color;
	}
);

const char* fragmentShaderParticle = GLSL (
	layout(binding = 0) uniform sampler2D s;

	in vec2 uv;
	in vec4 Color;

	out float gl_FragDepth;
	out vec4 color;

	// uniform float alphaTest = 0.5f;

	void main() {
		color = texture(s, uv) * Color;
		// color = texture(s, uv.xy);
		// color = Color;
		// color = vec4(1,0,0,1);

		// if(color.a <= alphaTest) discard;
	}
);

//

enum CubemapUniforms {
	CUBEMAP_UNIFORM_VIEW = 0,
	CUBEMAP_UNIFORM_PROJ,
	CUBEMAP_UNIFORM_CLIPPLANE,
	CUBEMAP_UNIFORM_CPLANE1,
	CUBEMAP_UNIFORM_FOGCOLOR,

	CUBEMAP_UNIFORM_SIZE,
};

ShaderUniformType cubemapShaderUniformType[] = {
	{UNIFORM_TYPE_MAT4, "view"},
	{UNIFORM_TYPE_MAT4, "proj"},
	{UNIFORM_TYPE_INT, "clipPlane"},
	{UNIFORM_TYPE_VEC4, "cPlane"},
	{UNIFORM_TYPE_VEC4, "fogColor"},
};

const char* vertexShaderCubeMap = GLSL (
	const vec3 cube[] = vec3[] (
	  vec3( -1.0,  1.0, -1.0 ),
	  vec3( -1.0, -1.0, -1.0 ),
	  vec3(  1.0, -1.0, -1.0 ),
	  vec3(  1.0, -1.0, -1.0 ),
	  vec3(  1.0,  1.0, -1.0 ),
	  vec3( -1.0,  1.0, -1.0 ),

	  vec3( -1.0, -1.0,  1.0 ),
	  vec3( -1.0, -1.0, -1.0 ),
	  vec3( -1.0,  1.0, -1.0 ),
	  vec3( -1.0,  1.0, -1.0 ),
	  vec3( -1.0,  1.0,  1.0 ),
	  vec3( -1.0, -1.0,  1.0 ),

	  vec3(  1.0, -1.0, -1.0 ),
	  vec3(  1.0, -1.0,  1.0 ),
	  vec3(  1.0,  1.0,  1.0 ),
	  vec3(  1.0,  1.0,  1.0 ),
	  vec3(  1.0,  1.0, -1.0 ),
	  vec3(  1.0, -1.0, -1.0 ),

	  vec3( -1.0, -1.0,  1.0 ),
	  vec3( -1.0,  1.0,  1.0 ),
	  vec3(  1.0,  1.0,  1.0 ),
	  vec3(  1.0,  1.0,  1.0 ),
	  vec3(  1.0, -1.0,  1.0 ),
	  vec3( -1.0, -1.0,  1.0 ),

	  vec3( -1.0,  1.0, -1.0 ),
	  vec3(  1.0,  1.0, -1.0 ),
	  vec3(  1.0,  1.0,  1.0 ),
	  vec3(  1.0,  1.0,  1.0 ),
	  vec3( -1.0,  1.0,  1.0 ),
	  vec3( -1.0,  1.0, -1.0 ),

	  vec3( -1.0, -1.0, -1.0 ),
	  vec3( -1.0, -1.0,  1.0 ),
	  vec3(  1.0, -1.0, -1.0 ),
	  vec3(  1.0, -1.0, -1.0 ),
	  vec3( -1.0, -1.0,  1.0 ),
	  vec3(  1.0, -1.0,  1.0 )
	);

	out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[]; };

	uniform mat4x4 view;
	uniform mat4x4 proj;

	smooth out vec3 pos;

	uniform bool clipPlane = false;
	uniform vec4 cPlane;

	void main() {
		pos = cube[gl_VertexID];

		if(clipPlane) {
			gl_ClipDistance[0] = dot(cPlane, vec4(pos,1));
			// pos.z *= -1;
			// pos.y *= -1;
		}

		gl_Position = proj*view*vec4(pos,1);
	}
);

const char* fragmentShaderCubeMap = GLSL (
	layout(depth_less) out float gl_FragDepth;
	layout(binding = 0) uniform samplerCubeArray s;
	smooth in vec3 pos;

	out vec4 color;

	uniform bool clipPlane = false;

	uniform vec4 fogColor;

	float mapRange01(float value, float min, float max) {
		float off = min < 0 ? abs(min) : -min;
		return ((value+off)/((max+off)-(min+off)));
	};

	void main() {
		vec3 clipPos = pos;
		if(clipPlane) clipPos.y *= -1;
		// color = texture(s, vec4(clipPos, 0));

		float d0 = -0.01f;
		if(clipPos.y <= 0) {
			vec4 c = texture(s, vec4(clipPos, 0));

			if(clipPos.y >= d0) {
				float f = mapRange01(clipPos.y, d0, 0);
				color = mix(fogColor, c, f);

			} else color = fogColor;

		} else color = texture(s, vec4(clipPos, 0));

		// color = texture(s, vec4(pos, 0));
	}
);

//

enum VoxelUniforms {
	VOXEL_UNIFORM_FACE_DATA = 0,
	VOXEL_UNIFORM_TRANSFORM,
	VOXEL_UNIFORM_TEX_ARRAY,
	VOXEL_UNIFORM_TEXSCALE,
	VOXEL_UNIFORM_COLOR_TABLE,
	VOXEL_UNIFORM_NORMALS,
	VOXEL_UNIFORM_TEXGEN,
	VOXEL_UNIFORM_AMBIENT,
	VOXEL_UNIFORM_CAMERA_POS,

	VOXEL_UNIFORM_LIGHT_SOURCE,

	VOXEL_UNIFORM_MODEL,
	VOXEL_UNIFORM_MODEL_VIEW,
	VOXEL_UNIFORM_CLIPPLANE,
	VOXEL_UNIFORM_CPLANE1,
	VOXEL_UNIFORM_CPLANE2,
	VOXEL_UNIFORM_ALPHATEST,

	VOXEL_UNIFORM_SIZE,
};

ShaderUniformType voxelShaderUniformType[] = {
	{UNIFORM_TYPE_INT, "facearray"},
	{UNIFORM_TYPE_VEC3, "transform"},
	{UNIFORM_TYPE_INT, "tex_array"},
	{UNIFORM_TYPE_VEC4, "texscale"},
	{UNIFORM_TYPE_VEC4, "color_table"},
	{UNIFORM_TYPE_VEC3, "normal_table"},
	{UNIFORM_TYPE_VEC3, "texgen"},
	{UNIFORM_TYPE_VEC4, "ambient"},
	{UNIFORM_TYPE_VEC4, "camera_pos"},

	{UNIFORM_TYPE_VEC3, "light_source"},

	{UNIFORM_TYPE_MAT4, "model"},
	{UNIFORM_TYPE_MAT4, "model_view"},
	{UNIFORM_TYPE_INT, "clipPlane"},
	{UNIFORM_TYPE_VEC4, "cPlane"},
	{UNIFORM_TYPE_VEC4, "cPlane2"},
	{UNIFORM_TYPE_FLOAT, "alphaTest"},
};

// Shader defined in stb_voxel_render.h

enum ShaderProgram {
	SHADER_CUBE = 0,
	SHADER_QUAD,
	SHADER_VOXEL,
	SHADER_TEST,
	SHADER_PARTICLE,
	SHADER_CUBEMAP,

	SHADER_SIZE,
};

MakeShaderInfo makeShaderInfo[SHADER_SIZE] = {
	{(char*)vertexShaderCube, (char*)fragmentShaderCube, CUBE_UNIFORM_SIZE, cubeShaderUniformType},
	{(char*)vertexShaderQuad, (char*)fragmentShaderQuad, QUAD_UNIFORM_SIZE, quadShaderUniformType},
	{(char*)stbvox_get_vertex_shader(), (char*)stbvox_get_fragment_shader(), VOXEL_UNIFORM_SIZE, voxelShaderUniformType},
	{(char*)vertexShaderTest, (char*)fragmentShaderTest, TEST_UNIFORM_SIZE, 0},
	{(char*)vertexShaderParticle, (char*)fragmentShaderParticle, PARTICLE_UNIFORM_SIZE, particleShaderUniformType},
	{(char*)vertexShaderCubeMap, (char*)fragmentShaderCubeMap, CUBEMAP_UNIFORM_SIZE, cubemapShaderUniformType},
};


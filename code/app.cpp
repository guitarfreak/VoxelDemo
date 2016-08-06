#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <gl\gl.h>
#include <gl\glext.h>

#include "rt_misc.h"
#include "rt_math.h"

#include "rt_hotload.h"
#include "rt_misc_win32.h"
#include "rt_platformWin32.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"


// char* dataFolder = "...\\data\\";
// #define GetFilePath(name) ...\\data\\##name

// #define GL_TEXTURE_CUBE_MAP_SEAMLESS      0x884F
// #define GL_FRAMEBUFFER_SRGB               0x8DB9
// #define GL_FRAMEBUFFER_SRGB               0x8DB9
// #define GL_TEXTURE_BUFFER                 0x8C2A
// #define GL_MAP_WRITE_BIT                  0x0002
// #define GL_MAP_PERSISTENT_BIT             0x0040
// #define GL_MAP_COHERENT_BIT               0x0080
// #define GL_VERTEX_SHADER                  0x8B31
// #define GL_FRAGMENT_SHADER                0x8B30
// #define GL_VERTEX_SHADER_BIT              0x00000001
// #define GL_FRAGMENT_SHADER_BIT            0x00000002
// #define GL_DEBUG_OUTPUT                   0x92E0
// #define WGL_CONTEXT_FLAGS_ARB             0x2094
// #define WGL_CONTEXT_DEBUG_BIT_ARB         0x0001
// #define WGL_CONTEXT_MAJOR_VERSION_ARB     0x2091
// #define WGL_CONTEXT_MINOR_VERSION_ARB     0x2092
// #define GL_MAJOR_VERSION                  0x821B
// #define GL_MINOR_VERSION                  0x821C
// #define GL_RGB32F                         0x8815
// #define GL_RGBA8I                         0x8D8E
// #define GL_RGBA8UI                        0x8D7C
// #define GL_R8                             0x8229


#define makeGLFunction(returnType, name, ...) \
		typedef returnType name##Function(__VA_ARGS__); \
		name##Function* gl##name; 
#define loadGLFunction(name) \
		gl##name = (name##Function*)wglGetProcAddress("gl" #name);

#define GL_FUNCTION_LIST \
		GLOP(void, CreateVertexArrays, GLsizei n, GLuint *arrays) \
		GLOP(void, BindVertexArray, GLuint array) \
		GLOP(void, CreateBuffers, GLsizei n, GLuint *buffers) \
		GLOP(void, CreateTextures, GLenum target, GLsizei n, GLuint *textures) \
		GLOP(void, NamedBufferStorage, GLuint buffer, GLsizei size, const void *data, GLbitfield flags) \
		GLOP(void*, MapNamedBufferRange, GLuint buffer, GLint* offset, GLsizei length, GLbitfield access) \
		GLOP(void, TextureBuffer, GLuint texture, GLenum internalformat, GLuint buffer) \
		GLOP(uint, CreateShaderProgramv, GLenum type, GLsizei count, const char **strings) \
		GLOP(void, CreateProgramPipelines, GLsizei n, GLuint *pipelines) \
		GLOP(void, UseProgramStages, GLuint pipeline, GLbitfield stages, GLuint program) \
		GLOP(void, BindTextures, GLuint first, GLsizei count, const GLuint *textures) \
		GLOP(void, BindProgramPipeline, GLuint pipeline) \
		GLOP(void, DrawArraysInstancedBaseInstance, GLenum mode, GLint first, GLsizei count, GLsizei primcount, GLuint baseinstance) \
		GLOP(GLuint, GetDebugMessageLog, GLuint counter‹, GLsizei bufSize, GLenum *source, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, char *messageLog) \
		GLOP(void, TextureStorage3D, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth) \
		GLOP(void, TextureSubImage3D, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels) \
		GLOP(void, GenerateTextureMipmap, GLuint texture) \
		GLOP(void, TextureStorage2D, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) \
		GLOP(void, TextureSubImage2D, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) \
		GLOP(void, CreateSamplers, GLsizei n, GLuint *samplers) \
		GLOP(void, SamplerParameteri, GLuint sampler, GLenum pname, GLint param) \
		GLOP(void, BindSamplers, GLuint first, GLsizei count, const GLuint *samplers) \
		GLOP(void, TextureParameteri, GLuint texture, GLenum pname, GLint param) \
		GLOP(void, ProgramUniform4f, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2,GLfloat v3) \
		GLOP(GLint, GetUniformLocation, GLuint program, const GLchar *name) \
		GLOP(void, ProgramUniform3f, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2) \
		GLOP(void, ProgramUniform2f, GLuint program, GLint location, GLfloat v0, GLfloat v1) \
		GLOP(void, ProgramUniform1f, GLuint program, GLint location, GLfloat v0) \
		GLOP(void, ProgramUniform1fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
		GLOP(void, ProgramUniformMatrix3fv, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) \
		GLOP(void, ProgramUniformMatrix4fv, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) 



typedef HGLRC wglCreateContextAttribsARBFunction(HDC hDC, HGLRC hshareContext, const int *attribList);
wglCreateContextAttribsARBFunction* wglCreateContextAttribsARB;


#define GLOP(returnType, name, ...) makeGLFunction(returnType, name, __VA_ARGS__)
GL_FUNCTION_LIST
#undef GLOP


void loadFunctions() {
	#define GLOP(returnType, name, ...) loadGLFunction(name)
	GL_FUNCTION_LIST
	#undef GLOP
}

#define GLSL(src) "#version 330\n \
	#extension GL_ARB_gpu_shader5               : enable\n \
	#extension GL_ARB_gpu_shader_fp64           : enable\n \
	#extension GL_ARB_shader_precision          : enable\n \
	#extension GL_ARB_conservative_depth        : enable\n \
	#extension GL_ARB_texture_cube_map_array    : enable\n \
	#extension GL_ARB_separate_shader_objects   : enable\n \
	#extension GL_ARB_shading_language_420pack  : enable\n \
	#extension GL_ARB_shading_language_packing  : enable\n \
	#extension GL_ARB_explicit_uniform_location : enable\n" #src

// const char* vertexShaderSimple = GLSL (
// 	out gl_PerVertex { vec4 gl_Position; };
// 	layout(binding = 0) uniform samplerBuffer s_pos;
	
// 	void main() 
// 	{
// 		vec3 pos = texelFetch(s_pos, gl_VertexID).xyz;
// 		gl_Position = vec4(pos, 1.f);
// 	}	
// );

// const char* fragmentShaderSimple = GLSL (
// 	layout(depth_less) out float gl_FragDepth;

// 	out vec4 color;

// 	void main() {
// 		color = vec4(1,0,0,1);
// 	}
// );





const char* vertexShaderCube = GLSL (
	// const vec3 cube[] = vec3[] (
	//     vec3(-0.5, -0.5,  0.5), // 1  left    First Strip
	//     vec3(-0.5,  0.5,  0.5), // 3
	//     vec3(-0.5, -0.5, -0.5), // 0
	//     vec3(-0.5,  0.5, -0.5), // 2
	//     vec3( 0.5, -0.5, -0.5), // 4  back
	//     vec3( 0.5,  0.5, -0.5), // 6
	//     vec3( 0.5, -0.5,  0.5), // 5  right
	//     vec3( 0.5,  0.5,  0.5), // 7
	//     vec3( 0.5,  0.5, -0.5), // 6  top     Second Strip
	//     vec3(-0.5,  0.5, -0.5), // 2
	//     vec3( 0.5,  0.5,  0.5), // 7
	//     vec3(-0.5,  0.5,  0.5), // 3
	//     vec3( 0.5, -0.5,  0.5), // 5  front
	//     vec3(-0.5, -0.5,  0.5), // 1
	//     vec3( 0.5, -0.5, -0.5), // 4  bottom
	//     vec3(-0.5, -0.5, -0.5)  // 0
	// );

	const vec3 cube[] = vec3[] (
	    vec3(-0.5f,-0.5f,-0.5f), // triangle 1 : begin
	    vec3(-0.5f,-0.5f, 0.5f),
	    vec3(-0.5f, 0.5f, 0.5f), // triangle 1 : end
	    vec3(0.5f, 0.5f,-0.5f), // triangle 2 : begin
	    vec3(-0.5f,-0.5f,-0.5f),
	    vec3(-0.5f, 0.5f,-0.5f), // triangle 2 : end
	    vec3(0.5f,-0.5f, 0.5f),
	    vec3(-0.5f,-0.5f,-0.5f),
	    vec3(0.5f,-0.5f,-0.5f),
	    vec3(0.5f, 0.5f,-0.5f),
	    vec3(0.5f,-0.5f,-0.5f),
	    vec3(-0.5f,-0.5f,-0.5f),
	    vec3(-0.5f,-0.5f,-0.5f),
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3(-0.5f, 0.5f,-0.5f),
	    vec3(0.5f,-0.5f, 0.5f),
	    vec3(-0.5f,-0.5f, 0.5f),
	    vec3(-0.5f,-0.5f,-0.5f),
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3(-0.5f,-0.5f, 0.5f),
	    vec3(0.5f,-0.5f, 0.5f),
	    vec3(0.5f, 0.5f, 0.5f),
	    vec3(0.5f,-0.5f,-0.5f),
	    vec3(0.5f, 0.5f,-0.5f),
	    vec3(0.5f,-0.5f,-0.5f),
	    vec3(0.5f, 0.5f, 0.5f),
	    vec3(0.5f,-0.5f, 0.5f),
	    vec3(0.5f, 0.5f, 0.5f),
	    vec3(0.5f, 0.5f,-0.5f),
	    vec3(-0.5f, 0.5f,-0.5f),
	    vec3(0.5f, 0.5f, 0.5f),
	    vec3(-0.5f, 0.5f,-0.5f),
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3(0.5f, 0.5f, 0.5f),
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3(0.5f,-0.5f, 0.5f)	
	);

	uniform mat4x4 model;
	uniform mat4x4 view;
	uniform mat4x4 proj;

	out gl_PerVertex { vec4 gl_Position; };
	out vec4 Color;

	void main() {
		float c = gl_VertexID;
		// Color = vec4(1/(c/36),1/(c/7),1/(c/2),1);
		Color = vec4(c/36,c/36,c/36,1);
	
		vec4 pos = vec4(cube[gl_VertexID], 1);
	
		gl_Position = proj*view*model*pos;
	}
);

const char* fragmentShaderCube = GLSL (
	layout(binding = 0) uniform sampler2D s;

	// smooth in vec2 uv;
	in vec4 Color;

	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	void main() {
		// color = texture(s, uv) * Color;
		color = Color;
	}
);





const char* vertexShaderQuad = GLSL (
	const vec2 quad[] = vec2[] (
	  vec2( -0.5f, -0.5f ),
	  vec2(  0.5f, -0.5f ),
	  vec2( -0.5f,  0.5f ),
	  vec2(  0.5f,  0.5f )
	);

	const ivec2 quad_uv[] = ivec2[] (
	  ivec2(  0.0,  0.0 ),
	  ivec2(  1.0,  0.0 ),
	  ivec2(  0.0,  1.0 ),
	  ivec2(  1.0,  1.0 )
	);

	uniform vec4 setUV;
	uniform vec4 mod;
	uniform vec4 setColor;
	uniform vec4 camera; // left bottom right top

	out gl_PerVertex { vec4 gl_Position; };
	smooth out vec2 uv;
	out vec4 Color;

	void main() {
		ivec2 pos = quad_uv[gl_VertexID];
		uv = vec2(setUV[pos.x], setUV[2 + pos.y]);
		Color = setColor;
		vec2 model = quad[gl_VertexID]*mod.zw + mod.xy;
		vec2 view = model/(camera.zw*0.5f) - camera.xy/(camera.zw*0.5f);
		gl_Position = vec4(view, 0, 1);
	}
);

const char* fragmentShaderQuad = GLSL (
	layout(binding = 0) uniform sampler2D s;

	smooth in vec2 uv;
	in vec4 Color;

	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	void main() {
		color = texture(s, uv) * Color;
	}
);

struct PipelineIds {
	uint programQuad;

	uint quadVertex;
	uint quadFragment;

	uint quadVertexMod;
	uint quadVertexUV;
	uint quadVertexColor;
	uint quadVertexCamera;


	uint programCube;

	uint cubeVertex;
	uint cubeFragment;

	// uint cubeVertexScale;
	// uint cubeVertexRot;
	// uint cubeVertexTrans;
	uint cubeVertexModel;
	uint cubeVertexView;
	uint cubeVertexProj;
};

void drawRect(PipelineIds ids, Rect r, Rect uv, Vec4 color, int texture) {
	uint uniformLocation;
	Rect cd = rectGetCenDim(r);
	glProgramUniform4f(1, ids.quadVertexMod, cd.min.x, cd.min.y, cd.max.x, cd.max.y);
	glProgramUniform4f(1, ids.quadVertexUV, uv.min.x, uv.max.x, uv.max.y, uv.min.y);
	glProgramUniform4f(1, ids.quadVertexColor, color.r, color.g, color.b, color.a);

	glBindTexture(GL_TEXTURE_2D, texture);
	glDrawArraysInstancedBaseInstance(GL_TRIANGLE_STRIP, 0, 4, 1, 0);
}

struct Mesh {
	char* ptr;

	int bytes;
	int count;
	uint format;
	uint bufferId;
	uint id;
};

uint loadTexture(unsigned char* buffer, int w, int h, int mipLevels, int internalFormat, int channelType, int channelFormat) {
	uint textureId;
	glCreateTextures(GL_TEXTURE_2D, 1, &textureId);
	glTextureStorage2D(textureId, mipLevels, internalFormat, w, h);
	glTextureSubImage2D(textureId, 0, 0, 0, w, h, channelType, channelFormat, buffer);
	glGenerateTextureMipmap(textureId);

	return textureId;
}

uint loadTextureFile(char* path, int mipLevels, int internalFormat, int channelType, int channelFormat) {
	int x,y,n;
	unsigned char* stbData = stbi_load(path, &x, &y, &n, 0);

	int result = loadTexture(stbData, x, y, mipLevels, internalFormat, channelType, channelFormat);
	stbi_image_free(stbData);	

	return result;
}

uint createShader(const char* vertexShaderString, const char* fragmentShaderString, uint* vId, uint* fId) {
	*vId = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &vertexShaderString);
	*fId = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &fragmentShaderString);

	uint shaderId;
	glCreateProgramPipelines(1, &shaderId);
	glUseProgramStages(shaderId, GL_VERTEX_SHADER_BIT, *vId);
	GLenum glError = glGetError(); printf("GLError: %i\n", glError);
	glUseProgramStages(shaderId, GL_FRAGMENT_SHADER_BIT, *fId);
	glError = glGetError(); printf("GLError: %i\n", glError);

	return shaderId;
}

void ortho(PipelineIds* ids, Rect r) {
	r = rectGetCenDim(r);
	glProgramUniform4f(1, ids->quadVertexCamera, r.cen.x, r.cen.y, r.dim.w, r.dim.h);
}

struct Font {
	char* fileBuffer;
	Vec2i size;
	int glyphStart, glyphCount;
	stbtt_bakedchar* cData;
	uint texId;
	int height;
};

void drawText(PipelineIds* ids, Vec2 pos, char* text, Font* font, int vAlign, int hAlign) {
	int length = strLen(text);
	Vec2 textDim = stbtt_GetTextDim(font->cData, font->height, font->glyphStart, text);
	pos.x -= vAlign*0.5f*textDim.w;
	pos.y -= hAlign*0.5f*textDim.h;

	Vec2 startPos = pos;
	for(int i = 0; i < length; i++) {
		char t = text[i];

		if(t == '\n') {
			pos.y -= font->height;
			pos.x = startPos.x;
			continue;
		}

		stbtt_aligned_quad q;
		stbtt_GetBakedQuad(font->cData, font->size.w, font->size.h, t-font->glyphStart, &pos.x, &pos.y, &q, 1);
		drawRect(*ids, rect(q.x0, q.y0, q.x1, q.y1), rect(q.s0,q.t0,q.s1,q.t1), vec4(1,1,1,1), 3);
	}
}

struct Texture {
	int id;
	int width;
	int height;
	int channels;
	int levels;
};

struct AppData {
	SystemData systemData;
	Input input;

	Mesh mesh;
	PipelineIds pipelineIds;
	uint programs[16];
	uint textures[16];
	int texCount;
	uint samplers[16];

	WindowSettings wSettings;
	Vec3 camera;
	Vec3 camPos;
	Vec3 camLook;
	Vec3 camUp;
	Vec3 camRight;
	float aspectRatio;

	Font fontArial;
};




MemoryBlock* globalMemory;

extern "C" APPMAINFUNCTION(appMain) {
	globalMemory = memoryBlock;
	AppData* ad = (AppData*)memoryBlock->permanent;
	Input* input = &ad->input;
	SystemData* systemData = &ad->systemData;
	HWND windowHandle = systemData->windowHandle;
	WindowSettings* wSettings = &ad->wSettings;

	if(init) {
		getPMemory(sizeof(AppData));
		*ad = {};

		initInput(&ad->input);
		
		wSettings->res.w = 1920;
		wSettings->res.h = 1080;
		wSettings->fullscreen = false;
		wSettings->fullRes.x = GetSystemMetrics(SM_CXSCREEN);
		wSettings->fullRes.y = GetSystemMetrics(SM_CYSCREEN);
		wSettings->style = (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU  | WS_MINIMIZEBOX  | WS_VISIBLE);

		uint wStyle = wSettings->style;
		// initSystem(systemData, windowsData, WS_VISIBLE, 0,0,1920,1080);
		// initSystem(systemData, windowsData, ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZE | WS_MAXIMIZE | WS_SYSMENU), 0,0,1920,1080);
		initSystem(systemData, windowsData, wStyle, -1900, 5, wSettings->res.w, wSettings->res.h);

		DEVMODE devMode;
		int index = 0;
		int dW = 0, dH = 0;
		Vec2i resolutions[90] = {};
		int resolutionCount = 0;
		while(bool result = EnumDisplaySettings(0, index, &devMode)) {
			Vec2i nRes = vec2i(devMode.dmPelsWidth, devMode.dmPelsHeight);
			if(resolutionCount == 0 || resolutions[resolutionCount-1] != nRes) {
				resolutions[resolutionCount++] = nRes;
			}
			index++;
		}

		loadFunctions();

		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		// glEnable(GL_FRAMEBUFFER_SRGB);
		// glEnable(GL_DEPTH_TEST);
		// glDepthRange(-1.0, 1.0);
		glEnable(GL_CULL_FACE);
		glEnable(GL_BLEND);
		// glBlendFunc(0x0302, 0x0303);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		
		uint vao = 0;
		glCreateVertexArrays(1, &vao);
		glBindVertexArray(vao);

		ad->textures[ad->texCount++] = loadTextureFile("..\\data\\white.png", 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		ad->textures[ad->texCount++] = loadTextureFile("..\\data\\rect.png", 2, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

		// glTextureParameteri(textureId, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		// glTextureParameteri(textureId, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// // glTextureParameteri(textureId, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
		// // glTextureParameteri(textureId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		// glTextureParameteri(textureId, GL_TEXTURE_WRAP_S, GL_REPEAT);
		// glTextureParameteri(textureId, GL_TEXTURE_WRAP_T, GL_REPEAT);

		PipelineIds* ids = &ad->pipelineIds;
		ids->programQuad = createShader(vertexShaderQuad, fragmentShaderQuad, &ids->quadVertex, &ids->quadFragment);
		ids->quadVertexMod = glGetUniformLocation(ids->quadVertex, "mod");
		ids->quadVertexUV = glGetUniformLocation(ids->quadVertex, "setUV");
		ids->quadVertexColor = glGetUniformLocation(ids->quadVertex, "setColor");
		ids->quadVertexCamera = glGetUniformLocation(ids->quadVertex, "camera");
		ad->programs[0] = ids->programQuad;

		ids->programCube = createShader(vertexShaderCube, fragmentShaderCube, &ids->cubeVertex, &ids->cubeFragment);

		// ids->cubeVertexScale = glGetUniformLocation(ids->cubeVertex, "scale");
		// ids->cubeVertexRot = glGetUniformLocation(ids->cubeVertex, "rot");
		// ids->cubeVertexTrans = glGetUniformLocation(ids->cubeVertex, "trans");
		ids->cubeVertexModel = glGetUniformLocation(ids->cubeVertex, "model");
		ids->cubeVertexView = glGetUniformLocation(ids->cubeVertex, "view");
		ids->cubeVertexProj = glGetUniformLocation(ids->cubeVertex, "proj");

		ad->programs[1] = ids->programCube;

		// GLenum glError = glGetError(); printf("GLError: %i\n", glError);

		ad->camera = vec3(0,0,10);

		ad->camPos = vec3(0,0,1);
		ad->camLook = vec3(0,0,1);
		ad->camUp = vec3(0,1,0);
		ad->camRight;




		Font font;
		char* path = "..\\data\\LiberationMono.ttf";
		font.fileBuffer = (char*)getPMemory(fileSize(path) + 1);
		readFileToBuffer(font.fileBuffer, path);
		font.size = vec2i(512,512);
		unsigned char* fontBitmapBuffer = (unsigned char*)getTMemory(font.size.x*font.size.y);
		unsigned char* fontBitmap = (unsigned char*)getTMemory(font.size.x*font.size.y*4);
		
		font.height = 30;
		font.glyphStart = 32;
		font.glyphCount = 95;
		font.cData = (stbtt_bakedchar*)getPMemory(sizeof(stbtt_bakedchar)*font.glyphCount);
		stbtt_BakeFontBitmap((unsigned char*)font.fileBuffer, 0, font.height, fontBitmapBuffer, font.size.w, font.size.h, font.glyphStart, font.glyphCount, font.cData);
		for(int i = 0; i < font.size.w*font.size.h; i++) {
			fontBitmap[i*4] = fontBitmapBuffer[i];
			fontBitmap[i*4+1] = fontBitmapBuffer[i];
			fontBitmap[i*4+2] = fontBitmapBuffer[i];
			fontBitmap[i*4+3] = fontBitmapBuffer[i];
		}
		uint texId = loadTexture(fontBitmap, font.size.w, font.size.h, 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		ad->textures[ad->texCount++] = texId;
		font.texId = texId;
		ad->fontArial = font;

	}

	if(reload) {
		loadFunctions();
	}

	static int secondFrame = 0;
	secondFrame++;
	if(secondFrame == 2) {
		setWindowMode(windowHandle, wSettings, WINDOW_MODE_FULLBORDERLESS);
	}

	updateInput(&ad->input, isRunning, windowHandle);
	getWindowProperties(windowHandle, &wSettings->currentRes.x, &wSettings->currentRes.y,0,0,0,0);
	ad->aspectRatio = wSettings->currentRes.x / (float)wSettings->currentRes.y;

	Vec3* cam = &ad->camera;	
	if(input->mouseButtonDown[0]) {
		cam->x += input->mouseDeltaX*(cam->z/wSettings->currentRes.w);
		cam->y -= input->mouseDeltaY*((cam->z/wSettings->currentRes.h)/ad->aspectRatio);
	}

	if(input->mouseWheel) {
		float zoom = cam->z;
		zoom -= input->mouseWheel/(float)1;
		cam->z = zoom;
	}

	if(input->keysPressed[VK_F1]) {
		int mode;
		if(wSettings->fullscreen) mode = WINDOW_MODE_WINDOWED;
		else mode = WINDOW_MODE_FULLBORDERLESS;
		setWindowMode(windowHandle, wSettings, mode);
	}

	#define VK_W 0x57
	#define VK_A 0x41
	#define VK_S 0x53
	#define VK_D 0x44

	ad->camRight = cross(ad->camUp, ad->camLook);

	if( input->keysDown[VK_W] || input->keysDown[VK_A] || input->keysDown[VK_S] || 
		input->keysDown[VK_D] || input->keysDown[VK_SHIFT] || input->keysDown[VK_CONTROL]) {

		Vec3 dir;
		if(input->keysDown[VK_W]) 			dir = -normVec3(ad->camLook);
		else if(input->keysDown[VK_A]) 		dir = -normVec3(ad->camRight);
		else if(input->keysDown[VK_S]) 		dir = normVec3(ad->camLook);
		else if(input->keysDown[VK_D]) 		dir = normVec3(ad->camRight);
		else if(input->keysDown[VK_SHIFT]) 	dir = normVec3(ad->camUp);
		else if(input->keysDown[VK_CONTROL])dir = -normVec3(ad->camUp);

		float speed = 0.1f;
		ad->camPos += dir*speed;
	}

	if(input->mouseButtonDown[1]) {
		Vec3 l = ad->camLook;
		float dt = 0.005f * input->mouseDeltaX;
		ad->camLook = vec3(l.x*cos(dt)+l.z*sin(dt), l.y, l.x*-sin(dt)+l.z*cos(dt));
	}

	


	glViewport(0,0, wSettings->currentRes.x, wSettings->currentRes.y);
	glClearColor(0.3f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		
	glBindProgramPipeline(ad->programs[1]);
	// glDepthRange(-1.0,1.0);

	glEnable(GL_DEPTH_TEST);
	// drawRect(ad->pipelineIds, rectCenDim(2,2,3,3), rect(0,0,1,1), vec4(1,1,0.2f,1), ad->textures[0]);

	// float c[9] = {0,1,0, 0,0,-1, 1,0,0};
	// glProgramUniform1fv(ad->pipelineIds.cubeVertex, ad->pipelineIds.cubeVertexCam, 9, c);


	// glProgramUniform3f(ad->pipelineIds.cubeVertex, ad->pipelineIds.cubeVertexTrans, 0.2f,0.2f,0);
	// glProgramUniform3f(ad->pipelineIds.cubeVertex, ad->pipelineIds.cubeVertexTrans, 0.2f,0.2f,0);

	// glProgramUniform1fv(ad->pipelineIds.cubeVertex, ad->pipelineIds.cubeVertexProj, 6, proj);
// GLOP(void, ProgramUniformMatrix4x3fv, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) 

	Mat4 view;
	viewMatrix(&view, ad->camPos, ad->camLook, ad->camUp);
	Mat4 proj;
	projMatrix(&proj, degreeToRadian(45), ad->aspectRatio, 0.1f, 20);
	glProgramUniformMatrix4fv(ad->pipelineIds.cubeVertex, ad->pipelineIds.cubeVertexView, 1, 1, view.e);
	glProgramUniformMatrix4fv(ad->pipelineIds.cubeVertex, ad->pipelineIds.cubeVertexProj, 1, 1, proj.e);

	static float dt = 0;
	dt += 0.01f;

	Mat4 sm, rm, tm, model;
	scaleMatrix(&sm, vec3(5,2,1));

	Quat total = quat();
	Quat xRot = quat(dt, vec3(1,0,0));
	Quat yRot = quat(dt, vec3(0,1,0));
	Quat zRot = quat(dt, vec3(0,0,1));
	quatRotationMatrix(&rm, zRot*yRot*xRot*total);

	translationMatrix(&tm, vec3(0,0,-7));
	model = tm*rm*sm;

	glProgramUniformMatrix4fv(ad->pipelineIds.cubeVertex, ad->pipelineIds.cubeVertexModel, 1, 1, model.e);

	// glProgramUniformMatrix4fv(ad->pipelineIds.cubeVertex, ad->pipelineIds.cubeVertexModel, 1, 0, matrix);

	// void drawRect(PipelineIds ids, Rect r, Rect uv, Vec4 color, int texture) {
	// uint uniformLocation;
	// Rect cd = rectGetCenDim(r);
	// glProgramUniform4f(1, ids.quadVertexMod, cd.min.x, cd.min.y, cd.max.x, cd.max.y);
	// glProgramUniform4f(1, ids.quadVertexUV, uv.min.x, uv.max.x, uv.max.y, uv.min.y);
	// glProgramUniform4f(1, ids.quadVertexColor, color.r, color.g, color.b, color.a);
	// glBindTexture(GL_TEXTURE_2D, texture);
	// glDrawArraysInstancedBaseInstance(GL_TRIANGLE_STRIP, 0, 8, 1, 0);
	glDrawArraysInstancedBaseInstance(GL_TRIANGLES, 0, 36, 1, 0);



	ortho(&ad->pipelineIds, rectCenDim(cam->x,cam->y, cam->z, cam->z/ad->aspectRatio));
	glBindProgramPipeline(ad->programs[0]);

	// drawRect(ad->pipelineIds, rectCenDim(0, 0, 0.01f, 100), rect(0,0,1,1), vec4(0.4f,1,0.4f,1), ad->textures[0]);
	// drawRect(ad->pipelineIds, rectCenDim(0, 0, 100, 0.01f), rect(0,0,1,1), vec4(0.4f,0.4f,1,1), ad->textures[0]);

	ortho(&ad->pipelineIds, rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0));

	drawText(&ad->pipelineIds, vec2(0,0), "Text_123...", &ad->fontArial, 0, 2);

	swapBuffers(&ad->systemData);
	glFinish();

	clearTMemory();
}
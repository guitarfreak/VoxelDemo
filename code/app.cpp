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
		GLOP(void, ProgramUniform2f, GLuint program, GLint location, GLfloat v0, GLfloat v1) 

		

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

		// vec2 wh = vec2(camera.z - camera.x, camera.w - camera.y);
		// vec2 xy = vec2(camera.x + wh.x/2.0f, camera.y + wh.y/2.0f);
		// vec2 view = model/(camera.zw*0.5f) + 1/camera.xy;
		// vec2 view = model/(camera.zw*0.5f) + xy/camera.zw;
		// vec2 endXY = camera.xy;
		// if(camera.xy != vec2(0,0)) endXY = 1/endXY;
		// vec2 view = model/(camera.zw*0.5f) + endXY;
		vec2 view = model/(camera.zw*0.5f) - camera.xy/(camera.zw*0.5f);
		// vec2 view = model/(camera.zw*0.5f) + camera.xy;
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
	uint vertexShaderId = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &vertexShaderString);
	uint fragmentShaderId = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &fragmentShaderString);

	uint shaderId;
	glCreateProgramPipelines(1, &shaderId);
	glUseProgramStages(shaderId, GL_VERTEX_SHADER_BIT, vertexShaderId);
	glUseProgramStages(shaderId, GL_FRAGMENT_SHADER_BIT, fragmentShaderId);

	return shaderId;
}

void ortho(PipelineIds* ids, Rect r) {
	r = rectGetCenDim(r);
	glProgramUniform4f(1, ids->quadVertexCamera, r.cen.x, r.cen.y, r.dim.w, r.dim.h);
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
	uint samplers[16];

	WindowSettings wSettings;
	Vec3 camera;
	float aspectRatio;

	stbtt_bakedchar cdata[96]; // ASCII 32..126 is 95 glyphs
};




MemoryBlock* globalMemory;

extern "C" APPMAINFUNCTION(appMain) {
	globalMemory = memoryBlock;
	AppData* appData = (AppData*)memoryBlock->permanent;
	Input* input = &appData->input;
	SystemData* systemData = &appData->systemData;
	HWND windowHandle = systemData->windowHandle;
	WindowSettings* wSettings = &appData->wSettings;

	if(init) {
		getPMemory(sizeof(AppData));
		*appData = {};

		initInput(&appData->input);
		
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

		appData->textures[0] = loadTextureFile("..\\data\\white.png", 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		appData->textures[1] = loadTextureFile("..\\data\\rect.png", 2, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

		// glTextureParameteri(textureId, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		// glTextureParameteri(textureId, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// // glTextureParameteri(textureId, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
		// // glTextureParameteri(textureId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		// glTextureParameteri(textureId, GL_TEXTURE_WRAP_S, GL_REPEAT);
		// glTextureParameteri(textureId, GL_TEXTURE_WRAP_T, GL_REPEAT);

		PipelineIds* ids = &appData->pipelineIds;
		ids->programQuad = createShader(vertexShaderQuad, fragmentShaderQuad, &ids->quadVertex, &ids->quadFragment);
		ids->quadVertexMod = glGetUniformLocation(1, "mod");
		ids->quadVertexUV = glGetUniformLocation(1, "setUV");
		ids->quadVertexColor = glGetUniformLocation(1, "setColor");
		ids->quadVertexCamera = glGetUniformLocation(1, "camera");

		appData->programs[0] = ids->programQuad;

		GLenum glError = glGetError(); printf("GLError: %i\n", glError);

		appData->camera = vec3(0,0,10);

		// LiberationMono.ttf


		// STBTT_DEF int stbtt_BakeFontBitmap(const unsigned char *data, int offset,  // font location (use offset=0 for plain .ttf)
		//                                 float pixel_height,                     // height of font in pixels
		//                                 unsigned char *pixels, int pw, int ph,  // bitmap to be filled in
		//                                 int first_char, int num_chars,          // characters to bake
		//                                 stbtt_bakedchar *chardata);             // you allocate this, it's num_chars long


		char* path = "..\\data\\arial.ttf";
		char* ttfBuffer = (char*)getTMemory(fileSize(path) + 1);
		readFileToBuffer(ttfBuffer, path);
		int fw = 512, fh = 512;
		unsigned char* fontBitmapBuffer = (unsigned char*)getTMemory(fw*fh);
		unsigned char* fontBitmap = (unsigned char*)getPMemory(fw*fh*4);
		
		stbtt_BakeFontBitmap((unsigned char*)ttfBuffer, 0, 70, fontBitmapBuffer, fw, fh, 32,96, appData->cdata);
		for(int i = 0; i < fw*fh; i++) {
			fontBitmap[i*4] = fontBitmapBuffer[i];
			fontBitmap[i*4+1] = fontBitmapBuffer[i];
			fontBitmap[i*4+2] = fontBitmapBuffer[i];
			fontBitmap[i*4+3] = fontBitmapBuffer[i];
		}
		// appData->textures[2] = loadTexture(fontBitmap, fw, fh, 1, GL_R8, GL_RED, GL_UNSIGNED_BYTE);
		appData->textures[2] = loadTexture(fontBitmap, fw, fh, 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

	}

	if(reload) {
		loadFunctions();
	}

	static int secondFrame = 0;
	secondFrame++;
	if(secondFrame == 2) {
		setWindowMode(windowHandle, wSettings, WINDOW_MODE_FULLBORDERLESS);
	}

	updateInput(&appData->input, isRunning, windowHandle);
	getWindowProperties(windowHandle, &wSettings->currentRes.x, &wSettings->currentRes.y,0,0,0,0);
	appData->aspectRatio = wSettings->currentRes.x / (float)wSettings->currentRes.y;

	if(input->mouseButtonDown[0]) {
		appData->camera.x += input->mouseDeltaX/(float)1000;
		appData->camera.y -= (input->mouseDeltaY/(float)1000) * appData->aspectRatio;
	}

	if(input->mouseWheel) {
		float zoom = appData->camera.z;
		zoom -= input->mouseWheel/(float)1;
		appData->camera.z = zoom;
	}

	if(input->keysPressed[VK_F1]) {
		int mode;
		if(wSettings->fullscreen) mode = WINDOW_MODE_WINDOWED;
		else mode = WINDOW_MODE_FULLBORDERLESS;
		setWindowMode(windowHandle, wSettings, mode);
	}

	Vec3 cam = appData->camera;
	ortho(&appData->pipelineIds, rectCenDim(cam.x,cam.y, cam.z, cam.z/appData->aspectRatio));

	glViewport(0,0, wSettings->currentRes.x, wSettings->currentRes.y);
	glClearColor(0.3f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glBindProgramPipeline(appData->programs[0]);


	// for(int i = 0; i < 10; i++) {
	// 	for(int j = 0; j < 10; j++) {
	// 		drawRect(appData->pipelineIds, vec2(i,j), vec2(0.4f,0.4f), rect(0,1,0,1), vec4(0.1f*i,0.1f*j,1,1), 1);
	// 	}
	// }

	// drawRect(appData->pipelineIds, vec2(0,0), vec2(5,5), rect(0,1,0,1), vec4(1,1,1,1), appData->textures[2]);
	// drawRect(appData->pipelineIds, vec2(0,0), vec2(2,2), rect(0,1,0,1), vec4(1,1,1,1), 2);


	// char* text = "ABCDEFG";
	// int length = strLen(text);
	// float x = 0; float y = 0;
	// for(int i = 0; i < length; i++) {
	// 	char t = text[i];
	// 	stbtt_aligned_quad q;
	// 	stbtt_GetBakedQuad(appData->cdata, 512, 512, t-32, &x, &y, &q, 1);

	// 	Rect r = rectGetCenDim(rect(q.x0, q.y0, q.x1, q.y1));
	// 	drawRect(appData->pipelineIds, r.cen, r.dim, rect(q.s0,q.t0,q.s1,q.t1), vec4(1,1,1,1), appData->textures[2]);
	// }

	// glProgramUniform4f(1, appData->pipelineIds.quadVertexCamera, 0, 0, wSettings->currentRes.w, wSettings->currentRes.h);
	drawRect(appData->pipelineIds, rectCenDim(0, 0, 2, 2), rect(0,0,1,1), vec4(1,1,1,1), appData->textures[2]);

	// glProgramUniform4f(1, appData->pipelineIds.quadVertexCamera, wSettings->currentRes.w/2, wSettings->currentRes.h/2, wSettings->currentRes.w, wSettings->currentRes.w/appData->aspectRatio);
	// glProgramUniform4f(1, appData->pipelineIds.quadVertexCamera, 0, -wSettings->currentRes.h, wSettings->currentRes.w, 0);

	// glProgramUniform4f(1, appData->pipelineIds.quadVertexCamera, 0, 0, wSettings->currentRes.w, wSettings->currentRes.h);
	// ortho(&appData->pipelineIds, rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0));
	// ortho(&appData->pipelineIds, rect(-wSettings->currentRes.w/2, -wSettings->currentRes.h/2, wSettings->currentRes.w/2, wSettings->currentRes.h/2));
	// ortho(&appData->pipelineIds, rect(-wSettings->currentRes.w/2, -wSettings->currentRes.h/2, wSettings->currentRes.w/2, wSettings->currentRes.h/2));
	// ortho(&appData->pipelineIds, rectCenDim(5,5,10,10));
	// ortho(&appData->pipelineIds, rect(0,0,10,10));
	ortho(&appData->pipelineIds, rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0));

	// ortho(&appData->pipelineIds, rectCenDim(-5,-5,10,10));
	// ortho(&appData->pipelineIds, rect(0,0,100,100));
	// ortho(&appData->pipelineIds, rect(vec2(-4,-4),vec2(6,6)));

	// drawRect(appData->pipelineIds, rectCenDim(-25, -25, 5,5), rect(0,0,1,1), vec4(1,1,1,1), appData->textures[0]);
	drawRect(appData->pipelineIds, rectCenDim(100,-100,10,10), rect(0,0,1,1), vec4(0,1,1,1), appData->textures[0]);



	char* text = "fr#T#$tsdGw3zW\n34zwesrg3423$3$4$$%#4\n23432rweQQQWE";
	int length = strLen(text);
	float x = 100; float y = 100;
	float xStart = x;
	for(int i = 0; i < length; i++) {
		char t = text[i];

		if(t == '\n') {
			y += 70;
			x = xStart;
			continue;
		}

		stbtt_aligned_quad q;
		stbtt_GetBakedQuad(appData->cdata, 512, 512, t-32, &x, &y, &q, 1);
		drawRect(appData->pipelineIds, rect(q.x0, -q.y1, q.x1, -q.y0), rect(q.s0,q.t0,q.s1,q.t1), vec4(1,1,1,1), appData->textures[2]);
	}

	// drawRect(appData->pipelineIds, vec2(0,0), vec2(500,500), rect(0.1f,0.1f,0.9f,0.9f), vec4(1,1,1,1), appData->textures[2]);
	// drawRect(appData->pipelineIds, vec2(0,0), vec2(500,500), rect(0,0,1,1), vec4(1,1,1,1), appData->textures[2]);

	// s0 0.55 s1 0.632 t0 0.107 t1 0.181

	swapBuffers(&appData->systemData);
	glFinish();
}
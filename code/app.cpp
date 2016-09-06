#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <gl\gl.h>
#include "glext.h"

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

#define STB_VOXEL_RENDER_IMPLEMENTATION
// #define STBVOX_CONFIG_LIGHTING_SIMPLE
#define STBVOX_CONFIG_FOG_SMOOTHSTEP

// #define STBVOX_CONFIG_MODE 0
#define STBVOX_CONFIG_MODE 1
#include "stb_voxel_render.h"



//-----------------------------------------
//				WHAT TO DO
//-----------------------------------------
/*
- Joysticks, Keyboard, Mouse, Xinput-DirectInput
- Sound
- Data Package - Streamingw
- Expand Font functionality
- Gui
- create simpler windows.h
- remove c runtime library, implement sin, cos...
- savestates, hotrealoading
- pre and post functions to Main
- memory dynamic expansion
- 32x32 voxel chunks
- ballistic motion on jumping
- round collision

- advance timestep when window not in focus (stop game when not in focus)

- reduce number of thread queue pushes
- insert and init mesh not thread proof, make sure every mesh is initialized before generating

- make voxel drop in tutorial code for stb_voxel

- hotload still gets stuck sometimes, thread that won't complete

- change bubblesort to mergesort\radixsort
- implement sun and clouds that block beams of light

- setup proper blocktype enums and properties

//-------------------------------------
//               BUGS
//-------------------------------------
- look has to be negative to work in view projection matrix
- app stops reacting to input after a while, hotloading still works though
- deleting a block at the edge of a mesh means remaking mesh on the other side 
  of the edge too
- distance jumping collision bug, possibly precision loss in distances

- no alpha in voxel rendering, see glass block

- gpu fucks up at some point making swapBuffers alternates between time steps 
  which makes the game stutter, restart is required
- game input gets stuck when buttons are pressed right at the start
- sort key assert firing randomly
*/

/*
	- 32x32 gen chunks

 	- blocks with different textures on each side
	- mak rect shader take 2d arrays to draw the voxel textures
	- trees

	- in general, try to use every feature of stb_voxel at least once to see what it feels like
*/

MemoryBlock* globalMemory;
ThreadQueue* globalThreadQueue;

struct GraphicsState;
struct DrawCommandList;
GraphicsState* globalGraphicsState;
DrawCommandList* globalCommandList3d;
DrawCommandList* globalCommandList2d;

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
		GLOP(void, ProgramUniform1i, GLuint program, GLint location, GLuint v0) \
		GLOP(void, ProgramUniform1fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
		GLOP(void, ProgramUniform2fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
		GLOP(void, ProgramUniform1iv, GLuint program, GLint location, GLsizei count, const GLint *value) \
		GLOP(void, ProgramUniformMatrix3fv, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) \
		GLOP(void, ProgramUniformMatrix4fv, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) \
		GLOP(void, UseProgram, GLuint program) \
		GLOP(void, Uniform1i, GLint location, GLint v0) \
		GLOP(void, Uniform1iv, GLint location, GLsizei count, const GLint *value) \
		GLOP(void, Uniform1fv, GLint location, GLsizei count, const GLfloat *value) \
		GLOP(void, Uniform2fv, GLint location, GLsizei count, const GLfloat *value) \
		GLOP(void, Uniform3fv, GLint location, GLsizei count, const GLfloat *value) \
		GLOP(void, Uniform4fv, GLint location, GLsizei count, const GLfloat *value) \
		GLOP(void, ProgramUniform4fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
		GLOP(void, ActiveTexture, GLenum texture) \
		GLOP(void, EnableVertexAttribArray, GLuint index) \
		GLOP(void, TexImage3DEXT,  GLenum	target, GLint	level, GLenum	internalformat, GLsizei	width, GLsizei	height, GLsizei	depth, GLint	border, GLenum	format, GLenum	type, const	GLvoid *pixels) \
		GLOP(void, ActiveTextureARB, GLenum texture) \
		GLOP(void, VertexAttribIPointer, GLuint index, GLint size, GLenum type, GLsizei stride, const GLvoid * pointer) \
		GLOP(void, BindBuffer, GLenum target, GLuint buffer) \
		GLOP(void, BindBufferARB, GLenum target, GLuint buffer) \
		GLOP(void, GenBuffersARB, GLsizei n, GLuint * buffers) \
		GLOP(void, NamedBufferData, GLuint buffer, GLsizeiptr size, const GLvoid * data, GLenum usage) \
		GLOP(void, BufferDataARB, GLenum target, GLsizeiptr size, const GLvoid * data, GLenum usage) \
		GLOP(void, TexBufferARB, GLenum target, GLenum internalFormat, GLuint buffer) \
		GLOP(void, UniformMatrix4fv, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) \
		GLOP(void, VertexAttribPointer, GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid * pointer) \
		GLOP(void, ProgramUniform3fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
		GLOP(GLint, GetAttribLocation, GLuint program, const GLchar *name) \
		GLOP(void, GenSamplers, GLsizei n​, GLuint *samplers​) \
		GLOP(void, BindSampler, GLuint unit, GLuint sampler​) \
		GLOP(void, BindTextureUnit, GLuint unit, GLuint texture) \
		GLOP(void, NamedBufferSubData, GLuint buffer, GLintptr offset, GLsizei size, const void *data) \
		GLOP(void, GetUniformiv, GLuint program, GLint location, GLint * params) \
		GLOP(void, CreateFramebuffers, GLsizei n, GLuint *framebuffers) \
		GLOP(void, NamedFramebufferParameteri, GLuint framebuffer, GLenum pname, GLint param) \
		GLOP(void, NamedRenderbufferStorageMultisample, GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height) \
		GLOP(void, CreateRenderbuffers, GLsizei n, GLuint *renderbuffers) \
		GLOP(void, NamedFramebufferRenderbuffer, GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) \
		GLOP(void, BindFramebuffer, GLenum target, GLuint framebuffer) \
		GLOP(void, NamedFramebufferDrawBuffer, GLuint framebuffer, GLenum buf) \
		GLOP(void, BlitFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) \
		GLOP(void, NamedFramebufferTexture, GLuint framebuffer, GLenum attachment, GLuint texture, GLint level) \
		GLOP(void, TextureStorage2DMultisample, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations) \
		GLOP(void, TexImage2DMultisample, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations) \
		GLOP(void, NamedRenderbufferStorage, GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height) \
		GLOP(GLenum, CheckNamedFramebufferStatus, GLuint framebuffer, GLenum target) \
		GLOP(void, GenFramebuffers, GLsizei n, GLuint *ids) \
		GLOP(void, FramebufferTexture, GLenum target, GLenum attachment, GLuint texture, GLint level) \
		GLOP(void, BlendFuncSeparate, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) \
		GLOP(void, BlendEquation, GLenum mode)





// typedef HGLRC wglCreateContextAttribsARBFunction(HDC hDC, HGLRC hshareContext, const int *attribList);
// wglCreateContextAttribsARBFunction* wglCreateContextAttribsARB;

typedef int wglGetSwapIntervalEXTFunction(void);
wglGetSwapIntervalEXTFunction* wglGetSwapIntervalEXT;
typedef int wglSwapIntervalEXTFunction(int);
wglSwapIntervalEXTFunction* wglSwapIntervalEXT;


#define GLOP(returnType, name, ...) makeGLFunction(returnType, name, __VA_ARGS__)
GL_FUNCTION_LIST
#undef GLOP


void loadFunctions() {
	#define GLOP(returnType, name, ...) loadGLFunction(name)
	GL_FUNCTION_LIST
	#undef GLOP
}

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
	uint cubeVertexModel;
	uint cubeVertexView;
	uint cubeVertexProj;
	uint cubeVertexColor;
	uint cubeVertexMode;
	uint cubeVertexVertices;

	uint programVoxel;
	uint voxelVertex;
	uint voxelFragment;
};

struct GraphicsState {
	PipelineIds pipelineIds;

	// // Mesh mesh;

	// uint programs[16];
	// uint textures[16];
	// int texCount;
	// uint samplers[16];

	// uint frameBuffers[16];
	// uint renderBuffers[16];
	// uint frameBufferTextures[2];

	// float aspectRatio;
	// float fieldOfView;
	// float nearPlane;
	// float farPlane;

	// // Font fontArial;

	// Vec2i curRes;
	// int msaaSamples;
	// Vec2i fboRes;
	// bool useNativeRes;

	// // VoxelData;

	// uint voxelShader;
	// uint voxelVertex;
	// uint voxelFragment;

	// GLuint voxelSamplers[3];
	// GLuint voxelTextures[3];

	GLuint textureUnits[16];
	GLuint samplerUnits[16];
};

enum DrawListCommand {
	Draw_Command_Viewport_Type,
	Draw_Command_Clear_Type,
	Draw_Command_Enable_Type,
	Draw_Command_Disable_Type,
	Draw_Command_FrontFace_Type,
	Draw_Command_BindFramebuffer_Type,
	Draw_Command_BindBuffer_Type,
	Draw_Command_BlendFunc,
	// Draw_Command_DepthFunc,
	// Draw_Command_ClearDepth,
	Draw_Command_ClearColor,
	// Draw_Command_GetUniformLocation,
	Draw_Command_ProgramUniform,
	// Draw_Command_glEnableVertexAttribArray,
	// Draw_Command_glVertexAttribIPointer,
	// Draw_Command_glBindTextures,
	// Draw_Command_glBindSamplers,
	// Draw_Command_glBindProgramPipeline,
	// Draw_Command_glDrawArrays,
	// Draw_Command_glError,
	// Draw_Command_,
	// Draw_Command_,
	// Draw_Command_,
	// Draw_Command_,
	// Draw_Command_,
	// Draw_Command_,
	// Draw_Command_,
	// Draw_Command_,

	Draw_Command_Cube_Type,
	Draw_Command_Line_Type,
	Draw_Command_Quad_Type,
	Draw_Command_Rect_Type,
	Draw_Command_Text_Type,
	Draw_Command_PolygonMode_Type,
	Draw_Command_LineWidth_Type,
	Draw_Command_Cull_Type,
	// Draw_Command_Rect_Type,
};

struct DrawCommandList {
	void* data;
	int count;
	int bytes;
	int maxBytes;
};

void drawCommandListInit(DrawCommandList* cl, char* data, int maxBytes) {
	cl->data = data;
	cl->count = 0;
	cl->bytes = 0;
	cl->maxBytes = maxBytes;
}

struct Draw_Command_Cube {
	Vec3 trans;
	Vec3 scale;
	Vec4 color;
	float degrees;
	Vec3 rot;
};

struct Draw_Command_Line {
	Vec3 p0, p1;
	Vec4 color;
};

struct Draw_Command_Quad {
	Vec3 p0, p1, p2, p3;
	Vec4 color;
};

struct Draw_Command_Rect {
	Rect r, uv;
	Vec4 color;
	int texture;
};

struct Font;
struct Draw_Command_Text {
	char* text;
	Font* font;
	Vec2 pos;
	Vec4 color;

	int vAlign;
	int hAlign;
	int shadow;
	Vec4 shadowColor;
};

enum Polygon_Mode {
	POLYGON_MODE_FILL = 0,
	POLYGON_MODE_LINE,
	POLYGON_MODE_POINT,
};

struct Draw_Command_PolygonMode {
	int mode;
};

struct Draw_Command_LineWidth {
	int width;
};

struct Draw_Command_Cull {
	bool b;
};




#define makeDrawCommandFunction(name, list) \
	void dc##name(Draw_Command_##name d, DrawCommandList* commandList = (DrawCommandList*)list) { \
		if(list == 0) commandList = globalCommandList3d; \
		else commandList = globalCommandList2d; \
		int* cl = (int*)(((char*)commandList->data)+commandList->bytes); \
		*(cl++) = Draw_Command_##name##_Type; \
		assert(sizeof(Draw_Command_##name) + commandList->bytes < commandList->maxBytes); \
		*((Draw_Command_##name*)(cl)) = d; \
		commandList->count++; \
		commandList->bytes += sizeof(Draw_Command_##name) + sizeof(int); \
	} 

makeDrawCommandFunction(Cube, 0);
makeDrawCommandFunction(Line, 0);
makeDrawCommandFunction(Quad, 0);
makeDrawCommandFunction(PolygonMode, 0);
makeDrawCommandFunction(LineWidth, 0);
makeDrawCommandFunction(Cull, 0);
makeDrawCommandFunction(Rect, 1);
makeDrawCommandFunction(Text, 1);

#define dcCase(name, var, index) \
	Draw_Command_##name var = *((Draw_Command_##name*)index); \
	index += sizeof(Draw_Command_##name); \








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
	const vec3 cube[] = vec3[] (
	    vec3(-0.5f,-0.5f,-0.5f), 
	    vec3( 0.5f,-0.5f,-0.5f), 
	    vec3( 0.5f, 0.5f,-0.5f),
	    vec3(-0.5f, 0.5f,-0.5f),
	    vec3(-0.5f,-0.5f, 0.5f), 
	    vec3(-0.5f, 0.5f, 0.5f), 
	    vec3( 0.5f, 0.5f, 0.5f),
	    vec3( 0.5f,-0.5f, 0.5f),
	    vec3(-0.5f, 0.5f,-0.5f), 
	    vec3( 0.5f, 0.5f,-0.5f), 
	    vec3( 0.5f, 0.5f, 0.5f),
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3(-0.5f,-0.5f,-0.5f), 
	    vec3(-0.5f,-0.5f, 0.5f), 
	    vec3( 0.5f,-0.5f, 0.5f),
	    vec3( 0.5f,-0.5f,-0.5f),
	    vec3( 0.5f,-0.5f,-0.5f), 
	    vec3( 0.5f,-0.5f, 0.5f), 
	    vec3( 0.5f, 0.5f, 0.5f),
	    vec3( 0.5f, 0.5f,-0.5f),
	    vec3(-0.5f,-0.5f,-0.5f), 
	    vec3(-0.5f, 0.5f,-0.5f), 
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3(-0.5f,-0.5f, 0.5f)
	);

	out gl_PerVertex { vec4 gl_Position; };
	out vec4 Color;

	uniform mat4x4 model;
	uniform mat4x4 view;
	uniform mat4x4 proj;
	// uniform mat4x4 projViewModel;

	uniform vec3 vertices[4];
	uniform bool mode;

	uniform vec4 setColor;

	void main() {
		Color = setColor;
	
		vec4 pos;
		if(mode == true) {
			pos = vec4(vertices[gl_VertexID], 1);
			gl_Position = proj*view*pos;
		} else {
			pos = vec4(cube[gl_VertexID], 1);
			gl_Position = proj*view*model*pos;
		}
	
		// gl_Position = proj*view*model*pos;
		// gl_Position = projViewModel*pos;
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


// const char* testFragmentShader = GLSL (
// 	// layout(binding = 0) uniform sampler2D s;

// 	// smooth in vec2 uv;
// 	// in vec4 Color;

// 	layout(depth_less) out float gl_FragDepth;
// 	out vec4 color;

// 	void main() {
// 		// color = texture(s, uv) * Color;
// 		color = vec4(1,0,0,1);
// 	}
// );





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
	// uniform sampler2DArray s[2];
	// uniform sampler2D s;

	smooth in vec2 uv;
	in vec4 Color;

	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	void main() {
		color = texture(s, uv) * Color;
	}
);


// const char* vertexShaderPrimitive = GLSL (
// 	out gl_PerVertex { vec4 gl_Position; };
// 	out vec4 Color;

// 	uniform mat4x4 view;
// 	uniform mat4x4 proj;
// 	uniform vec4 setColor;

// 	uniform vec3 vertices[4];

// 	void main() {
// 		Color = setColor;

// 		vec4 pos = vec4(vertices[gl_VertexID], 1);
// 		gl_Position = proj*view*pos;
// 	}
// );

// const char* fragmentShaderPrimitive = GLSL (
// 	in vec4 Color;
// 	layout(depth_less) out float gl_FragDepth;
// 	out vec4 color;

// 	void main() {
// 		color = Color;
// 	}
// );


void updateAfterWindowResizing(WindowSettings* wSettings, float* ar, uint fb0, uint fb1, uint rb0, uint rb1, uint* fbt, int msaaSamples, Vec2i* fboRes, Vec2i* curRes, bool useNativeRes) {
	*ar = wSettings->aspectRatio;
	
	fboRes->x = fboRes->y*(*ar);

	if(useNativeRes) *curRes = wSettings->currentRes;
	else *curRes = *fboRes;

	Vec2i s = *curRes;

	glNamedRenderbufferStorageMultisample(rb0, msaaSamples, GL_RGBA8, s.w, s.h);
	glNamedRenderbufferStorageMultisample(rb1, msaaSamples, GL_DEPTH_COMPONENT, s.w, s.h);

	glDeleteTextures(1, fbt);
	glCreateTextures(GL_TEXTURE_2D, 1, fbt);
	glTextureStorage2D(*fbt, 1, GL_RGBA8, s.w, s.h);

	glNamedFramebufferRenderbuffer(fb0, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb0);
	glNamedFramebufferRenderbuffer(fb0, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb1);
	glNamedFramebufferTexture(fb1, GL_COLOR_ATTACHMENT0, *fbt, 0);
}


void drawRect(Rect r, Rect uv, Vec4 color, int texture) {
	uint uniformLocation;
	Rect cd = rectGetCenDim(r);
	glProgramUniform4f(globalGraphicsState->pipelineIds.quadVertex, globalGraphicsState->pipelineIds.quadVertexMod, cd.min.x, cd.min.y, cd.max.x, cd.max.y);
	glProgramUniform4f(globalGraphicsState->pipelineIds.quadVertex, globalGraphicsState->pipelineIds.quadVertexUV, uv.min.x, uv.max.x, uv.max.y, uv.min.y);
	glProgramUniform4f(globalGraphicsState->pipelineIds.quadVertex, globalGraphicsState->pipelineIds.quadVertexColor, color.r, color.g, color.b, color.a);
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
	// GLenum glError = glGetError(); printf("GLError: %i\n", glError);
	glUseProgramStages(shaderId, GL_FRAGMENT_SHADER_BIT, *fId);
	// glError = glGetError(); printf("GLError: %i\n", glError);

	return shaderId;
}

void ortho(Rect r) {
	r = rectGetCenDim(r);
	glProgramUniform4f(1, globalGraphicsState->pipelineIds.quadVertexCamera, r.cen.x, r.cen.y, r.dim.w, r.dim.h);
}

void lookAt(Vec3 pos, Vec3 look, Vec3 up, Vec3 right) {
	Mat4 view;
	viewMatrix(&view, pos, look, up, right);

	glProgramUniformMatrix4fv(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexView, 1, 1, view.e);
}

void perspective(float fov, float aspect, float n, float f) {
	Mat4 proj;
	projMatrix(&proj, fov, aspect, n, f);

	glProgramUniformMatrix4fv(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexProj, 1, 1, proj.e);
}

struct Font {
	char* fileBuffer;
	Vec2i size;
	int glyphStart, glyphCount;
	stbtt_bakedchar* cData;
	uint texId;
	int height;
};

void drawText(char* text, Font* font, Vec2 pos, Vec4 color, int vAlign = 0, int hAlign = 0, int shadow = 0, Vec4 shadowColor = vec4(0,0,0,1)) {
	int length = strLen(text);
	Vec2 textDim = stbtt_GetTextDim(font->cData, font->height, font->glyphStart, text);
	pos.x -= vAlign*0.5f*textDim.w;
	pos.y -= hAlign*0.5f*textDim.h;

	Vec2 shadowOffset = vec2(shadow, -shadow);

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

		Rect r = rect(q.x0, q.y0, q.x1, q.y1);
		if(shadow > 0) {
			drawRect(rectAddOffset(r, shadowOffset), rect(q.s0,q.t0,q.s1,q.t1), shadowColor, 3);
		}
		drawRect(r, rect(q.s0,q.t0,q.s1,q.t1), color, 3);
	}
}

void drawCube(Vec3 trans, Vec3 scale, Vec4 color, float degrees, Vec3 rot) {
	Mat4 sm; scaleMatrix(&sm, scale);
	Mat4 rm; quatRotationMatrix(&rm, quat(degrees, rot));
	Mat4 tm; translationMatrix(&tm, trans);
	Mat4 model = tm*rm*sm;

	glProgramUniformMatrix4fv(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexModel, 1, 1, model.e);
	glProgramUniform4f(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexColor, color.r, color.g, color.b, color.a);

	glProgramUniform1i(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexMode, false);

	// glDrawArraysInstancedBaseInstance(GL_TRIANGLES, 0, 36, 1, 0);
	glDrawArrays(GL_QUADS, 0, 6*4);
}

void drawLine(Vec3 p0, Vec3 p1, Vec4 color) {
	Vec3 verts[4] = {p0, p1};
	glProgramUniform3fv(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexVertices, 4, verts[0].e);
	glProgramUniform4f(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexColor, color.r, color.g, color.b, color.a);
	glProgramUniform1i(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexMode, true);

	glDrawArrays(GL_LINES, 0, 2);
}

void drawQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color) {
	Vec3 verts[4] = {p0, p1, p2, p3};
	glProgramUniform3fv(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexVertices, 4, verts[0].e);
	glProgramUniform4f(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexColor, color.r, color.g, color.b, color.a);
	glProgramUniform1i(globalGraphicsState->pipelineIds.cubeVertex, globalGraphicsState->pipelineIds.cubeVertexMode, true);

	glDrawArrays(GL_QUADS, 0, 4);
}

void drawQuad(Vec3 p, Vec3 normal, float size, Vec4 color) {
	Vec3 base = p;

	int sAxis[2];
	int biggestAxis = getBiggestAxis(normal, sAxis);

	float s2 = size*0.5f;

	Vec3 verts[4] = {};
	for(int i = 0; i < 4; i++) {
		Vec3 d = base;
			 if(i == 0) { d.e[sAxis[0]] += -s2; d.e[sAxis[1]] += -s2; }
		else if(i == 1) { d.e[sAxis[0]] += -s2; d.e[sAxis[1]] +=  s2; }
		else if(i == 2) { d.e[sAxis[0]] +=  s2; d.e[sAxis[1]] +=  s2; }
		else if(i == 3) { d.e[sAxis[0]] +=  s2; d.e[sAxis[1]] += -s2; }
		verts[i] = d;
	}

	drawQuad(verts[0], verts[1], verts[2], verts[3], color);
}

uint createSampler(int wrapS, int wrapT, int magF, int minF) {
	uint result;
	glCreateSamplers(1, &result);

	glSamplerParameteri(result, GL_TEXTURE_MAX_ANISOTROPY_EXT, 4.0f);
	glSamplerParameteri(result, GL_TEXTURE_WRAP_S, wrapS);
	glSamplerParameteri(result, GL_TEXTURE_WRAP_T, wrapT);
	glSamplerParameteri(result, GL_TEXTURE_MAG_FILTER, magF);
	glSamplerParameteri(result, GL_TEXTURE_MIN_FILTER, minF);

	return result;
}



struct Texture {
	int id;
	int width;
	int height;
	int channels;
	int levels;
};




#include <stdio.h>

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



enum BlockTypes {
	BT_None = 0,
	BT_Water,
	BT_Sand,
	BT_Grass,
	BT_Stone,
	BT_Snow,
	// BT_Trunk,
	// BT_Leaves,
	// BT_Water,
	// BT_Water,
	// BT_Water,

	BT_Size,
};

enum BlockTextures {
	BX_None = 0,
	BX_Water,
	BX_Sand,
	BX_GrassTop, BX_GrassSide, BX_GrassBottom,
	BX_Stone,
	BX_Snow,

	BX_Size,
};

const char* textureFilePaths[BX_Size] = {
	"..\\data\\minecraft textures\\none.png",
	"..\\data\\minecraft textures\\water.png",
	"..\\data\\minecraft textures\\sand.png",
	"..\\data\\minecraft textures\\grass_top.png",
	"..\\data\\minecraft textures\\grass_side.png",
	"..\\data\\minecraft textures\\grass_bottom.png",
	"..\\data\\minecraft textures\\stone.png",
	"..\\data\\minecraft textures\\snow.png",
};

#define allTexSame(t) t,t,t,t,t,t

uchar texture2[BT_Size] = {0,1,1,1,1,1};
uchar texture1Faces[BT_Size][6] = {
	{0,0,0,0,0,0},
	{allTexSame(BX_Water)},
	{allTexSame(BX_Sand)},
	{BX_GrassSide, BX_GrassSide, BX_GrassSide, BX_GrassSide, BX_GrassTop, BX_GrassBottom},
	{allTexSame(BX_Stone)},
	{allTexSame(BX_Snow)},
};
uchar geometry[BT_Size] = {
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_empty,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_transp,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
	STBVOX_MAKE_GEOMETRY(STBVOX_GEOM_solid,0,0),
};
uchar meshSelection[BT_Size] = {0,1,0,0,0,0};

// #define VIEW_DISTANCE 4096 // 64
// #define VIEW_DISTANCE 3072 // 32

// #define VIEW_DISTANCE 2500 // 32
// #define VIEW_DISTANCE 2048 // 32
// #define VIEW_DISTANCE 1024 // 16
// #define VIEW_DISTANCE 512  // 8
#define VIEW_DISTANCE 256 // 4
// #define VIEW_DISTANCE 128 // 2

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

void initVoxelMesh(VoxelMesh* m, Vec2i coord) {
	*m = {};
	m->coord = coord;

	if(USE_MALLOC) {
		m->voxels = (uchar*)malloc(VOXEL_SIZE);
		m->lighting = (uchar*)malloc(VOXEL_SIZE);
	} else {
		// m->meshBufferCapacity = kiloBytes(150);
		m->meshBufferCapacity = kiloBytes(200);
		m->meshBuffer = (char*)getPMemory(m->meshBufferCapacity);
		m->texBufferCapacity = m->meshBufferCapacity/4;
		m->texBuffer = (char*)getPMemory(m->texBufferCapacity);

		m->meshBufferTransCapacity = kiloBytes(200);
		m->meshBufferTrans = (char*)getPMemory(m->meshBufferTransCapacity);
		m->texBufferTransCapacity = m->meshBufferTransCapacity/4;
		m->texBufferTrans = (char*)getPMemory(m->texBufferTransCapacity);

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

struct VoxelNode {
	VoxelMesh mesh;
	VoxelNode* next;
};

VoxelMesh* getVoxelMesh(VoxelNode** voxelHash, int voxelHashSize, Vec2i coord) {
	int hashIndex = mod(coord.x*9 + coord.y*23, voxelHashSize);

	VoxelMesh* m = 0;
	VoxelNode* node = voxelHash[hashIndex];
	while(node->next) {
		if(node->mesh.coord == coord) {
			m = &node->mesh;
			break;
		}

		node = node->next;
	}

	if(!m) {
		m = &node->mesh;
		initVoxelMesh(m, coord);

		node->next = (VoxelNode*)getPMemory(sizeof(VoxelNode));
		*node->next = {};
	}

	return m;
}

// int startX = 37800;
// int startY = 48000;

int startX = 37750;
int startY = 47850;

int startXMod = 58000;
int startYMod = 68000;

void s(void* data) {
	VoxelMesh* m = (VoxelMesh*)data;
	Vec2i coord = m->coord;

	if(!m->generated) {
		Vec3i min = vec3i(0,0,0);
		Vec3i max = vec3i(VOXEL_X,VOXEL_Y,VOXEL_Z);

	    for(int y = min.y; y < max.y; y++) {
	    	for(int x = min.x; x < max.x; x++) {
	    		int gx = (coord.x*VOXEL_X)+x;
	    		int gy = (coord.y*VOXEL_Y)+y;

	    		// float perlin = perlin2d(gx+4000, gy+4000, 0.015f, 4);
	    		// float height = perlin*50;
	    		// float height = 5;

	    		// static int startX = randomInt(0, 10000);
	    		// static int startY = randomInt(0, 10000);

	    		// static int startXMod = randomInt(0,10000);
	    		// static int startYMod = randomInt(0,10000);

	    		// float height = perlin2d(gx+4000+startX, gy+4000+startY, 0.004f, 7);w
	    		float height = perlin2d(gx+4000+startX, gy+4000+startY, 0.004f, 6);
	    		height -= 0.1f; 

	    		// float mod = perlin2d(gx+startXMod, gy+startYMod, 0.008f, 4);
	    		float mod = perlin2d(gx+startXMod, gy+startYMod, 0.02f, 4);
	    		float modOffset = 0.1f;
	    		mod = mapRange(mod, 0, 1, -modOffset, modOffset);

	    		int blockType;
	    		// 	 if(height < 0.35f) blockType = 10; // water
	    		// else if(height < 0.4f + mod) blockType = 11; // sand
	    		if(height < 0.4f + mod) blockType = BT_Sand; // sand
	    		else if(height < 0.6f + mod) blockType = BT_Grass; // grass
	    		else if(height < 0.8f + mod) blockType = BT_Stone; // stone
	    		else if(height <= 1.0f + mod) blockType = BT_Snow; // snow

	    		float heightPercent = height;
	    		height = clamp(height, 0, 1);
	    		// height = pow(height,3.5f);
	    		height = pow(height,4.0f);
	    		height = mapRange(height, 0, 1, 20, 200);

	    		for(int z = 0; z < height; z++) {
	    			m->voxels[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = blockType;
	    			m->lighting[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = 0;
	    		}

	    		for(int z = height; z < VOXEL_Z; z++) {
	    			m->voxels[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = 0;
	    			m->lighting[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = 255;
	    		}

	    		int waterLevel = 22;
	    		if(height < waterLevel) {
	    			for(int z = height; z < waterLevel; z++) {
	    				m->voxels[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = BT_Water;

	    				int lightValue;
	    				if(z == 20)  lightValue = 50;
	    				else if(z == 21) lightValue = 150;
	    				m->lighting[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = lightValue;
	    			}
	    		}
	    	}
	    }
	}

	atomicSub(&m->activeGeneration);
	m->generated = true;
}

struct MakeMeshThreadedData {
	VoxelMesh* m;
	VoxelNode** voxelHash;
	int voxelHashSize;

	int inProgress;
};

void makeMeshThreaded(void* data) {
	MakeMeshThreadedData* d = (MakeMeshThreadedData*)data;
	VoxelMesh* m = d->m;
	// VoxelMesh* vms = d->vms;
	// int* vmsSize = d->vmsSize;

	VoxelNode** voxelHash = d->voxelHash;
	int voxelHashSize = d->voxelHashSize;

	int cacheId = getThreadQueueId(globalThreadQueue);

	// gather voxel data in radius and copy to cache
	Vec2i coord = m->coord;
	for(int y = -1; y < 2; y++) {
		for(int x = -1; x < 2; x++) {
			Vec2i c = coord + vec2i(x,y);
			VoxelMesh* lm = getVoxelMesh(voxelHash, voxelHashSize, c);

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

	unsigned char tLerp[50] = {};
	// for(int i = 1; i < arrayCount(tLerp)-1; i++) tLerp[i] = 6;
	// tLerp[10] = 4;
	inputDesc->block_texlerp = tLerp;



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



	atomicSub(&m->activeMaking);
	m->upToDate = true;
	d->inProgress = false;
}

MakeMeshThreadedData* threadData;

void makeMesh(VoxelMesh* m, VoxelNode** voxelHash, int voxelHashSize) {
	// int threadJobsMax = 20;

	bool notAllMeshsAreReady = false;
	Vec2i coord = m->coord;
	for(int y = -1; y < 2; y++) {
		for(int x = -1; x < 2; x++) {
			Vec2i c = coord + vec2i(x,y);
			VoxelMesh* lm = getVoxelMesh(voxelHash, voxelHashSize, c);

			if(!lm->generated) {
				if(!lm->activeGeneration && !threadQueueFull(globalThreadQueue)) {
				// if(!lm->activeGeneration && threadQueueOpenJobs(globalThreadQueue) < threadJobsMax) {
					atomicAdd(&lm->activeGeneration);
					threadQueueAdd(globalThreadQueue, generateVoxelMeshThreaded, lm);
				}
				notAllMeshsAreReady = true;
			} 
		}
	}

	if(notAllMeshsAreReady) return;

	if(!m->upToDate) {
		if(!m->activeMaking) {
			if(threadQueueFull(globalThreadQueue)) return;
			// if(threadQueueOpenJobs(globalThreadQueue) < threadJobsMax) return;

			MakeMeshThreadedData* data;
			for(int i = 0; i < 256; i++) {
				if(!threadData[i].inProgress) {
					threadData[i] = {m, voxelHash, voxelHashSize, true};
					// data = threadData + i;
					atomicAdd(&m->activeMaking);
					threadQueueAdd(globalThreadQueue, makeMeshThreaded, &threadData[i]);

					break;
				}
			}
		}

		return;
	} 

	glNamedBufferData(m->bufferId, m->bufferSizePerQuad*m->quadCount, m->meshBuffer, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, m->bufferId);

	if(STBVOX_CONFIG_MODE == 1) {
		glNamedBufferData(m->texBufferId, m->textureBufferSizePerQuad*m->quadCount, m->texBuffer, GL_STATIC_DRAW);
		glTextureBuffer(m->textureId, GL_RGBA8UI, m->texBufferId);
	}

	if(USE_MALLOC) {
		free(m->meshBuffer);
		free(m->texBuffer);
	}

	glNamedBufferData(m->bufferTransId, m->bufferSizePerQuad*m->quadCountTrans, m->meshBufferTrans, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, m->bufferTransId);

	if(STBVOX_CONFIG_MODE == 1) {
		glNamedBufferData(m->texBufferTransId, m->textureBufferSizePerQuad*m->quadCountTrans, m->texBufferTrans, GL_STATIC_DRAW);
		glTextureBuffer(m->textureTransId, GL_RGBA8UI, m->texBufferTransId);
	}

	if(USE_MALLOC) {
		free(m->meshBufferTrans);
		free(m->texBufferTrans);
	}

	m->meshUploaded = true;
}

Vec3i getVoxelCoordFromGlobalCoord(Vec3 globalCoord) {
	Vec3i result;
	if(globalCoord.x < 0) globalCoord.x -= 1;
	if(globalCoord.y < 0) globalCoord.y -= 1;
	result = vec3i(globalCoord);

	return result;
}

Vec3 getGlobalCoordFromVoxelCoord(Vec3i voxelCoord) {
	Vec3 result;
	// assumes voxel has length of 1
	result = vec3(voxelCoord) + vec3(0.5f, 0.5f, 0.5f);
	return result;
}

Vec2i getMeshCoordFromVoxelCoord(Vec3i voxelCoord) {
	Vec2i result;
	result = vec2i(floor(voxelCoord.x/(float)VOXEL_X), floor(voxelCoord.y/(float)VOXEL_Y));
	return result;
}

Vec2i getMeshCoordFromGlobalCoord(Vec3 globalCoord) {
	Vec2i result;
	Vec3i mc = getVoxelCoordFromGlobalCoord(globalCoord);
	result = getMeshCoordFromVoxelCoord(mc);
	return result;
}

Vec3i getLocalVoxelCoord(Vec3i voxelCoord) {
	Vec3i result = voxelCoord;
	result.x = mod(voxelCoord.x, VOXEL_X);
	result.y = mod(voxelCoord.y, VOXEL_Y);

	return result;
}

Vec3 getGlobalMeshCoord(Vec2i coord) {
	Vec3 result = vec3(coord.x*VOXEL_X + VOXEL_X*0.5f, coord.y*VOXEL_Y + VOXEL_Y*0.5f, VOXEL_Z*0.5f);
	return result;
}

uchar* getBlockFromVoxelCoord(VoxelNode** voxelHash, int voxelHashSize, Vec3i voxelCoord) {
	VoxelMesh* vm = getVoxelMesh(voxelHash, voxelHashSize, getMeshCoordFromVoxelCoord(voxelCoord));
	Vec3i localCoord = getLocalVoxelCoord(voxelCoord);
	uchar* block = &vm->voxels[voxelArray(localCoord.x, localCoord.y, localCoord.z)];

	return block;
}

uchar* getBlockFromGlobalCoord(VoxelNode** voxelHash, int voxelHashSize, Vec3 coord) {
	return getBlockFromVoxelCoord(voxelHash, voxelHashSize, getVoxelCoordFromGlobalCoord(coord));
}

uchar* getLightingFromVoxelCoord(VoxelNode** voxelHash, int voxelHashSize, Vec3i voxelCoord) {
	VoxelMesh* vm = getVoxelMesh(voxelHash, voxelHashSize, getMeshCoordFromVoxelCoord(voxelCoord));
	Vec3i localCoord = getLocalVoxelCoord(voxelCoord);
	uchar* block = &vm->lighting[voxelArray(localCoord.x, localCoord.y, localCoord.z)];

	return block;
}

uchar* getLightingFromGlobalCoord(VoxelNode** voxelHash, int voxelHashSize, Vec3 coord) {
	return getLightingFromVoxelCoord(voxelHash, voxelHashSize, getVoxelCoordFromGlobalCoord(coord));
}

// uchar* getLightingFromGlobalCoord(VoxelNode** voxelHash, int voxelHashSize, Vec3 coord) {
// 	VoxelMesh* vm = getVoxelMesh(voxelHash, voxelHashSize, getMeshCoordFromGlobalCoord(coord));
// 	Vec3i localCoord = getLocalVoxelCoord(getVoxelCoordFromGlobalCoord(coord));
// 	uchar* block = &vm->lighting[voxelArray(localCoord.x, localCoord.y, localCoord.z)];

// 	return block;
// }

Vec3 getBlockCenterFromGlobalCoord(Vec3 coord) {
	Vec3 result = getGlobalCoordFromVoxelCoord(getVoxelCoordFromGlobalCoord(coord));
	return result;
}

void setupVoxelUniforms(Vec4 camera, uint texUnit1, uint texUnit2, uint faceUnit, Mat4 view, Mat4 proj, Vec3 fogColor) {
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
	int loc = glGetUniformLocation(globalGraphicsState->pipelineIds.voxelFragment, "light_source");
	glProgramUniform3fv(globalGraphicsState->pipelineIds.voxelFragment, loc, 2, (GLfloat*)light);
	dcCube({light[0], vec3(3,3,3), vec4(lColor, 1), 0, vec3(0,0,0)});
	#endif

	// ambient direction is sky-colored upwards
	// "ambient" lighting is from above
	al.e2[0][0] =  0.3f;
	al.e2[0][1] = -0.5f;
	al.e2[0][2] =  0.9f;
	al.e2[0][3] = 0;

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

	for (int i=0; i < STBVOX_UNIFORM_count; ++i) {
		stbvox_uniform_info sui;
		if (stbvox_get_uniform_info(&sui, i)) {
			if(i == STBVOX_UNIFORM_transform) continue;

			for(int shaderStage = 0; shaderStage < 2; shaderStage++) {
				GLint location;
				GLuint program;
				if(shaderStage == 0) {
					location = glGetUniformLocation(globalGraphicsState->pipelineIds.voxelVertex, sui.name);
					program = globalGraphicsState->pipelineIds.voxelVertex;
				} else {
					location = glGetUniformLocation(globalGraphicsState->pipelineIds.voxelFragment, sui.name);
					program = globalGraphicsState->pipelineIds.voxelFragment;
				}

				if (location != -1) {
					int arrayLength = sui.array_length;
					void* data = sui.default_value;

					switch (i) {
						case STBVOX_UNIFORM_camera_pos: { // only needed for fog
						   		data = camera.e;
						   } break;

						case STBVOX_UNIFORM_tex_array: {
							data = texUnit;
						} break;

						case STBVOX_UNIFORM_face_data: {
							data = &faceUnit;
						} break;

						case STBVOX_UNIFORM_ambient: {
							data = ambientLighting.e;
						} break;

						case STBVOX_UNIFORM_color_table: // you might want to override this
						case STBVOX_UNIFORM_texscale:    // you may want to override this
						case STBVOX_UNIFORM_normals:     // you never want to override this
						case STBVOX_UNIFORM_texgen:      // you never want to override this
							break;
					}

					switch(sui.type) {
						case STBVOX_UNIFORM_TYPE_none: // glProgramUniformX(program, loc2, sui.array_length, sui.default_value); break;
						case STBVOX_UNIFORM_TYPE_sampler: glProgramUniform1iv(program, location, arrayLength, (GLint*)data); break;
						case STBVOX_UNIFORM_TYPE_vec2: glProgramUniform2fv(program, location, arrayLength, (GLfloat*)data); break;
						case STBVOX_UNIFORM_TYPE_vec3: glProgramUniform3fv(program, location, arrayLength, (GLfloat*)data); break;
						case STBVOX_UNIFORM_TYPE_vec4: glProgramUniform4fv(program, location, arrayLength, (GLfloat*)data); break;
					}
				}
			}
		}
	}

	// Mat4 finalMat = proj*view*model;
	Mat4 finalMat = proj*view;
	GLint modelViewUni = glGetUniformLocation(globalGraphicsState->pipelineIds.voxelVertex, "model_view");
	glProgramUniformMatrix4fv(globalGraphicsState->pipelineIds.voxelVertex, modelViewUni, 1, 1, finalMat.e);
}

void drawVoxelMesh(VoxelMesh* m) {
	if(STBVOX_CONFIG_MODE == 0) {
		// interleaved buffer - 2 uints in a row -> 8 bytes stride
		glBindBuffer(GL_ARRAY_BUFFER, m->bufferId);
		int vaLoc = glGetAttribLocation(globalGraphicsState->pipelineIds.voxelVertex, "attr_vertex");
		glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 8, (void*)0);
		glEnableVertexAttribArray(vaLoc);
		int fLoc = glGetAttribLocation(globalGraphicsState->pipelineIds.voxelVertex, "attr_face");
		glVertexAttribIPointer(fLoc, 4, GL_UNSIGNED_BYTE, 8, (void*)4);
		glEnableVertexAttribArray(fLoc);

	} else {
		glBindBuffer(GL_ARRAY_BUFFER, m->bufferId);
		int vaLoc = glGetAttribLocation(globalGraphicsState->pipelineIds.voxelVertex, "attr_vertex");
		glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 4, (void*)0);
		glEnableVertexAttribArray(vaLoc);
	}

	GLuint transformUniform1 = glGetUniformLocation(globalGraphicsState->pipelineIds.voxelVertex, "transform");
	glProgramUniform3fv(globalGraphicsState->pipelineIds.voxelVertex, transformUniform1, 3, m->transform[0]);
	GLuint transformUniform2 = glGetUniformLocation(globalGraphicsState->pipelineIds.voxelFragment, "transform");
	glProgramUniform3fv(globalGraphicsState->pipelineIds.voxelFragment, transformUniform2, 3, m->transform[0]);

	globalGraphicsState->textureUnits[2] = m->textureId;
	glBindTextures(0,16,globalGraphicsState->textureUnits);
	glBindSamplers(0,16,globalGraphicsState->samplerUnits);
	
	glBindProgramPipeline(globalGraphicsState->pipelineIds.programVoxel);

	glDrawArrays(GL_QUADS, 0, m->quadCount*4);




	if(STBVOX_CONFIG_MODE == 0) {
		// interleaved buffer - 2 uints in a row -> 8 bytes stride
		glBindBuffer(GL_ARRAY_BUFFER, m->bufferTransId);
		int vaLoc = glGetAttribLocation(globalGraphicsState->pipelineIds.voxelVertex, "attr_vertex");
		glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 8, (void*)0);
		glEnableVertexAttribArray(vaLoc);
		int fLoc = glGetAttribLocation(globalGraphicsState->pipelineIds.voxelVertex, "attr_face");
		glVertexAttribIPointer(fLoc, 4, GL_UNSIGNED_BYTE, 8, (void*)4);
		glEnableVertexAttribArray(fLoc);

	} else {
		glBindBuffer(GL_ARRAY_BUFFER, m->bufferTransId);
		int vaLoc = glGetAttribLocation(globalGraphicsState->pipelineIds.voxelVertex, "attr_vertex");
		glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 4, (void*)0);
		glEnableVertexAttribArray(vaLoc);
	}

	globalGraphicsState->textureUnits[2] = m->textureTransId;
	glBindTextures(0,16,globalGraphicsState->textureUnits);
	glBindSamplers(0,16,globalGraphicsState->samplerUnits);
	
	glDrawArrays(GL_QUADS, 0, m->quadCountTrans*4);
}

void getCamData(Vec3 look, Vec2 rot, Vec3 gUp, Vec3* cLook, Vec3* cRight, Vec3* cUp) {
	rotateVec3(&look, rot.x, gUp);
	rotateVec3(&look, rot.y, normVec3(cross(gUp, look)));
	*cUp = normVec3(cross(look, normVec3(cross(gUp, look))));
	*cRight = normVec3(cross(gUp, look));
	*cLook = -look;
}

struct AppData {
	SystemData systemData;
	Input input;
	WindowSettings wSettings;

	GraphicsState graphicsState;
	DrawCommandList commandList2d;
	DrawCommandList commandList3d;

	LONGLONG lastTimeStamp;
	float dt;

	Vec3 activeCamPos;
	Vec3 activeCamLook;
	Vec3 activeCamUp;
	Vec3 activeCamRight;

	Vec3 camera;
	Vec3 camPos;
	Vec3 camLook;
	Vec2 camRot;
	Vec3 camVel;
	Vec3 camAcc;

	uint programs[16];
	uint textures[16];
	int texCount;
	uint samplers[16];

	uint frameBuffers[16];
	uint renderBuffers[16];
	uint frameBufferTextures[2];

	float aspectRatio;
	float fieldOfView;
	float nearPlane;
	float farPlane;

	Font font;
	Font font2;

	Vec2i curRes;
	int msaaSamples;
	Vec2i fboRes;
	bool useNativeRes;

	Vec3 playerPos;
	Vec3 playerLook;
	Vec2 playerRot;
	Vec3 playerSize;
	Vec3 playerVel;
	Vec3 playerAcc;
	float playerCamZOffset;
	bool playerMode;
	bool pickMode;

	bool playerOnGround;
	int blockMenu[10];
	int blockMenuSelected;


	int selectionRadius;
	bool blockSelected;
	Vec3 selectedBlock;
	Vec3 selectedBlockFaceDir;

	VoxelNode* voxelHash[1024];
	int voxelHashSize;
	uchar* voxelCache[8];
	uchar* voxelLightingCache[8];

	MakeMeshThreadedData threadData[256];

	GLuint voxelSamplers[3];
	GLuint voxelTextures[3];
};

char* fillString(char* text, ...) {
	va_list vl;
	va_start(vl, text);

	int length = strLen(text);
	char* buffer = getTString(length+1);

	char valueBuffer[20] = {};

	int ti = 0;
	int bi = 0;
	while(true) {
		char t = text[ti];

		if(text[ti] == '%' && text[ti+1] == 'f') {
			float v = va_arg(vl, double);
			floatToStr(valueBuffer, v, 2);
			int sLen = strLen(valueBuffer);
			memCpy(buffer + bi, valueBuffer, sLen);

			ti += 2;
			bi += sLen;
			getTString(sLen);
		} else if(text[ti] == '%' && text[ti+1] == 'i') {
			int v = va_arg(vl, int);
			intToStr(valueBuffer, v);
			int sLen = strLen(valueBuffer);
			memCpy(buffer + bi, valueBuffer, sLen);

			ti += 2;
			bi += sLen;
			getTString(sLen);
		} if(text[ti] == '%' && text[ti+1] == '%') {
			buffer[bi++] = '%';
			ti += 2;
			getTString(1);
		} else {
			buffer[bi++] = text[ti++];
			getTString(1);

			if(buffer[bi-1] == '\0') break;
		}
	}

	return buffer;
}

void getPointsFromQuadAndNormal(Vec3 p, Vec3 normal, float size, Vec3 verts[4]) {
	int sAxis[2];
	int biggestAxis = getBiggestAxis(normal, sAxis);

	float s2 = size*0.5f;

	for(int i = 0; i < 4; i++) {
		Vec3 d = p;
			 if(i == 0) { d.e[sAxis[0]] += -s2; d.e[sAxis[1]] += -s2; }
		else if(i == 1) { d.e[sAxis[0]] += -s2; d.e[sAxis[1]] +=  s2; }
		else if(i == 2) { d.e[sAxis[0]] +=  s2; d.e[sAxis[1]] +=  s2; }
		else if(i == 3) { d.e[sAxis[0]] +=  s2; d.e[sAxis[1]] += -s2; }
		verts[i] = d;
	}
}

extern "C" APPMAINFUNCTION(appMain) {
	globalMemory = memoryBlock;
	AppData* ad = (AppData*)memoryBlock->permanent;
	Input* input = &ad->input;
	SystemData* systemData = &ad->systemData;
	HWND windowHandle = systemData->windowHandle;
	WindowSettings* wSettings = &ad->wSettings;

	// globalThreadQueue = &ad->highQueue;
	globalThreadQueue = threadQueue;

	globalGraphicsState = &ad->graphicsState;

	globalCommandList3d = &ad->commandList3d;
	globalCommandList2d = &ad->commandList2d;

	threadData = ad->threadData;

	// HWND window, UINT message, WPARAM wParam, LPARAM lParam
	// mainWindowCallBack(0,0,0,0);
	globalMainWindowCallBack = mainWindowCallBack;


	for(int i = 0; i < 8; i++) {
		voxelCache[i] = ad->voxelCache[i];
		voxelLightingCache[i] = ad->voxelLightingCache[i];
	}

	if(init) {
		getPMemory(sizeof(AppData));
		*ad = {};

		// threadInit(globalThreadQueue, 3);
		// threadInit(globalThreadQueue, 7);

		ad->dt = 1/(float)60;

		initInput(&ad->input);
		
		wSettings->res.w = 1920;
		wSettings->res.h = 1080;
		wSettings->fullscreen = false;
		wSettings->fullRes.x = GetSystemMetrics(SM_CXSCREEN);
		wSettings->fullRes.y = GetSystemMetrics(SM_CYSCREEN);
		wSettings->style = (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU  | WS_MINIMIZEBOX  | WS_VISIBLE);
		initSystem(systemData, windowsData, 0, 0,0,0,0);

		ad->fieldOfView = 55;
		ad->msaaSamples = 4;
		ad->fboRes = vec2i(0, 120);
		ad->useNativeRes = true;
		ad->nearPlane = 0.1f;
		// ad->farPlane = 2000;
		ad->farPlane = 3000;


		// DEVMODE devMode;
		// int index = 0;
		// int dW = 0, dH = 0;
		// Vec2i resolutions[90] = {};
		// int resolutionCount = 0;
		// while(bool result = EnumDisplaySettings(0, index, &devMode)) {
		// 	Vec2i nRes = vec2i(devMode.dmPelsWidth, devMode.dmPelsHeight);
		// 	if(resolutionCount == 0 || resolutions[resolutionCount-1] != nRes) {
		// 		resolutions[resolutionCount++] = nRes;
		// 	}
		// 	index++;
		// }

		loadFunctions();


		// typedef int wglGetSwapIntervalEXTFunction(void);
		// wglGetSwapIntervalEXTFunction* wglGetSwapIntervalEXT;
		// typedef int wglSwapIntervalEXTFunction(void);
		// wglSwapIntervalEXTFunction* wglSwapIntervalEXT;


		// gl##name = (name##Function*)wglGetProcAddress("gl" #name);
		wglGetSwapIntervalEXT = (wglGetSwapIntervalEXTFunction*)wglGetProcAddress("wglGetSwapIntervalEXT");
		wglSwapIntervalEXT = (wglSwapIntervalEXTFunction*)wglGetProcAddress("wglSwapIntervalEXT");

		wglSwapIntervalEXT(1);


		// BOOL wglSwapIntervalEXT(int interval)
		// int wglGetSwapIntervalEXT(void)

		int interval = wglGetSwapIntervalEXT();

		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		// glEnable(GL_FRAMEBUFFER_SRGB);
		// glEnable(GL_DEPTH_TEST);
		// glDepthRange(-1.0, 1.0);
		glEnable(GL_CULL_FACE);
		// glEnable(GL_BLEND);
		// glBlendFunc(0x0302, 0x0303);
		// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		
		uint vao = 0;
		glCreateVertexArrays(1, &vao);
		glBindVertexArray(vao);

		ad->textures[ad->texCount++] = loadTextureFile("..\\data\\white.png", 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		ad->textures[ad->texCount++] = loadTextureFile("..\\data\\rect.png", 2, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);



		PipelineIds* ids = &globalGraphicsState->pipelineIds;
		ids->programQuad = createShader(vertexShaderQuad, fragmentShaderQuad, &ids->quadVertex, &ids->quadFragment);
		ids->quadVertexMod = glGetUniformLocation(ids->quadVertex, "mod");
		ids->quadVertexUV = glGetUniformLocation(ids->quadVertex, "setUV");
		ids->quadVertexColor = glGetUniformLocation(ids->quadVertex, "setColor");
		ids->quadVertexCamera = glGetUniformLocation(ids->quadVertex, "camera");
		ad->programs[0] = ids->programQuad;

		ids->programCube = createShader(vertexShaderCube, fragmentShaderCube, &ids->cubeVertex, &ids->cubeFragment);
		ids->cubeVertexModel = glGetUniformLocation(ids->cubeVertex, "model");
		ids->cubeVertexView = glGetUniformLocation(ids->cubeVertex, "view");
		ids->cubeVertexProj = glGetUniformLocation(ids->cubeVertex, "proj");
		ids->cubeVertexColor = glGetUniformLocation(ids->cubeVertex, "setColor");

		ids->cubeVertexMode = glGetUniformLocation(ids->cubeVertex, "mode");
		ids->cubeVertexVertices = glGetUniformLocation(ids->cubeVertex, "vertices");
		ad->programs[1] = ids->programCube;


		ad->camera = vec3(0,0,10);

		// ad->camPos = vec3(-10,-10,5);
		// ad->camPos = vec3(30,30,50);
		ad->camPos = vec3(35,35,32);
		// ad->camLook = normVec3(vec3(-1,-1,0));
		ad->camLook = normVec3(vec3(0,1,0));
		// ad->camLook = vec3(0,0,1);
		ad->camRot = vec2(0,0);



		Font font;
		char* path = "..\\data\\LiberationMono-Bold.ttf";
		font.fileBuffer = (char*)getPMemory(fileSize(path) + 1);
		readFileToBuffer(font.fileBuffer, path);
		font.size = vec2i(512,512);
		unsigned char* fontBitmapBuffer = (unsigned char*)getTMemory(font.size.x*font.size.y);
		unsigned char* fontBitmap = (unsigned char*)getTMemory(font.size.x*font.size.y*4);
		
		font.height = 22;
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
		ad->font = font;





		GLenum glError = glGetError(); printf("GLError: %i\n", glError);

		ids->programVoxel = createShader(stbvox_get_vertex_shader(), stbvox_get_fragment_shader(), &ids->voxelVertex, &ids->voxelFragment);

		ad->voxelSamplers[0] = createSampler(GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_NEAREST);
		ad->voxelSamplers[1] = createSampler(GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_NEAREST);
		ad->voxelSamplers[2] = createSampler(GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);



		glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, ad->voxelTextures);

		char** files = (char**)getTMemory(sizeof(char*)*1024);

		int index = 0;
		int skip = 0;
		WIN32_FIND_DATA data;
		HANDLE handle = FindFirstFile("..\\data\\minecraft textures\\*", &data);
		if(handle) {
			while(FindNextFile(handle, &data)) {
				skip--;
				if(skip > 0) continue;

				char* name = data.cFileName;
				if(strLen(name) < 5) continue;

				files[index] = getTString(strLen(name)+1);
				strClear(files[index]);

				strCpy(files[index++], name);
			}
		}
		FindClose(handle);


		uint format = GL_RGBA;
		uint internalFormat = GL_RGBA8;
		int texCount = index;
		// int texCount = 90;
		int width = 32;
		int height = 32;

		char* p = getTString(34);
		strClear(p);
		strAppend(p, "..\\data\\minecraft textures\\");

		char* fullPath = getTString(234);

		// texId = ad->voxelTextures[0];
		// glTextureStorage3D(texId, 6, internalFormat, width, height, texCount);
		// for(int tc = 0; tc < texCount; tc++) {
		// 	unsigned char* stbData;
		// 	int x,y,n;

		// 	strClear(fullPath);
		// 	strAppend(fullPath, p);
		// 	strAppend(fullPath, files[tc]);
		// 	stbData = stbi_load(fullPath, &x, &y, &n, 4);

		// 	if(x == width && y == height) {
		// 		glTextureSubImage3D(texId, 0, 0, 0, tc, x, y, 1, format, GL_UNSIGNED_BYTE, stbData);
		// 		glGenerateTextureMipmap(texId);

		// 		ad->textures[ad->texCount++] = loadTexture(stbData, x, y, 2, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		// 	}

		// 	stbi_image_free(stbData);
		// }


		// glTextureStorage3D(ad->voxelTextures[0], 6, GL_RGBA8, 32, 32, BX_Size);

		// for(int layerIndex = 0; layerIndex < BX_Size; layerIndex++) {
		// 	char* filePath;
		// 	if(layerIndex == 0) filePath = "..\\data\\minecraft textures\\none.png";
		// 	else if(layerIndex == 1) filePath = "..\\data\\minecraft textures\\water.png";
		// 	else if(layerIndex == 2) filePath = "..\\data\\minecraft textures\\sand.png";
		// 	else if(layerIndex == 3) filePath = "..\\data\\minecraft textures\\grass.png";
		// 	else if(layerIndex == 4) filePath = "..\\data\\minecraft textures\\stone.png";
		// 	else if(layerIndex == 5) filePath = "..\\data\\minecraft textures\\snow.png";

		// 	int x,y,n;
		// 	unsigned char* stbData = stbi_load(filePath, &x, &y, &n, 4);
			
		// 	glTextureSubImage3D(ad->voxelTextures[0], 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);
		// 	glGenerateTextureMipmap(ad->voxelTextures[0]);

		// 	stbi_image_free(stbData);
		// }

			glTextureStorage3D(ad->voxelTextures[0], 6, GL_RGBA8, 32, 32, BX_Size);

			for(int layerIndex = 0; layerIndex < BX_Size; layerIndex++) {
				int x,y,n;
				unsigned char* stbData = stbi_load(textureFilePaths[layerIndex], &x, &y, &n, 4);
				
				glTextureSubImage3D(ad->voxelTextures[0], 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);
				glGenerateTextureMipmap(ad->voxelTextures[0]);

				stbi_image_free(stbData);
			}


		// ad->textures[ad->texCount++] = loadTexture("..\\data\\minecraft textures\\water.png", x, y, 2, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);


		glTextureStorage3D(ad->voxelTextures[1], 6, GL_RGBA8, 32, 32, 1);

		int x,y,n;
		unsigned char* stbData = stbi_load("..\\data\\minecraft textures\\test.png", &x, &y, &n, 4);
		
		glTextureSubImage3D(ad->voxelTextures[1], 0, 0, 0, 0, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);
		glGenerateTextureMipmap(ad->voxelTextures[1]);

		stbi_image_free(stbData);






		glCreateFramebuffers(2, ad->frameBuffers);
		glCreateRenderbuffers(2, ad->renderBuffers);
		glCreateTextures(GL_TEXTURE_2D, 2, ad->frameBufferTextures);
		// GLenum result = glCheckNamedFramebufferStatus(ad->frameBuffers[0], GL_FRAMEBUFFER);




		// setup
		ad->voxelHashSize = sizeof(arrayCount(ad->voxelHash));
		for(int i = 0; i < ad->voxelHashSize; i++) {
			ad->voxelHash[i] = (VoxelNode*)getPMemory(sizeof(VoxelNode));
			*ad->voxelHash[i] = {};
		}

		for(int i = 0; i < arrayCount(ad->threadData); i++) {
			ad->threadData[i] = {};
		} 

		for(int i = 0; i < arrayCount(ad->voxelCache); i++) {
			ad->voxelCache[i] = (uchar*)getPMemory(sizeof(uchar)*VOXEL_CACHE_SIZE);
			ad->voxelLightingCache[i] = (uchar*)getPMemory(sizeof(uchar)*VOXEL_CACHE_SIZE);
		}


		#if 0
		int radius = 10;
		for(int y = -radius; y < radius; y++) {
			for(int x = -radius; x < radius; x++) {
				Vec2i coord = vec2i(x,y);
				getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
			}
		}
		#endif


		// ad->playerPos = vec3(35.5f,35.3f,30);
		ad->playerPos = vec3(0,0,0);
		ad->playerLook = vec3(0,1,0);
		ad->playerSize = vec3(0.8f, 0.8f, 1.8f);
		ad->playerCamZOffset = ad->playerSize.z*0.5f - ad->playerSize.x*0.25f;

		ad->playerMode = true;
		ad->pickMode = true;
		ad->selectionRadius = 7;
		input->captureMouse = true;

		*ad->blockMenu = {};
		ad->blockMenuSelected = 0;

		return; // window operations only work after first frame?
	}

	if(second) {
		setWindowProperties(windowHandle, wSettings->res.w, wSettings->res.h, -1920, 0);
		// setWindowProperties(windowHandle, wSettings->res.w, wSettings->res.h, 0, 0);
		setWindowStyle(windowHandle, wSettings->style);
		setWindowMode(windowHandle, wSettings, WINDOW_MODE_FULLBORDERLESS);

		updateAfterWindowResizing(wSettings, &ad->aspectRatio, ad->frameBuffers[0], ad->frameBuffers[1], ad->renderBuffers[0], ad->renderBuffers[1], &ad->frameBufferTextures[0], ad->msaaSamples, &ad->fboRes, &ad->curRes, ad->useNativeRes);
	}

	if(reload) {
		loadFunctions();
	}

	// alloc drawcommandlist	
	int clSize = kiloBytes(100);
	drawCommandListInit(globalCommandList3d, (char*)getTMemory(clSize), clSize);
	drawCommandListInit(globalCommandList2d, (char*)getTMemory(clSize), clSize);

	LARGE_INTEGER counter;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency); 

	if(second) {
		QueryPerformanceCounter(&counter);
		ad->lastTimeStamp = counter.QuadPart;
		ad->dt = 1/(float)60;
	} else {
		QueryPerformanceCounter(&counter);
		float timeStamp = counter.QuadPart;
		ad->dt = (timeStamp - ad->lastTimeStamp);
		ad->dt *= 1000000;
		ad->dt = ad->dt/frequency.QuadPart;
		ad->dt = ad->dt / 1000000;
		ad->dt = clampMax(ad->dt, 1/(float)20);

		ad->lastTimeStamp = timeStamp;
	}
	// printf("%f \n", ad->dt);
	// ad->dt = 0.016f;

	updateInput(&ad->input, isRunning, windowHandle);

	if(input->keysPressed[VK_F1]) {
		int mode;
		if(wSettings->fullscreen) mode = WINDOW_MODE_WINDOWED;
		else mode = WINDOW_MODE_FULLBORDERLESS;
		setWindowMode(windowHandle, wSettings, mode);

		updateAfterWindowResizing(wSettings, &ad->aspectRatio, ad->frameBuffers[0], ad->frameBuffers[1], ad->renderBuffers[0], ad->renderBuffers[1], &ad->frameBufferTextures[0], ad->msaaSamples, &ad->fboRes, &ad->curRes, ad->useNativeRes);

		// static float d = 0;
		// d += ad->dt;

		// ad->msaaSamples = 1;
		// ad->fboRes = vec2i(0, 100 + sin(d)*100);
		// ad->useNativeRes = false;
	}

	if(input->keysPressed[VK_F2]) {
		input->captureMouse = !input->captureMouse;
	}

	bool focus = GetFocus() == windowHandle;
	bool fpsMode = input->captureMouse && focus;

    if(fpsMode) {
		int w,h;
		Vec2i wPos;
		getWindowProperties(systemData->windowHandle, &w, &h, 0, 0, &wPos.x, &wPos.y);
		SetCursorPos(wPos.x + wSettings->currentRes.x/2, wPos.y + wSettings->currentRes.y/2);

		while(ShowCursor(false) >= 0);
	} else {
		while(ShowCursor(true) < 0);
	}

	// 2d camera controls
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


	if(input->keysPressed[VK_F4]) {
		if(ad->playerMode) {
			ad->camPos = ad->playerPos + vec3(0,0,ad->playerCamZOffset);
			ad->camLook = ad->playerLook;
			ad->camRot = ad->playerRot;
		}
		ad->playerMode = !ad->playerMode;
	}

	#define KEYCODE_0 0x30
	#define KEYCODE_1 0x31
	#define KEYCODE_2 0x32

	// if(input->keysPressed[KEYCODE_1]) {
	// 	ad->pickMode = true;
	// }

	// if(input->keysPressed[KEYCODE_2]) {
	// 	ad->pickMode = false;
	// }

	if(input->mouseWheel) {
		ad->blockMenuSelected += -input->mouseWheel;
		ad->blockMenuSelected = mod(ad->blockMenuSelected, arrayCount(ad->blockMenu));
	}

	if(input->keysPressed[KEYCODE_0]) ad->blockMenuSelected = 9;
	for(int i = 0; i < 9; i++) {
		if(input->keysPressed[KEYCODE_0 + i+1]) ad->blockMenuSelected = i;
	}

	if(ad->playerMode) {
		ad->camVel = vec3(0,0,0);
	} else {
		ad->playerVel = vec3(0,0,0);
	}

	if(!ad->playerMode && input->keysPressed[VK_SPACE]) {
		ad->playerPos = ad->camPos;
		ad->playerLook = ad->camLook;
		ad->playerRot = ad->camRot;
		ad->playerVel = ad->camVel;
		ad->playerMode = true;
		input->keysPressed[VK_SPACE] = false;
		input->keysDown[VK_SPACE] = false;
	}


	#define VK_W 0x57
	#define VK_A 0x41
	#define VK_S 0x53
	#define VK_D 0x44
	#define VK_E 0x45
	#define VK_Q 0x51

	Vec3 gUp = vec3(0,0,1);

	Vec3 pLook, pRight, pUp;
	getCamData(ad->playerLook, ad->playerRot, gUp, &pLook, &pRight, &pUp);
	ad->playerAcc = vec3(0,0,0);

	Vec3 cLook, cRight, cUp;
	getCamData(ad->camLook, ad->camRot, gUp, &cLook, &cRight, &cUp);
	ad->camAcc = vec3(0,0,0);

	Vec3 look, right, up;
	Vec2* rot;
	Vec3* acc;
	float runBoost;
	float speed;
	bool rightLock;

	if(!ad->playerMode) {
		look = cLook;
		right = cRight;
		up = cUp;
		rot = &ad->camRot;
		acc = &ad->camAcc;

		runBoost = 2.0f;
		speed = 150;

		#define VK_T 0x54
		if(input->keysDown[VK_T]) speed = 1000;

		rightLock = false;
	} else {
		look = pLook;
		right = pRight;
		up = pUp;
		rot = &ad->playerRot;
		acc = &ad->playerAcc;

		runBoost = 1.5f;
		speed = 30;

		rightLock = true;
	}

	if((!fpsMode && input->mouseButtonDown[1]) || fpsMode) {
		float turnRate = ad->dt*0.3f;
		rot->y += turnRate * input->mouseDeltaY;
		rot->x += turnRate * input->mouseDeltaX;

		float margin = 0.00001f;
		clamp(&rot->y, -M_PI+margin, M_PI-margin);
	}

	if( input->keysDown[VK_W] || input->keysDown[VK_A] || input->keysDown[VK_S] || 
		input->keysDown[VK_D] || input->keysDown[VK_E] || input->keysDown[VK_Q]) {

		if(rightLock || input->keysDown[VK_CONTROL]) look = cross(gUp, right);

		Vec3 acceleration = vec3(0,0,0);
		if(input->keysDown[VK_SHIFT]) speed *= runBoost;
		if(input->keysDown[VK_W]) acceleration += normVec3(look);
		if(input->keysDown[VK_S]) acceleration += -normVec3(look);
		if(input->keysDown[VK_D]) acceleration += normVec3(right);
		if(input->keysDown[VK_A]) acceleration += -normVec3(right);
		if(input->keysDown[VK_E]) acceleration += normVec3(gUp);
		if(input->keysDown[VK_Q]) acceleration += -normVec3(gUp);
		*acc += normVec3(acceleration)*speed;
	}

	ad->playerAcc.z = 0;

	if(ad->playerMode) {
		// if(input->keysPressed[VK_SPACE]) {
		if(input->keysDown[VK_SPACE]) {
			if(ad->playerOnGround) {
				ad->playerVel += gUp*7.0f;
				ad->playerOnGround = false;
			}
		}
	}

	// make sure the meshs around the player are loaded at startup
	if(second) {
		Vec2i pPos = getMeshCoordFromGlobalCoord(ad->activeCamPos);
		for(int i = 0; i < 2; i++) {
			for(int y = -1; y < 2; y++) {
				for(int x = -1; x < 2; x++) {
					Vec2i coord = pPos - vec2i(x,y);

					VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
					makeMesh(m, ad->voxelHash, ad->voxelHashSize);
				}
			}

			threadQueueComplete(globalThreadQueue);
		}
	}

	bool playerGroundCollision = false;
	bool playerCeilingCollision = false;
	bool playerSideCollision = false;

	float dt = ad->dt;
	if(!ad->playerMode) {
		ad->camVel += ad->camAcc*dt;
		float friction = 0.01f;
		ad->camVel *= pow(friction,dt);

		if(ad->camVel != vec3(0,0,0)) {
			ad->camPos = ad->camPos - 0.5f*ad->camAcc*dt*dt + ad->camVel*dt;
		}
	} else {
		float gravity = 20.0f;

		if(!ad->playerOnGround) ad->playerAcc += -gUp*gravity;
		ad->playerVel = ad->playerVel + ad->playerAcc*dt;

		float friction = 0.01f;
		ad->playerVel.x *= pow(friction,dt);
		ad->playerVel.y *= pow(friction,dt);
		// ad->playerVel *= 0.9f;

		if(ad->playerOnGround) ad->playerVel.z = 0;
		// if(ad->playerOnGround) ad->playerAcc.z = 0;

		if(ad->playerVel != vec3(0,0,0)) {
			Vec3 pPos = ad->playerPos;
			Vec3 pSize = ad->playerSize;

			Vec3 nPos = pPos + -0.5f*ad->playerAcc*dt*dt + ad->playerVel*dt;

			int collisionCount = 0;
			bool collision = true;
			while(collision) {

				// get mesh coords that touch the player box
				Rect3 box = rect3CenDim(nPos, pSize);
				Vec3i voxelMin = getVoxelCoordFromGlobalCoord(box.min);
				Vec3i voxelMax = getVoxelCoordFromGlobalCoord(box.max+1);

				Vec3 collisionBox;
				collision = false;
				float minDistance = 100000;

				// check collision with the voxel thats closest
				for(int x = voxelMin.x; x < voxelMax.x; x++) {
					for(int y = voxelMin.y; y < voxelMax.y; y++) {
						for(int z = voxelMin.z; z < voxelMax.z; z++) {
							Vec3i coord = vec3i(x,y,z);
							uchar* block = getBlockFromVoxelCoord(ad->voxelHash, ad->voxelHashSize, coord);

							if(*block > 0) {
								Vec3 cBox = getGlobalCoordFromVoxelCoord(coord);
								float distance = lenVec3(nPos - cBox);
								if(minDistance == 100000 || distance > minDistance) {
									minDistance = distance;
									collisionBox = cBox;
								}
								collision = true;
							}
						}
					}
				}

				if(collision) {
					collisionCount++;

					float minDistance;
					Vec3 dir = vec3(0,0,0);
					int face;

					// check all the 6 planes and take the one with the shortest distance
					for(int i = 0; i < 6; i++) {
						Vec3 n;
						if(i == 0) 		n = vec3(1,0,0);
						else if(i == 1) n = vec3(-1,0,0);
						else if(i == 2) n = vec3(0,1,0);
						else if(i == 3) n = vec3(0,-1,0);
						else if(i == 4) n = vec3(0,0,1);
						else if(i == 5) n = vec3(0,0,-1);

						// assuming voxel size is 1
						// this could be simpler because the voxels are axis aligned
						Vec3 p = collisionBox + (n * ((pSize/2) + 0.5));
						float d = -dot(p, n);
						float d2 = dot(nPos, n);

						// distances are lower then zero in this case where the point is 
						// not on the same side as the normal
						float distance = d + d2;

						if(i == 0 || distance > minDistance) {
							minDistance = distance;
							dir = n;
							face = i;
						}
					}

					float error = 0.0001f;
					// float error = 0;
					nPos += dir*(-minDistance + error);

					if(face == 5) playerCeilingCollision = true;
					else if(face == 4) playerGroundCollision = true;
					else playerSideCollision = true;
				}

				// something went wrong and we reject the move, for now
				if(collisionCount > 5) {
					// nPos = ad->playerPos;

					nPos.z += 5;
					// ad->playerVel = vec3(0,0,0);
					// break;
				}
			}

			float stillnessThreshold = 0.0001f;
			if(valueBetween(ad->playerVel.z, -stillnessThreshold, stillnessThreshold)) {
				ad->playerVel.z = 0;
			}

			if(playerCeilingCollision) {
				ad->playerVel.z = 0;
			}

			if(playerSideCollision) {
				float sideFriction = 0.0010f;
				ad->playerVel.x *= pow(sideFriction,dt);
				ad->playerVel.y *= pow(sideFriction,dt);
			}

			if(collisionCount > 5) {
				ad->playerVel = vec3(0,0,0);
			}

			ad->playerPos = nPos;
		}
	}

	// raycast for touching ground
	if(ad->playerMode) {
		if(playerGroundCollision && ad->playerVel.z <= 0) {
			ad->playerOnGround = true;
			ad->playerVel.z = 0;
		}

		if(ad->playerOnGround) {
			Vec3 pos = ad->playerPos;
			Vec3 size = ad->playerSize;
			Rect3 box = rect3CenDim(pos, size);

			bool groundCollision = false;

			for(int i = 0; i < 4; i++) {
				Vec3 gp;
				if(i == 0) 		gp = box.min + size*vec3(0,0,0);
				else if(i == 1) gp = box.min + size*vec3(1,0,0);
				else if(i == 2) gp = box.min + size*vec3(0,1,0);
				else if(i == 3) gp = box.min + size*vec3(1,1,0);

				// drawCube(&ad->pipelineIds, block, vec3(1,1,1)*1.01f, vec4(1,0,1,1), 0, vec3(0,0,0));

				float raycastThreshold = 0.01f;
				gp -= gUp*raycastThreshold;

				Vec3 block = getBlockCenterFromGlobalCoord(gp);
				uchar* blockType = getBlockFromGlobalCoord(ad->voxelHash, ad->voxelHashSize, gp);

				if(*blockType > 0) {
					groundCollision = true;
					break;
				}
			}

			if(groundCollision) {
				if(ad->playerVel.z <= 0) ad->playerOnGround = true;
			} else {
				ad->playerOnGround = false;
			}
		}
	}

	if(ad->playerMode) {
		ad->activeCamPos = ad->playerPos + vec3(0,0,ad->playerCamZOffset);
		ad->activeCamLook = pLook;
		ad->activeCamUp = pUp;
		ad->activeCamRight = pRight;
	} else {
		ad->activeCamPos = ad->camPos;
		ad->activeCamLook = cLook;
		ad->activeCamUp = cUp;
		ad->activeCamRight = cRight;
	}

	// selecting blocks and modifying them
	// if(input->mouseButtonPressed[0] && ad->playerMode) {
	if(ad->playerMode) {
		ad->blockSelected = false;

		// get block in line
		Vec3 startDir = pLook;
		Vec3 startPos = ad->playerPos + vec3(0,0,ad->playerCamZOffset);

		Vec3 newPos = startPos;
		int smallerAxis[2];
		int biggestAxis = getBiggestAxis(startDir, smallerAxis);

		bool intersection = false;
		Vec3 intersectionBox;

		int intersectionFace;

		for(int i = 0; i < ad->selectionRadius; i++) {
			newPos = newPos + normVec3(startDir);

			Vec3 coords[9];
			int coordsSize = 0;

			Vec3 blockCoords = getGlobalCoordFromVoxelCoord(getVoxelCoordFromGlobalCoord(newPos));

			// we populate 8 blocks around the biggest axis
			for(int y = -1; y < 2; y++) {
				for(int x = -1; x < 2; x++) {
					Vec3 dir = vec3(0,0,0);
					dir.e[smallerAxis[0]] = y;
					dir.e[smallerAxis[1]] = x;
	
					coords[coordsSize++] = blockCoords + dir;
				}
			}

			bool firstIntersection = true;
			float minDistance = -1;

			for(int i = 0; i < coordsSize; i++) {
				Vec3 block = coords[i];

				uchar* blockType = getBlockFromGlobalCoord(ad->voxelHash, ad->voxelHashSize, block);
				Vec3 temp = getGlobalCoordFromVoxelCoord(getVoxelCoordFromGlobalCoord(block));

				// Vec4 c;
				// if(i == 0) c = vec4(1,0,1,1);
				// else c = vec4(0,0,1,1);
				// glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				// drawCube(&ad->pipelineIds, temp, vec3(1.0f,1.0f,1.0f), c, 0, vec3(0,0,0));

				// if(blockType > 0) {
				if(*blockType > 0) {
					Vec3 iBox = getGlobalCoordFromVoxelCoord(getVoxelCoordFromGlobalCoord(block));
					float distance;
					int face;
					bool inter = boxRaycast(startPos, startDir, rect3CenDim(iBox, vec3(1,1,1)), &distance, &face);

					if(inter) {
						if(firstIntersection) {
							minDistance = distance;
							intersectionBox = iBox;
							firstIntersection = false;
							intersectionFace = face;
						} else if(distance < minDistance) {
							minDistance = distance;
							intersectionBox = iBox;
							intersectionFace = face;
						}

						intersection = true;
					}
				}
			}

			if(intersection) break;
		}

		if(intersection) {
			ad->selectedBlock = intersectionBox;
			ad->blockSelected = true;

			Vec3 faceDir = vec3(0,0,0);
				 if(intersectionFace == 0) faceDir = vec3(-1,0,0);
			else if(intersectionFace == 1) faceDir = vec3(1,0,0);
			else if(intersectionFace == 2) faceDir = vec3(0,-1,0);
			else if(intersectionFace == 3) faceDir = vec3(0,1,0);
			else if(intersectionFace == 4) faceDir = vec3(0,0,-1);
			else if(intersectionFace == 5) faceDir = vec3(0,0,1);
			ad->selectedBlockFaceDir = faceDir;

			// if(input->mouseButtonPressed[0] && ad->playerMode) {
			if(ad->playerMode) {
				VoxelMesh* vm = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, getMeshCoordFromGlobalCoord(intersectionBox));

				uchar* block = getBlockFromGlobalCoord(ad->voxelHash, ad->voxelHashSize, intersectionBox);
				uchar* lighting = getLightingFromGlobalCoord(ad->voxelHash, ad->voxelHashSize, intersectionBox);

				bool mouse1 = input->mouseButtonPressed[0];
				bool mouse2 = input->mouseButtonPressed[1];
				bool placeBlock = (!fpsMode && ad->pickMode && mouse1) || (fpsMode && mouse1);
				bool removeBlock = (!fpsMode && !ad->pickMode && mouse1) || (fpsMode && mouse2);

				if(placeBlock || removeBlock) {
					vm->upToDate = false;
					vm->meshUploaded = false;
					vm->modifiedByUser = true;

					// if block at edge of mesh, we have to update the mesh on the other side too
					Vec2i currentCoord = getMeshCoordFromGlobalCoord(intersectionBox);
					for(int i = 0; i < 4; i++) {
						Vec3 offset;
						if(i == 0) offset = vec3(1,0,0);
						else if(i == 1) offset = vec3(-1,0,0);
						else if(i == 2) offset = vec3(0,1,0);
						else if(i == 3) offset = vec3(0,-1,0);

						Vec2i mc = getMeshCoordFromGlobalCoord(intersectionBox + offset);
						if(mc != currentCoord) {
							VoxelMesh* edgeMesh = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, mc);
							edgeMesh->upToDate = false;
							edgeMesh->meshUploaded = false;
							edgeMesh->modifiedByUser = true;
						}
					}
				}

				// if(ad->pickMode) {
				if(placeBlock) {
					Vec3 boxToCamDir = startPos - intersectionBox;
					Vec3 sideBlock = getBlockCenterFromGlobalCoord(intersectionBox + faceDir);
					Vec3i voxelSideBlock = getVoxelCoordFromGlobalCoord(sideBlock);

					// CodeDuplication:
					// get mesh coords that touch the player box
					Rect3 box = rect3CenDim(ad->playerPos, ad->playerSize);
					Vec3i voxelMin = getVoxelCoordFromGlobalCoord(box.min);
					Vec3i voxelMax = getVoxelCoordFromGlobalCoord(box.max+1);
					bool collision = false;

					for(int x = voxelMin.x; x < voxelMax.x; x++) {
						for(int y = voxelMin.y; y < voxelMax.y; y++) {
							for(int z = voxelMin.z; z < voxelMax.z; z++) {
								Vec3i coord = vec3i(x,y,z);

								if(coord == voxelSideBlock) {
									collision = true;
									goto forBreak;
								}
							}
						}
					} forBreak:

					if(!collision) {
						uchar* sideBlockType = getBlockFromVoxelCoord(ad->voxelHash, ad->voxelHashSize, voxelSideBlock);
						uchar* sideBlockLighting = getLightingFromVoxelCoord(ad->voxelHash, ad->voxelHashSize, voxelSideBlock);

						// *sideBlockType = 11;
						*sideBlockType = ad->blockMenu[ad->blockMenuSelected];
						*sideBlockLighting = 0;
						// if(*sideBlockType == 6) {
							// *sideBlockLighting = 255;
						// }
					}
				// } else {
				} else if(removeBlock) {
					if(*block > 0) {
						*block = 0;			
						*lighting = 255;
					}
				}
			}
		}
	}




	// Vec3 skyColor = vec3(0.90f, 0.90f, 0.95f);
	// Vec3 skyColor = vec3(0.95f);
	Vec3 skyColor = vec3(0.90f);
	Vec3 fogColor = vec3(0.75f, 0.85f, 0.95f);

	// for tech showcase
	#ifdef STBVOX_CONFIG_LIGHTING_SIMPLE
		skyColor = skyColor * vec3(0.3f);
		fogColor = fogColor * vec3(0.3f);
	#endif 

	glViewport(0,0, ad->curRes.x, ad->curRes.y);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer (GL_DRAW_FRAMEBUFFER, ad->frameBuffers[1]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[0]);
	glClearColor(skyColor.x, skyColor.y, skyColor.z, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// glDepthRange(-1.0,1.0);
	glFrontFace(GL_CW);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_SCISSOR_TEST);
	glEnable(GL_LINE_SMOOTH);
	

	glEnable(GL_TEXTURE_2D);

	// glEnable(GL_ALPHA_TEST);
	// glAlphaFunc(GL_GREATER, 0.9);

	// glDisable(GL_LIGHTING);
	// glDepthFunc(GL_LESS);
	// glClearDepth(1);
	// glDepthMask(GL_TRUE);

	glEnable(GL_MULTISAMPLE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendEquation(GL_FUNC_ADD);

	lookAt(ad->activeCamPos, -ad->activeCamLook, ad->activeCamUp, ad->activeCamRight);
	perspective(degreeToRadian(ad->fieldOfView), ad->aspectRatio, ad->nearPlane, ad->farPlane);

	Mat4 view, proj; 
	viewMatrix(&view, ad->activeCamPos, -ad->activeCamLook, ad->activeCamUp, ad->activeCamRight);
	projMatrix(&proj, degreeToRadian(ad->fieldOfView), ad->aspectRatio, ad->nearPlane, ad->farPlane);

	globalGraphicsState->textureUnits[0] = ad->voxelTextures[0];
	globalGraphicsState->textureUnits[1] = ad->voxelTextures[1];
	globalGraphicsState->samplerUnits[0] = ad->voxelSamplers[0];
	globalGraphicsState->samplerUnits[1] = ad->voxelSamplers[1];
	globalGraphicsState->samplerUnits[2] = ad->voxelSamplers[2];


	// setupVoxelUniforms(vec4(ad->activeCamPos, 1), 1, 0, 2, view, proj, fogColor);
	setupVoxelUniforms(vec4(ad->activeCamPos, 1), 0, 1, 2, view, proj, fogColor);




#if 1
	// @worldgen
	if(reload) {
		// for(int i = 0; i < ad->vMeshsSize; i++) {
			// ad->vMeshs[i].generated = false;
			// ad->vMeshs[i].upToDate = false;

			// startX = randomInt(0, 100000);
			// startXMod = randomInt(0, 100000);
			// startYMod = randomInt(0, 100000);
		// }

		int radius = VIEW_DISTANCE/VOXEL_X;

		// bool generated;
		// bool upToDate;
		// bool meshUploaded;

		// // volatile uint activeGeneration;
		// // volatile uint activeMaking;


		for(int y = -radius; y < radius; y++) {
			for(int x = -radius; x < radius; x++) {
				Vec2i coord = vec2i(y, x);
				// Vec2i coord = vec2i(0,0);	
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
				m->upToDate = false;
				m->meshUploaded = false;

				// m->generated = fwalse;
			}
		}
		return;
	}
#endif

	Vec2i* coordList = (Vec2i*)getTMemory(sizeof(Vec2i)*2000);
	int coordListSize = 0;

	int meshGenerationCount = 0;
	int radCounter = 0;
	int triangleCount = 0;
	int drawCounter = 0;

	Vec2i pPos = getMeshCoordFromGlobalCoord(ad->activeCamPos);
	int radius = VIEW_DISTANCE/VOXEL_X;
	// VoxelMesh* vms = ad->vMeshs;
	// int* vmsSize = &ad->vMeshsSize;

	// generate the meshes around the player in a spiral by drawing lines and rotating
	// the directing every time we reach a corner
	for(int r = 0; r < radius; r++) {
		int lineLength = r == 0? 1 : 8*r;
		int segment = r*2;

		Vec2i lPos = pPos+r;

		int lLength = 0;
		Vec2i lDir = vec2i(0,-1);

		for(int lineId = 0; lineId < lineLength; lineId++) {
			if(r == 0) lPos = pPos;
			else {
				if(lLength == segment) {
					lLength = 0;
					lDir = vec2i(lDir.y, -lDir.x);
				}
				lLength++;

				lPos += lDir;
			}

			radCounter++;

			Vec2i coord = lPos;
			VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);

			if(!m->meshUploaded) {
				makeMesh(m, ad->voxelHash, ad->voxelHashSize);
				meshGenerationCount++;

				if(!m->modifiedByUser) continue;
			}

			// frustum culling
			Vec3 cp = ad->activeCamPos;
			Vec3 cl = ad->activeCamLook;
			Vec3 cu = ad->activeCamUp;
			Vec3 cr = ad->activeCamRight;

			float ar = ad->aspectRatio;
			float fov = degreeToRadian(ad->fieldOfView);
			// float ne = ad->nearPlane;
			// float fa = ad->farPlane;

			Vec3 left = rotateVec3(cl, fov*ar, cu);
			Vec3 right = rotateVec3(cl, -fov*ar, cu);
			Vec3 top = rotateVec3(cl, fov, cr);
			Vec3 bottom = rotateVec3(cl, -fov, cr);

			Vec3 normalLeftPlane = cross(cu, left);
			Vec3 normalRightPlane = cross(right, cu);
			Vec3 normalTopPlane = cross(cr, top);
			Vec3 normalBottomPlane = cross(bottom, cr);

			Vec3 boxPos = vec3(coord.x*VOXEL_X+VOXEL_X*0.5f, coord.y*VOXEL_Y+VOXEL_Y*0.5f, VOXEL_Z*0.5f);
			Vec3 boxSize = vec3(VOXEL_X, VOXEL_Y, VOXEL_Z);

			bool isIntersecting = true;	
			for(int test = 0; test < 4; test++) {

				Vec3 testNormal;
				if(test == 0) testNormal = normalLeftPlane;
				else if(test == 1) testNormal = normalRightPlane;
				else if(test == 2) testNormal = normalTopPlane;
				else if(test == 3) testNormal = normalBottomPlane;

				bool inside = false;
				for(int i = 0; i < 8; i++) {
					Vec3 off;
					switch (i) {
						case 0: off = vec3( 0.5f,  0.5f, -0.5f); break;
						case 1: off = vec3(-0.5f,  0.5f, -0.5f); break;
						case 2: off = vec3( 0.5f, -0.5f, -0.5f); break;
						case 3: off = vec3(-0.5f, -0.5f, -0.5f); break;
						case 4: off = vec3( 0.5f,  0.5f,  0.5f); break;
						case 5: off = vec3(-0.5f,  0.5f,  0.5f); break;
						case 6: off = vec3( 0.5f, -0.5f,  0.5f); break;
						case 7: off = vec3(-0.5f, -0.5f,  0.5f); break;
					}

					Vec3 boxPoint = boxPos + boxSize*off;
					Vec3 p = boxPoint - cp;

					if(dot(p, testNormal) < 0) {
						inside = true;
						break;
					}
				}

				if(!inside) {
					isIntersecting = false;
					break;
				}
			}
			
			if(isIntersecting) {
				// drawVoxelMesh(m);
				coordList[coordListSize++] = m->coord;

				// triangleCount += m->quadCount*4;
				triangleCount += m->quadCount/(float)2;
				drawCounter++;
			}
		}
	}

	float* distanceList = (float*)getTMemory(sizeof(float)*2000);
	int distanceListSize = 0;

	// dcCube({vec3(0,0,30), vec3(2,2,2), vec4(1,1,0,1), 0, vec3(0.0f)});
	// dcCube({vec3(VOXEL_X,VOXEL_Y,30), vec3(2,2,2), vec4(1,0,1,1), 0, vec3(0.0f)});

	Vec3 mc = getGlobalMeshCoord(vec2i(0,0));
	dcCube({mc, vec3(2,2,2), vec4(1,0,0,1), 0, vec3(0.0f)});
	mc = getGlobalMeshCoord(vec2i(1,0));
	dcCube({mc, vec3(2,2,2), vec4(1,0,0,1), 0, vec3(0.0f)});
	mc = getGlobalMeshCoord(vec2i(1,1));
	dcCube({mc, vec3(2,2,2), vec4(1,0,0,1), 0, vec3(0.0f)});
	mc = getGlobalMeshCoord(vec2i(0,1));
	dcCube({mc, vec3(2,2,2), vec4(1,0,0,1), 0, vec3(0.0f)});

	struct SortPair {
		float key;
		int index;
	};

	SortPair* sortList = (SortPair*)getTMemory(sizeof(SortPair)*2000);
	int sortListSize = 0;

	for(int i = 0; i < coordListSize; i++) {
		Vec2 c = getGlobalMeshCoord(coordList[i]).xy;
		float distanceToCamera = lenVec2(ad->activeCamPos.xy - c);
		sortList[sortListSize++] = {distanceToCamera, i};
	}

	for(int off = 0; off < sortListSize-2; off++) {
		bool swap = false;

		for(int i = 0; i < sortListSize-1 - off; i++) {
			if(sortList[i+1].key < sortList[i].key) {
				SortPair temp = sortList[i];
				sortList[i] = sortList[i+1];
				sortList[i+1] = temp;

				swap = true;
			}
		}

		if(!swap) break;
	}

	for(int i = 0; i < sortListSize-1; i++) {
		assert(sortList[i].key <= sortList[i+1].key);
	}

	for(int i = sortListSize-1; i >= 0; i--) {
		VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
		drawVoxelMesh(m);
	}



	// Vec3 off = vec3(0.5f, 0.5f, 0.5f);
	// Vec3 s = vec3(1.01f, 1.01f, 1.01f);

	// for(int i = 0; i < 10; i++) dcCube({vec3(i*10,0,0) + off, s, vec4(0,1,1,1), 0, vec3(1,2,3)});
	// for(int i = 0; i < 10; i++) dcCube({vec3(0,i*10,0) + off, s, vec4(0,1,1,1), 0, vec3(1,2,3)});
	// for(int i = 0; i < 10; i++) dcCube({vec3(0,0,i*10) + off, s, vec4(0,1,1,1), 0, vec3(1,2,3)});


	dcCube({vec3(0,0,40), vec3(10,1,10), vec4(1,0,0,0.2f), 0, vec3(0.0f)});
	dcCube({vec3(0,-10,40), vec3(10,1,10), vec4(0,1,0,0.2f), 0, vec3(0.0f)});


	dcLineWidth({3});

	if(!ad->playerMode) {
		Vec3 pCamPos = ad->playerPos + vec3(0,0,ad->playerCamZOffset);
		dcLine({pCamPos, pCamPos + pLook*0.5f, vec4(1,0,0,1)});
		dcLine({pCamPos, pCamPos + pUp*0.5f, vec4(0,1,0,1)});
		dcLine({pCamPos, pCamPos + pRight*0.5f, vec4(0,0,1,1)});

		dcPolygonMode({POLYGON_MODE_LINE});
		dcCube({ad->playerPos, ad->playerSize, vec4(1,1,1,1), 0, vec3(0,0,0)});
		dcPolygonMode({POLYGON_MODE_FILL});

	} else {
		if(ad->blockSelected) {
			dcCull({false});
			Vec3 vs[4];
			getPointsFromQuadAndNormal(ad->selectedBlock + ad->selectedBlockFaceDir*0.5f*1.01f, ad->selectedBlockFaceDir, 1, vs);
			dcQuad({vs[0], vs[1], vs[2], vs[3], vec4(1,1,1,0.05f)});
			dcCull({true});

			dcPolygonMode({POLYGON_MODE_LINE});
			dcCube({ad->selectedBlock, vec3(1.01f), vec4(0.9f), 0, vec3(0,0,0)});
			dcPolygonMode({POLYGON_MODE_FILL});
		}
	}

	int fontSize = 22;
	int pi = 0;
	Vec4 c = vec4(1.0f,0.3f,0.0f,1);
	Vec4 c2 = vec4(0,0,0,1);

	#define PVEC3(v) v.x, v.y, v.z
	#define PVEC2(v) v.x, v.y
	dcText({fillString("Pos  : (%f,%f,%f)", PVEC3(ad->activeCamPos)), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});
	dcText({fillString("Look : (%f,%f,%f)", PVEC3(ad->activeCamLook)), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});
	dcText({fillString("Up   : (%f,%f,%f)", PVEC3(ad->activeCamUp)), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});
	dcText({fillString("Right: (%f,%f,%f)", PVEC3(ad->activeCamRight)), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});
	dcText({fillString("Rot  : (%f,%f)", PVEC2(ad->camRot)), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});
	dcText({fillString("Vec  : (%f,%f,%f)", PVEC3(ad->playerVel)), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});
	dcText({fillString("Acc  : (%f,%f,%f)", PVEC3(ad->playerAcc)), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});
	dcText({fillString("Draws: (%i)", drawCounter), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});
	dcText({fillString("Quads: (%i)", triangleCount), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});

	// dcText({fillString("Quads: (%i %i)", (int)input->mouseButtonDown[0], (int)input->mouseButtonDown[1]), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});
	// if(input->keysDown[VK_W])
	// dcText({fillString("W", (int)input->mouseButtonDown[1]), &ad->font, vec2(0,-fontSize*pi++), c, 0, 2, 1, vec4(0,0,0,1)});




	if(ad->playerMode) {
		for(int i = 0; i < 10; i++) {
			ad->blockMenu[i] = i+1;
		}

		Vec2 res = vec2(wSettings->currentRes);
		float bottom = 0.950f;
		float size = 0.0527f;
		float dist = 0.5f;
		Vec2 iconSize = vec2(size * res.h);
		float iconDist = iconSize.w * dist;
		// int count = arrayCount(ad->blockMenu);
		int count = 10;

		float selectColor = 1.5f;
		float trans = 0.8f;

		float start = res.x*0.5f - ((count-1)*0.5f)*(iconDist+iconSize.w);
		for(int i = 0; i < count; i++) {
			Vec4 color = vec4(1,1,1,trans);
			float iconOff = 1;
			if(ad->blockMenuSelected == i) {
				color = vec4(1*selectColor,1*selectColor,1*selectColor,trans);
				iconOff = 1.2f;
			}

			Vec2 pos = vec2(start + i*(iconSize.w + iconDist), -res.y*bottom);
			dcRect({rectCenDim(pos, iconSize*1.2f*iconOff), rect(0,0,1,1), vec4(vec3(0.1f),trans), 1});
			dcRect({rectCenDim(pos, iconSize*iconOff), rect(0,0,1,1), color, 6+i});
		}
	}







	glBindProgramPipeline(globalGraphicsState->pipelineIds.programCube);

	char* drawListIndex = (char*)globalCommandList3d->data;
	for(int i = 0; i < globalCommandList3d->count; i++) {
		int command = *((int*)drawListIndex);
		drawListIndex += sizeof(int);

		switch(command) {
			case Draw_Command_Cube_Type: {
				dcCase(Cube, dc, drawListIndex);
				drawCube(dc.trans, dc.scale, dc.color, dc.degrees, dc.rot);
			} break;

			case Draw_Command_Line_Type: {
				dcCase(Line, dc, drawListIndex);
				drawLine(dc.p0, dc.p1, dc.color);
			} break;

			case Draw_Command_Quad_Type: {
				dcCase(Quad, dc, drawListIndex);
				drawQuad(dc.p0, dc.p1, dc.p2, dc.p3, dc.color);
			} break;

			case Draw_Command_PolygonMode_Type: {
				dcCase(PolygonMode, dc, drawListIndex);
				int m;
				switch(dc.mode) {
					case POLYGON_MODE_FILL: m = GL_FILL; break;
					case POLYGON_MODE_LINE: m = GL_LINE; break;
					case POLYGON_MODE_POINT: m = GL_POINT; break;
				}
				glPolygonMode(GL_FRONT_AND_BACK, m);
			} break;

			case Draw_Command_LineWidth_Type: {
				dcCase(LineWidth, dc, drawListIndex);
				glLineWidth(dc.width);
			} break;

			case Draw_Command_Cull_Type: {
				dcCase(Cull, dc, drawListIndex);
				if(dc.b) glEnable(GL_CULL_FACE);
				else glDisable(GL_CULL_FACE);
			} break;

			default: {

			} break;
		}
	}

	// ortho(rectCenDim(cam->x,cam->y, cam->z, cam->z/ad->aspectRatio));
	// glBindProgramPipeline(globalGraphicsState->pipelineIds.programQuad);
	// drawRect(rectCenDim(0, 0, 0.01f, 100), rect(0,0,1,1), vec4(0.4f,1,0.4f,1), ad->textures[0]);
	// drawRect(rectCenDim(0, 0, 100, 0.01f), rect(0,0,1,1), vec4(0.4f,0.4f,1,1), ad->textures[0]);

	// drawRect(rectCenDim(0, 0, 5, 5), rect(0,0,1,1), vec4(1,1,1,1), ad->textures[2]);
	// drawRect(rectCenDim(0, 0, 5, 5), rect(0,0,1,1), vec4(1,1,1,1), 3);

	// drawRect(rect(2,2,4,4), rect(0,0,1,1), vec4(1,1,0,1), 2);



	ortho(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0));
	glDisable(GL_DEPTH_TEST);
	glBindProgramPipeline(globalGraphicsState->pipelineIds.programQuad);

	glBindFramebuffer (GL_READ_FRAMEBUFFER, ad->frameBuffers[0]);
	glBindFramebuffer (GL_DRAW_FRAMEBUFFER, ad->frameBuffers[1]);
	glBlitFramebuffer (0,0, ad->curRes.x, ad->curRes.y,
	                   0,0, ad->curRes.x, ad->curRes.y,
	                   // GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
	                   GL_COLOR_BUFFER_BIT,
	                   // GL_NEAREST);
	                   GL_LINEAR);

	glBindFramebuffer (GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);

	glViewport(0,0, wSettings->currentRes.x, wSettings->currentRes.y);
	glBindProgramPipeline(globalGraphicsState->pipelineIds.programQuad);
	drawRect(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0), rect(0,1,1,0), vec4(1), ad->frameBufferTextures[0]);

	glBindFramebuffer (GL_FRAMEBUFFER, 0);	

	char* drawListIndex2 = (char*)globalCommandList2d->data;
	for(int i = 0; i < globalCommandList2d->count; i++) {
		int command = *((int*)drawListIndex2);
		drawListIndex2 += sizeof(int);

		switch(command) {
			case Draw_Command_Rect_Type: {
				dcCase(Rect, dc, drawListIndex2);
				drawRect(dc.r, dc.uv, dc.color, dc.texture);
			} break;

			case Draw_Command_Text_Type: {
				dcCase(Text, dc, drawListIndex2);
				drawText(dc.text, dc.font, dc.pos, dc.color, dc.vAlign, dc.hAlign, dc.shadow, dc.shadowColor);
			} break;

			default: {
			} break;
		}
	}

	swapBuffers(&ad->systemData);
	glFinish();

	if(second) {
		GLenum glError = glGetError(); printf("GLError: %i\n", glError);
		int stop = 2;
	}

	clearTMemory();
}
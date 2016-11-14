// #pragma optimize( "", off )
#pragma optimize( "", on )

#include <iacaMarks.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <gl\gl.h>
// #include "glext.h"

#include "rt_misc.h"
#include "rt_math.h"
#include "rt_hotload.h"
#include "rt_misc_win32.h"
#include "rt_platformWin32.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_JPEG

#include "stb_image.h"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#define STB_VOXEL_RENDER_IMPLEMENTATION
// #define STBVOX_CONFIG_LIGHTING_SIMPLE
#define STBVOX_CONFIG_FOG_SMOOTHSTEP
// #define STBVOX_CONFIG_MODE 0
#define STBVOX_CONFIG_MODE 1
#include "stb_voxel_render.h"

#define USE_SRGB 1
 
// #pragma optimize( "", off )
// #pragma optimize( "", on )

/*
//-----------------------------------------
//				WHAT TO DO
//-----------------------------------------
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
- screen space reflections

- advance timestep when window not in focus (stop game when not in focus)
- reduce number of thread queue pushes
- insert and init mesh not thread proof, make sure every mesh is initialized before generating
- implement sun and clouds that block beams of light
- glowstone emmiting light
- make voxel drop in tutorial code for stb_voxel
- stb_voxel push alpha test in voxel vertex shader
- rocket launcher
- antialiased pixel graphics with neighbour sampling 
- macros for array stuff so i dont have to inline updateMeshList[updateMeshListSize++] every time 
- level of detail for world gen back row								
- 32x32 gen chunks

- put in spaces for fillString
- save mouse position at startup and place the mouse there when the app closes
- simplex noise instead of perlin noise
- make thread push copy the thead specific data on push in a seperate buffer for all possible threadjobs
- frametime timer and fps counter

- simd voxel generation
- simd on vectors as a test first?

- experiment with directx 11
- font drawing bold

gui stuff: 
additionally:
- 2d drawing
- 3d drawing
- vector representation
- sound 
- entities
- make columns stack
- drag element to seperate column elements
- struct as defer for section begin and such
- make settings pushable as a whole
  -> doesn't that remove the need for explicit stacking in scrollbars and such?
- top menu bar

//-------------------------------------
//               BUGS
//-------------------------------------
- window operations only work after first frame
- look has to be negative to work in view projection matrix
- distance jumping collision bug, possibly precision loss in distances
- gpu fucks up at some point making swapBuffers alternates between time steps 
  which makes the game stutter, restart is required
- game input gets stuck when buttons are pressed right at the start
- sort key assert firing randomly
- hotload gets stuck sometimes, thread that won't complete

- draw line crashes
- text looks wrong after srgb conversion
- release build takes forever
- threadQueueComplete(ThreadQueue* queue) doesnt work
- threadqueue do next work from main thread bug
*/

/*
	switching shader -> 550 ticks
	using namedBufferSubData vs uniforms for vertices -> 2400 ticks vs 400 ticks
*/

// main 171, radix 70 - 285,70

ThreadQueue* globalThreadQueue;

struct GraphicsState;
struct DrawCommandList;
GraphicsState* globalGraphicsState;
DrawCommandList* globalCommandList;
struct MemoryBlock;
MemoryBlock* globalMemory;

struct MemoryBlock {
	ExtendibleMemoryArray* pMemory;
	MemoryArray* tMemory;
	ExtendibleBucketMemory* dMemory;

	MemoryArray* tMemoryDebug;
};

#define getPStruct(type) 		(type*)(getPMemory(sizeof(type)))
#define getPArray(type, count) 	(type*)(getPMemory(sizeof(type) * count))
#define getTStruct(type) 		(type*)(getTMemory(sizeof(type)))
#define getTArray(type, count) 	(type*)(getTMemory(sizeof(type) * count))
#define getTString(size) 		(char*)(getTMemory(size)) 
#define getDStruct(type) 		(type*)(getDMemory(sizeof(type)))
#define getDArray(type, count) 	(type*)(getDMemory(sizeof(type) * count))

void *getPMemory(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getExtendibleMemoryArray(size, memory->pMemory);
    return location;
}

void * getTMemory(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getMemoryArray(size, memory->tMemory);
    return location;
}

void clearTMemory(MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	clearMemoryArray(memory->tMemory);

	// zeroMemory(memory->temporary, memory->temporarySize);
}

// void pushMarkerTMemory(MemoryBlock * memory = 0)  {
    // if(!memory) memory = globalMemory;
    // memory->markerStack[memory->markerStackIndex] = memory->temporaryIndex;
    // memory->markerStackIndex++;
// }

// void popMarkerTMemory(MemoryBlock * memory = 0)  {
    // if(!memory) memory = globalMemory;
    // int size = memory->temporaryIndex - memory->markerStack[memory->markerStackIndex];
    // memory->markerStackIndex--;
    // freeTMemory(size, memory);
// }

void * getDMemory(int size, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	void* location = getExtendibleBucketMemory(memory->dMemory);
    return location;
}

void freeDMemory(void* address, MemoryBlock * memory = 0) {
	if(memory == 0) memory = globalMemory;

	freeExtendibleBucketMemory(address, memory->dMemory);
}





// char* dataFolder = "...\\data\\";
// #define GetFilePath(name) ...\\data\\##name

#define GL_TEXTURE_CUBE_MAP_SEAMLESS      0x884F
#define GL_FRAMEBUFFER_SRGB               0x8DB9
#define GL_FRAMEBUFFER_SRGB               0x8DB9
#define GL_TEXTURE_BUFFER                 0x8C2A
#define GL_MAP_WRITE_BIT                  0x0002
#define GL_MAP_PERSISTENT_BIT             0x0040
#define GL_MAP_COHERENT_BIT               0x0080
#define GL_VERTEX_SHADER                  0x8B31
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_VERTEX_SHADER_BIT              0x00000001
#define GL_FRAGMENT_SHADER_BIT            0x00000002
#define GL_DEBUG_OUTPUT                   0x92E0
#define WGL_CONTEXT_FLAGS_ARB             0x2094
#define WGL_CONTEXT_DEBUG_BIT_ARB         0x0001
#define WGL_CONTEXT_MAJOR_VERSION_ARB     0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB     0x2092
#define GL_MAJOR_VERSION                  0x821B
#define GL_MINOR_VERSION                  0x821C
#define GL_RGB32F                         0x8815
#define GL_RGBA8I                         0x8D8E
#define GL_RGBA8UI                        0x8D7C
#define GL_R8                             0x8229
#define GL_ARRAY_BUFFER                   0x8892

#define GL_CLAMP_TO_EDGE                  0x812F
#define GL_TEXTURE_MAX_ANISOTROPY_EXT     0x84FE
#define GL_TEXTURE_WRAP_R                 0x8072
#define GL_STATIC_DRAW                    0x88E4
#define GL_SRGB8_ALPHA8                   0x8C43
#define GL_TEXTURE_CUBE_MAP_ARRAY         0x9009
#define GL_STREAM_DRAW                    0x88E0
#define GL_TEXTURE_2D_ARRAY               0x8C1A
#define GL_FRAMEBUFFER                    0x8D40
#define GL_COLOR_ATTACHMENT0              0x8CE0
#define GL_RENDERBUFFER                   0x8D41
#define GL_DEPTH_STENCIL                  0x84F9
#define GL_DEPTH_STENCIL_ATTACHMENT       0x821A
#define GL_DEPTH24_STENCIL8               0x88F0
#define GL_DEBUG_OUTPUT_SYNCHRONOUS       0x8242
#define GL_DEBUG_SEVERITY_HIGH            0x9146
#define GL_DEBUG_SEVERITY_MEDIUM          0x9147
#define GL_DEBUG_SEVERITY_LOW             0x9148
#define GL_DEBUG_SEVERITY_NOTIFICATION    0x826B
#define GL_MULTISAMPLE                    0x809D
#define GL_FUNC_ADD                       0x8006
#define GL_CLIP_DISTANCE0                 0x3000
#define GL_CLIP_DISTANCE1                 0x3001
#define GL_FUNC_ADD                       0x8006
#define GL_MAX                            0x8008
typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;


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
	GLOP(GLuint, GetDebugMessageLog, GLuint counter, GLsizei bufSize, GLenum *source, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, char *messageLog) \
	GLOP(void, TextureStorage3D, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth) \
	GLOP(void, TextureSubImage3D, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels) \
	GLOP(void, GenerateTextureMipmap, GLuint texture) \
	GLOP(void, TextureStorage2D, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) \
	GLOP(void, TextureSubImage2D, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) \
	GLOP(void, CreateSamplers, GLsizei n, GLuint *samplers) \
	GLOP(void, SamplerParameteri, GLuint sampler, GLenum pname, GLint param) \
	GLOP(void, BindSamplers, GLuint first, GLsizei count, const GLuint *samplers) \
	GLOP(void, TextureParameteri, GLuint texture, GLenum pname, GLint param) \
	GLOP(GLint, GetUniformLocation, GLuint program, const GLchar *name) \
	GLOP(void, ProgramUniform4f, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2,GLfloat v3) \
	GLOP(void, ProgramUniform3f, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2) \
	GLOP(void, ProgramUniform2f, GLuint program, GLint location, GLfloat v0, GLfloat v1) \
	GLOP(void, ProgramUniform1f, GLuint program, GLint location, GLfloat v0) \
	GLOP(void, ProgramUniform1i, GLuint program, GLint location, GLuint v0) \
	GLOP(void, ProgramUniform1fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
	GLOP(void, ProgramUniform2fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
	GLOP(void, ProgramUniform1iv, GLuint program, GLint location, GLsizei count, const GLint *value) \
	GLOP(void, ProgramUniformMatrix4fv, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) \
	GLOP(void, ProgramUniform4fv, GLuint program, GLint location, GLsizei count, const GLfloat *value) \
	GLOP(void, UseProgram, GLuint program) \
	GLOP(void, ActiveTexture, GLenum texture) \
	GLOP(void, EnableVertexAttribArray, GLuint index) \
	GLOP(void, DisableVertexAttribArray, GLuint index) \
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
	GLOP(void, GenSamplers, GLsizei n, GLuint *samplers) \
	GLOP(void, BindSampler, GLuint unit, GLuint sampler) \
	GLOP(void, BindTextureUnit, GLuint unit, GLuint texture) \
	GLOP(void, NamedBufferSubData, GLuint buffer, GLintptr offset, GLsizei size, const void *data) \
	GLOP(void, NamedBufferSubDataEXT, GLuint buffer, GLintptr offset, GLsizei size, const void *data) \
	GLOP(void, GetUniformiv, GLuint program, GLint location, GLint * params) \
	GLOP(void, GetUniformfv, GLuint program, GLint location, GLfloat * params) \
	GLOP(void, CreateFramebuffers, GLsizei n, GLuint *framebuffers) \
	GLOP(void, NamedFramebufferParameteri, GLuint framebuffer, GLenum pname, GLint param) \
	GLOP(void, NamedRenderbufferStorageMultisample, GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height) \
	GLOP(void, CreateRenderbuffers, GLsizei n, GLuint *renderbuffers) \
	GLOP(void, NamedFramebufferRenderbuffer, GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer) \
	GLOP(void, BindFramebuffer, GLenum target, GLuint framebuffer) \
	GLOP(void, NamedFramebufferDrawBuffer, GLuint framebuffer, GLenum buf) \
	GLOP(void, BlitFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) \
	GLOP(void, BlitNamedFramebuffer, GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) \
	GLOP(void, NamedFramebufferTexture, GLuint framebuffer, GLenum attachment, GLuint texture, GLint level) \
	GLOP(void, TextureStorage2DMultisample, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations) \
	GLOP(void, TexImage2DMultisample, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations) \
	GLOP(void, NamedRenderbufferStorage, GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height) \
	GLOP(GLenum, CheckNamedFramebufferStatus, GLuint framebuffer, GLenum target) \
	GLOP(void, GenFramebuffers, GLsizei n, GLuint *ids) \
	GLOP(void, FramebufferTexture, GLenum target, GLenum attachment, GLuint texture, GLint level) \
	GLOP(void, BlendFuncSeparate, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha) \
	GLOP(void, BlendEquation, GLenum mode) \
	GLOP(void, BlendEquationSeparate, GLenum modeRGB, GLenum modeAlpha) \
	GLOP(void, GetTextureSubImage, uint texture, int level, int xoffset, int yoffset, int zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void *pixels) \
	GLOP(GLubyte*, GetStringi, GLenum name, GLuint index) \
	GLOP(void, DrawArraysInstanced, GLenum mode, GLint first, GLsizei count, GLsizei primcount) \
	GLOP(void, VertexAttribDivisor, GLuint index, GLuint divisor)






// typedef HGLRC wglCreateContextAttribsARBFunction(HDC hDC, HGLRC hshareContext, const int *attribList);
// wglCreateContextAttribsARBFunction* wglCreateContextAttribsARB;

// wglGetSwapIntervalEXT = (wglGetSwapIntervalEXTFunction*)wglGetProcAddress("wglGetSwapIntervalEXT");


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



enum CommandState {
	STATE_SCISSOR,
	STATE_POLYGONMODE, 
	STATE_LINEWIDTH,
	STATE_CULL,
};

enum Polygon_Mode {
	POLYGON_MODE_FILL = 0,
	POLYGON_MODE_LINE,
	POLYGON_MODE_POINT,
};

enum DrawListCommand {
	Draw_Command_State_Type,
	Draw_Command_Enable_Type,
	Draw_Command_Disable_Type,
	Draw_Command_Cube_Type,
	Draw_Command_Line_Type,
	Draw_Command_Quad_Type,
	Draw_Command_Rect_Type,
	Draw_Command_Text_Type,
	Draw_Command_Scissor_Type,
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

#pragma pack(push)
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
	int texZ;
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

struct Draw_Command_Scissor {
	Rect rect;
};

struct Draw_Command_Int {
	int state;
};

struct Draw_Command_State {
	int state;
	int value;
};
#pragma pack(pop)


#define PUSH_DRAW_COMMAND(commandType, structType) \
	if(!drawList) drawList = globalCommandList; \
	char* list = (char*)drawList->data + drawList->bytes; \
	*((int*)list) = Draw_Command_##commandType##_Type; \
	list += sizeof(int); \
	Draw_Command_##structType* command = (Draw_Command_##structType*)list; \
	drawList->count++; \
	drawList->bytes += sizeof(Draw_Command_##structType) + sizeof(int); \
	assert(sizeof(Draw_Command_##structType) + drawList->bytes < drawList->maxBytes);


void dcCube(Vec3 trans, Vec3 scale, Vec4 color, float degrees = 0, Vec3 rot = vec3(0,0,0), DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Cube, Cube);

	command->trans = trans;
	command->scale = scale;
	command->color = color;
	command->degrees = degrees;
	command->rot = rot;
}

void dcLine(Vec3 p0, Vec3 p1, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Line, Line);

	command->p0 = p0;
	command->p1 = p1;
	command->color = color;
}

void dcQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Quad, Quad);

	command->p0 = p0;
	command->p1 = p1;
	command->p2 = p2;
	command->p3 = p3;
	command->color = color;
}

void dcRect(Rect r, Rect uv, Vec4 color, int texture = -1, int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Rect, Rect);

	command->r = r;
	command->uv = uv;
	command->color = color;
	command->texture = texture;
	command->texZ = texZ;
}
void dcRect(Rect r, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Rect, Rect);

	command->r = r;
	command->uv = rect(0,0,1,1);
	command->color = color;
	command->texture = -1;
	command->texZ = -1;
}

void dcText(char* text, Font* font, Vec2 pos, Vec4 color, int vAlign = 0, int hAlign = 0, int shadow = 0, Vec4 shadowColor = vec4(0,0,0,1), DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Text, Text);

	command->text = text;
	command->font = font;
	command->pos = pos;
	command->color = color;
	command->vAlign = vAlign;
	command->hAlign = hAlign;
	command->shadow = shadow;
	command->shadowColor = shadowColor;
}

void dcScissor(Rect rect, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Scissor, Scissor);

	command->rect = rect;
}

void dcState(int state, int value, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(State, State);

	command->state = state;
	command->value = value;
}

void dcEnable(int state, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Enable, Int);

	command->state = state;
}

void dcDisable(int state, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Disable, Int);

	command->state = state;
}






// #define GLSL(src) "#version 330\n \
// 	#extension GL_ARB_gpu_shader5               : enable\n \
// 	#extension GL_ARB_gpu_shader_fp64           : enable\n \
// 	#extension GL_ARB_shader_precision          : enable\n \
// 	#extension GL_ARB_conservative_depth        : enable\n \
// 	#extension GL_ARB_texture_cube_map_array    : enable\n \
// 	#extension GL_ARB_separate_shader_objects   : enable\n \
// 	#extension GL_ARB_shading_language_420pack  : enable\n \
// 	#extension GL_ARB_shading_language_packing  : enable\n \
// 	#extension GL_ARB_explicit_uniform_location : enable\n" #src


// #define GLSL(src) "#version 330\n \

#define GLSL(src) "#version 430\n \
	#extension GL_ARB_bindless_texture 			: enable\n \
	#extension GL_NV_command_list 				: enable\n \
	#extension GL_ARB_shading_language_include 	: enable\n \
	#extension GL_ARB_uniform_buffer_object 	: enable\n \
	#extension GL_NV_vertex_buffer_unified_memory : enable\n \
	#extension GL_NV_uniform_buffer_unified_memory : enable\n \
	#extension GL_NV_shader_buffer_load : enable\n \
	#extension GL_ARB_gpu_shader5               : enable\n \
	#extension GL_ARB_gpu_shader_fp64           : enable\n \
	#extension GL_ARB_shader_precision          : enable\n \
	#extension GL_ARB_conservative_depth        : enable\n \
	#extension GL_ARB_texture_cube_map_array    : enable\n \
	#extension GL_ARB_separate_shader_objects   : enable\n \
	#extension GL_ARB_shading_language_420pack  : enable\n \
	#extension GL_ARB_shading_language_packing  : enable\n \
	#extension GL_ARB_explicit_uniform_location : enable\n" #src




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

	out gl_PerVertex { vec4 gl_Position; };
	// smooth out vec2 uv;
	smooth out vec3 uv;
	out vec4 Color;

	void main() {
		ivec2 pos = quad_uv[gl_VertexID];
		uv = vec3(setUV[pos.x], setUV[2 + pos.y], texZ);
		Color = setColor;
		vec2 model = quad[gl_VertexID]*mod.zw + mod.xy;
		vec2 view = model/(camera.zw*0.5f) - camera.xy/(camera.zw*0.5f);
		gl_Position = vec4(view, 0, 1);
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

	void main() {
		vec3 clipPos = pos;
		if(clipPlane) clipPos.y *= -1;
		color = texture(s, vec4(clipPos, 0));

		// color = texture(s, vec4(pos, 0));
	}
);




struct ShaderUniform {
	int type;
	int vertexLocation;
	int fragmentLocation;
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

enum ShaderProgram {
	SHADER_CUBE = 0,
	SHADER_QUAD,
	SHADER_VOXEL,
	SHADER_TEST,
	SHADER_PARTICLE,
	SHADER_CUBEMAP,

	SHADER_SIZE,
};

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

enum QuadUniforms {
	QUAD_UNIFORM_UV = 0,
	QUAD_UNIFORM_TEXZ,
	QUAD_UNIFORM_MOD,
	QUAD_UNIFORM_COLOR,
	QUAD_UNIFORM_CAMERA,
	QUAD_UNIFORM_SIZE,
};

ShaderUniformType quadShaderUniformType[] = {
	{UNIFORM_TYPE_VEC4, "setUV"},
	{UNIFORM_TYPE_FLOAT, "texZ"},
	{UNIFORM_TYPE_VEC4, "mod"},
	{UNIFORM_TYPE_VEC4, "setColor"},
	{UNIFORM_TYPE_VEC4, "camera"},
};

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

enum TestUniforms {
	TEST_UNIFORM_SIZE = 0,
};

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

enum CubemapUniforms {
	CUBEMAP_UNIFORM_VIEW = 0,
	CUBEMAP_UNIFORM_PROJ,
	CUBEMAP_UNIFORM_CLIPPLANE,
	CUBEMAP_UNIFORM_CPLANE1,

	CUBEMAP_UNIFORM_SIZE,
};

ShaderUniformType cubemapShaderUniformType[] = {
	{UNIFORM_TYPE_MAT4, "view"},
	{UNIFORM_TYPE_MAT4, "proj"},
	{UNIFORM_TYPE_INT, "clipPlane"},
	{UNIFORM_TYPE_VEC4, "cPlane"},
};

MakeShaderInfo makeShaderInfo[] = {
	{(char*)vertexShaderCube, (char*)fragmentShaderCube, CUBE_UNIFORM_SIZE, cubeShaderUniformType},
	{(char*)vertexShaderQuad, (char*)fragmentShaderQuad, QUAD_UNIFORM_SIZE, quadShaderUniformType},
	{(char*)stbvox_get_vertex_shader(), (char*)stbvox_get_fragment_shader(), VOXEL_UNIFORM_SIZE, voxelShaderUniformType},
	{(char*)vertexShaderTest, (char*)fragmentShaderTest, TEST_UNIFORM_SIZE, 0},
	{(char*)vertexShaderParticle, (char*)fragmentShaderParticle, PARTICLE_UNIFORM_SIZE, particleShaderUniformType},
	{(char*)vertexShaderCubeMap, (char*)fragmentShaderCubeMap, CUBEMAP_UNIFORM_SIZE, cubemapShaderUniformType},
};

struct Shader {
	uint program;
	uint vertex;
	uint fragment;
	int uniformCount;
	ShaderUniform* uniforms;
};

enum TextureId {
	TEXTURE_WHITE = 0,
	TEXTURE_RECT,
	TEXTURE_CIRCLE,
	TEXTURE_TEST,
	TEXTURE_SIZE,
};

char* texturePaths[] = {
	"..\\data\\white.png",
	"..\\data\\rect.png",
	"..\\data\\circle.png",
	"..\\data\\test.png",
};

struct Texture {
	uint id;
	Vec2i dim;
	int channels;
	int levels;
};

Texture loadTexture(unsigned char* buffer, int w, int h, int mipLevels, int internalFormat, int channelType, int channelFormat) {
	uint textureId;
	glCreateTextures(GL_TEXTURE_2D, 1, &textureId);
	glTextureStorage2D(textureId, mipLevels, internalFormat, w, h);
	glTextureSubImage2D(textureId, 0, 0, 0, w, h, channelType, channelFormat, buffer);
	glGenerateTextureMipmap(textureId);

	Texture tex = {textureId, vec2i(w,h), 4, 1};

	return tex;
}

Texture loadTextureFile(char* path, int mipLevels, int internalFormat, int channelType, int channelFormat) {
	int x,y,n;
	unsigned char* stbData = stbi_load(path, &x, &y, &n, 0);

	Texture tex = loadTexture(stbData, x, y, mipLevels, internalFormat, channelType, channelFormat);
	stbi_image_free(stbData);	

	return tex;
}

enum FontId {
	FONT_LIBERATION_MONO = 0,
	FONT_SOURCESANS_PRO,
	FONT_CONSOLAS,
	FONT_ARIAL,
	FONT_CALIBRI,
	FONT_SIZE,
};

char* fontPaths[] = {
	"..\\data\\LiberationMono-Bold.ttf",
	"..\\data\\SourceSansPro-Regular.ttf",
	"..\\data\\consola.ttf",
	"..\\data\\arial.ttf",
	"..\\data\\calibri.ttf",
};

struct Font {
	char* fileBuffer;
	Texture tex;
	int glyphStart, glyphCount;
	stbtt_bakedchar* cData;
	int height;
};

enum MeshId {
	MESH_CUBE = 0,
	MESH_QUAD,
	MESH_SIZE,
};

struct Vertex {
	Vec3 pos;
	Vec2 uv;
	Vec3 normal;
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

struct MeshMap {
	Vertex* vertexArray;
	int size;
};

MeshMap meshArrays[] = {
	{(Vertex*)cubeArray, sizeof(cubeArray)},
	{(Vertex*)quadArray, sizeof(quadArray)},
};

// char* fontPaths[] = {
	// "..\\data\\LiberationMono-Bold.ttf",
// };

struct Mesh {
	uint bufferId;
	uint elementBufferId;

	// char* buffer;
	// char* elementBuffer;
	int vertCount;
	int elementCount;
};

struct GraphicsState {
	Shader shaders[SHADER_SIZE];
	Texture textures[TEXTURE_SIZE];
	Font fonts[FONT_SIZE];

	Mesh meshs[MESH_SIZE];

	GLuint textureUnits[16];
	GLuint samplerUnits[16];
};

Shader* getShader(int shaderId) {
	Shader* s = globalGraphicsState->shaders + shaderId;
	return s;
}

Mesh* getMesh(int meshId) {
	Mesh* m = globalGraphicsState->meshs + meshId;
	return m;
}

Texture* getTexture(int textureId) {
	Texture* t = globalGraphicsState->textures + textureId;
	return t;
}

Font* getFont(int fontId) {
	Font* f = globalGraphicsState->fonts + fontId;
	return f;
}

Font* getFont(int fontId, int height) {
	Font* f = globalGraphicsState->fonts + fontId;

	if(f->height != height) {
		Font font;
		char* path = fontPaths[fontId];

		// font.fileBuffer = (char*)getPMemory(fileSize(path) + 1);
		char* fileBuffer = getTArray(char, fileSize(path) + 1);

		readFileToBuffer(fileBuffer, path);
		Vec2i size = vec2i(512,512);
		unsigned char* fontBitmapBuffer = (unsigned char*)getTMemory(size.x*size.y);
		unsigned char* fontBitmap = (unsigned char*)getTMemory(size.x*size.y*4);
		
		font.height = height;
		font.glyphStart = 32;
		font.glyphCount = 95;
		font.cData = (stbtt_bakedchar*)getPMemory(sizeof(stbtt_bakedchar)*font.glyphCount);
		stbtt_BakeFontBitmap((unsigned char*)fileBuffer, 0, font.height, fontBitmapBuffer, size.w, size.h, font.glyphStart, font.glyphCount, font.cData);
		for(int i = 0; i < size.w*size.h; i++) {
			fontBitmap[i*4] = fontBitmapBuffer[i];
			fontBitmap[i*4+1] = fontBitmapBuffer[i];
			fontBitmap[i*4+2] = fontBitmapBuffer[i];
			fontBitmap[i*4+3] = fontBitmapBuffer[i];
		}

		Texture tex;
		tex = loadTexture(fontBitmap, size.w, size.h, 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		font.tex = tex;

		globalGraphicsState->fonts[fontId] = font;
	}
	return f;
}

void bindShader(int shaderId) {
	int shader = globalGraphicsState->shaders[shaderId].program;
	glBindProgramPipeline(shader);
}

void pushUniform(uint shaderId, int shaderStage, uint uniformId, void* data, int count = 1) {
	Shader* s = globalGraphicsState->shaders + shaderId;
	ShaderUniform* uni = s->uniforms + uniformId;

	int i = shaderStage;
	bool setBothStages = false;
	if(shaderStage == 2) {
		i = 0;
		setBothStages = true;
	}

	for(; i < 2; i++) {
		uint stage = i == 0 ? s->vertex : s->fragment;
		uint location = i == 0 ? uni->vertexLocation : uni->fragmentLocation;

		switch(uni->type) {
			case UNIFORM_TYPE_MAT4: glProgramUniformMatrix4fv(stage, location, count, 1, (float*)data); break;
			case UNIFORM_TYPE_VEC4: glProgramUniform4fv(stage, location, count, (float*)data); break;
			case UNIFORM_TYPE_VEC3: glProgramUniform3fv(stage, location, count, (float*)data); break;
			case UNIFORM_TYPE_VEC2: glProgramUniform2fv(stage, location, count, (float*)data); break;
			case UNIFORM_TYPE_INT:  glProgramUniform1iv(stage, location, count, (int*)data); break;
			case UNIFORM_TYPE_FLOAT: glProgramUniform1fv(stage, location, count, (float*)data); break;
		}

		if(!setBothStages) break;
	}
};

void pushUniform(uint shaderId, int shaderStage, uint uniformId, float data) {
	pushUniform(shaderId, shaderStage, uniformId, &data);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, int data) {
	pushUniform(shaderId, shaderStage, uniformId, &data);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, float f0, float f1, float f2, float f3) {
	Vec4 d = vec4(f0, f1, f2, f3);
	pushUniform(shaderId, shaderStage, uniformId, &d);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, Vec4 v) {
	pushUniform(shaderId, shaderStage, uniformId, v.e);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, Vec3 v) {
	pushUniform(shaderId, shaderStage, uniformId, v.e);
};
void pushUniform(uint shaderId, int shaderStage, uint uniformId, Mat4 m) {
	pushUniform(shaderId, shaderStage, uniformId, m.e);
};

void getUniform(uint shaderId, int shaderStage, uint uniformId, float* data) {
	Shader* s = globalGraphicsState->shaders + shaderId;
	ShaderUniform* uni = s->uniforms + uniformId;

	uint stage = shaderStage == 0 ? s->vertex : s->fragment;
	uint location = shaderStage == 0 ? uni->vertexLocation : uni->fragmentLocation;

	glGetUniformfv(stage, location, data);
}



void drawRect(Rect r, Rect uv, Vec4 color, int texture, float texZ = -1) {	
	// TIMER_BLOCK();
	if(globalDebugState->timerBuffer == 0) {
		int stop = 234;
	}
	Rect cd = rectGetCenDim(r);

	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_MOD, cd.e);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_UV, uv.min.x, uv.max.x, uv.max.y, uv.min.y);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_COLOR, colorSRGB(color).e);
	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_TEXZ, texZ);

	uint tex[2] = {texture, texture};
	glBindTextures(0,2,tex);

	glDrawArraysInstancedBaseInstance(GL_TRIANGLE_STRIP, 0, 4, 1, 0);
}

uint createShader(const char* vertexShaderString, const char* fragmentShaderString, uint* vId, uint* fId) {
	*vId = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &vertexShaderString);
	*fId = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &fragmentShaderString);

	uint shaderId;
	glCreateProgramPipelines(1, &shaderId);
	glUseProgramStages(shaderId, GL_VERTEX_SHADER_BIT, *vId);
	glUseProgramStages(shaderId, GL_FRAGMENT_SHADER_BIT, *fId);

	return shaderId;
}

void ortho(Rect r) {
	r = rectGetCenDim(r);

	pushUniform(SHADER_QUAD, 0, QUAD_UNIFORM_CAMERA, r.e);
}

void lookAt(Vec3 pos, Vec3 look, Vec3 up, Vec3 right) {
	Mat4 view;
	viewMatrix(&view, pos, look, up, right);

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VIEW, view.e);
}

void perspective(float fov, float aspect, float n, float f) {
	Mat4 proj;
	projMatrix(&proj, fov, aspect, n, f);

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_PROJ, proj.e);
}

Vec2 getTextDim(char* text, Font* font) {
	Vec2 textDim = stbtt_GetTextDim(font->cData, font->height, font->glyphStart, text);
	return textDim;
}

void drawText(char* text, Font* font, Vec2 pos, Vec4 color, int vAlign = 0, int hAlign = 0, int shadow = 0, Vec4 shadowColor = vec4(0,0,0,1)) {
	int length = strLen(text);
	Vec2 textDim = getTextDim(text, font);
	pos.x -= vAlign*0.5f*textDim.w;
	pos.y -= hAlign*0.5f*textDim.h;

	Vec2 shadowOffset = vec2(shadow, -shadow);

	if(shadow != 0 && shadowColor == vec4(0,0,0,0)) shadowColor = vec4(0,0,0,1);

	pos = vec2((int)pos.x, roundInt((int)pos.y));
	Vec2 startPos = pos;
	for(int i = 0; i < length; i++) {
		char t = text[i];

		if(t == '\n') {
			pos.y -= font->height;
			pos.x = startPos.x;
			continue;
		}

		stbtt_aligned_quad q;
		stbtt_GetBakedQuad(font->cData, font->tex.dim.w, font->tex.dim.h, t-font->glyphStart, &pos.x, &pos.y, &q, 1);

		Rect r = rect(q.x0, q.y0, q.x1, q.y1);
		if(shadow > 0) {
			drawRect(rectAddOffset(r, shadowOffset), rect(q.s0,q.t0,q.s1,q.t1), shadowColor, font->tex.id);
		}
		drawRect(r, rect(q.s0,q.t0,q.s1,q.t1), color, font->tex.id);
	}
}

float getTextPos(char* text, int index, Font* font) {
	float result = 0;
	Vec2 pos = vec2(0,0);

	// no support for new line
	int length = strLen(text);
	for(int i = 0; i < length+1; i++) {
		char t = text[i];

		if(i == index) {
			result = pos.x;
			break;
		}

		stbtt_aligned_quad q;
		stbtt_GetBakedQuad(font->cData, font->tex.dim.w, font->tex.dim.h, t-font->glyphStart, &pos.x, &pos.y, &q, 1);
		Rect r = rect(q.x0, q.y0, q.x1, q.y1);
	}

	return result;
}

void drawCube(Vec3 trans, Vec3 scale, Vec4 color, float degrees, Vec3 rot) {
	// glBindTextures(0,1,&getTexture(TEXTURE_WHITE)->id);

	// Mat4 model = modelMatrix(trans, scale, degrees, rot);
	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODEL, model.e);
	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, &color);
	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, false);

	// glDrawArrays(GL_QUADS, 0, 6*4);

	glBindTextures(0,1,&getTexture(TEXTURE_WHITE)->id);

	Mesh* cubeMesh = getMesh(MESH_CUBE);
	glBindBuffer(GL_ARRAY_BUFFER, cubeMesh->bufferId);

	glVertexAttribPointer(0, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3) + sizeof(Vec2)));
	glEnableVertexAttribArray(2);

	Mat4 model = modelMatrix(trans, scale, degrees, rot);
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODEL, model.e);
	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, color.e);
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, colorSRGB(color).e);
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, false);

	glDrawArrays(GL_QUADS, 0, cubeMesh->vertCount);
	// glDrawElements(GL_QUADS, cubeMesh->elementCount, GL_UNSIGNED_INT, (void*)0);
}

void drawLine(Vec3 p0, Vec3 p1, Vec4 color) {
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

	glBindTextures(0,1,&getTexture(TEXTURE_WHITE)->id);

	Vec3 verts[] = {p0, p1};
	Vec2 quadUVs[] = {{0,0}, {0,1}};
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_UV, quadUVs[0].e, arrayCount(quadUVs));

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VERTICES, verts[0].e, arrayCount(verts));
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, colorSRGB(color).e);
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, true);

	glDrawArrays(GL_LINES, 0, arrayCount(verts));
}

void drawQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color) {
	Vec3 verts[] = {p0, p1, p2, p3};

	uint tex[2] = {getTexture(TEXTURE_WHITE)->id, 0};
	glBindTextures(0,2,tex);

	Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_UV, quadUVs[0].e, arrayCount(quadUVs));

	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VERTICES, verts[0].e, arrayCount(verts));
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, colorSRGB(color).e);
	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, true);

	glDrawArrays(GL_QUADS, 0, arrayCount(verts));
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

uint createSampler(float ani, int wrapS, int wrapT, int magF, int minF, int wrapR = GL_CLAMP_TO_EDGE) {
	uint result;
	glCreateSamplers(1, &result);

	glSamplerParameteri(result, GL_TEXTURE_MAX_ANISOTROPY_EXT, ani);
	glSamplerParameteri(result, GL_TEXTURE_WRAP_S, wrapS);
	glSamplerParameteri(result, GL_TEXTURE_WRAP_T, wrapT);
	glSamplerParameteri(result, GL_TEXTURE_MAG_FILTER, magF);
	glSamplerParameteri(result, GL_TEXTURE_MIN_FILTER, minF);

	glSamplerParameteri(result, GL_TEXTURE_WRAP_R, wrapR);

	return result;
}



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

const char* textureFilePaths[BX_Size] = {
	"..\\data\\minecraft textures\\none.png",
	"..\\data\\minecraft textures\\water.png",
	"..\\data\\minecraft textures\\sand.png",
	"..\\data\\minecraft textures\\grass_top.png",
	"..\\data\\minecraft textures\\grass_side.png",
	"..\\data\\minecraft textures\\grass_bottom.png",
	"..\\data\\minecraft textures\\stone.png",
	"..\\data\\minecraft textures\\snow.png",
	"..\\data\\minecraft textures\\tree_log_top.png",
	"..\\data\\minecraft textures\\tree_log_side.png",
	"..\\data\\minecraft textures\\leaves.png",
	"..\\data\\minecraft textures\\glass.png",
	"..\\data\\minecraft textures\\glowstone.png",
	"..\\data\\minecraft textures\\pumpkin_top.png",
	"..\\data\\minecraft textures\\pumpkin_side.png",
	"..\\data\\minecraft textures\\pumpkin_bottom.png",
};

enum BlockTextures2 {
	BX2_None = 0,
	BX2_Leaves,

	BX2_Size,
};

const char* textureFilePaths2[BX2_Size] = {
	"..\\data\\minecraft textures\\none.png",
	"..\\data\\minecraft textures\\leaves.png",
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



// #define VIEW_DISTANCE 4096 // 64
// #define VIEW_DISTANCE 3072 // 32

// #define VIEW_DISTANCE 2500 // 32
// #define VIEW_DISTANCE 2048 // 32
// #define VIEW_DISTANCE 1024 // 16
#define VIEW_DISTANCE 512  // 8
// #define VIEW_DISTANCE 256 // 4
// #define VIEW_DISTANCE 128 // 2

#define USE_MALLOC 0

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

void generateVoxelMeshThreaded(void* data) {
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

	int cacheId = THREADING ? getThreadQueueId(globalThreadQueue) : 1;

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
					if(!threadData[i].inProgress) {
						threadData[i] = {m, voxelHash, voxelHashSize, true};
						// data = threadData + i;
						
						atomicAdd(&m->activeMaking);
						threadQueueAdd(globalThreadQueue, makeMeshThreaded, &threadData[i]);

						break;
					}
				}
			} else {
				threadData[1] = {m, voxelHash, voxelHashSize, true};
				makeMeshThreaded(&threadData[1]);
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
uchar* getBlockFromVoxel(VoxelNode** voxelHash, int voxelHashSize, Vec3i voxel) {
	VoxelMesh* vm = getVoxelMesh(voxelHash, voxelHashSize, voxelToMesh(voxel));
	Vec3i localCoord = voxelToLocalVoxel(voxel);
	uchar* block = &vm->voxels[voxelArray(localCoord.x, localCoord.y, localCoord.z)];

	return block;
}

// coord -> block
uchar* getBlockFromCoord(VoxelNode** voxelHash, int voxelHashSize, Vec3 coord) {
	return getBlockFromVoxel(voxelHash, voxelHashSize, coordToVoxel(coord));
}

// voxel -> lighting
uchar* getLightingFromVoxel(VoxelNode** voxelHash, int voxelHashSize, Vec3i voxel) {
	VoxelMesh* vm = getVoxelMesh(voxelHash, voxelHashSize, voxelToMesh(voxel));
	Vec3i localCoord = voxelToLocalVoxel(voxel);
	uchar* block = &vm->lighting[voxelArray(localCoord.x, localCoord.y, localCoord.z)];

	return block;
}

// coord -> lighting
uchar* getLightingFromCoord(VoxelNode** voxelHash, int voxelHashSize, Vec3 coord) {
	return getLightingFromVoxel(voxelHash, voxelHashSize, coordToVoxel(coord));
}

void setupVoxelUniforms(Vec4 camera, uint texUnit1, uint texUnit2, uint faceUnit, Mat4 view, Mat4 proj, Vec3 fogColor, Vec3 trans = vec3(0,0,0), Vec3 scale = vec3(1,1,1), Vec3 rotation = vec3(0,0,0)) {
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


#define dcGetStructAndIncrement(structType) \
	Draw_Command_##structType dc = *((Draw_Command_##structType*)drawListIndex); \
	drawListIndex += sizeof(Draw_Command_##structType); \

int stateSwitch(int state) {
	switch(state) {
		case STATE_CULL: return GL_CULL_FACE;
		case STATE_SCISSOR: return GL_SCISSOR_TEST;
	}
	return 0;
}

void executeCommandList(DrawCommandList* list) {
	// TIMER_BLOCK();

	char* drawListIndex = (char*)list->data;
	for(int i = 0; i < list->count; i++) {
		int command = *((int*)drawListIndex);
		drawListIndex += sizeof(int);

		switch(command) {
			case Draw_Command_Cube_Type: {
				dcGetStructAndIncrement(Cube);
				drawCube(dc.trans, dc.scale, dc.color, dc.degrees, dc.rot);
			} break;

			case Draw_Command_Line_Type: {
				dcGetStructAndIncrement(Line);
				drawLine(dc.p0, dc.p1, dc.color);
			} break;

			case Draw_Command_Quad_Type: {
				dcGetStructAndIncrement(Quad);
				drawQuad(dc.p0, dc.p1, dc.p2, dc.p3, dc.color);
			} break;

			case Draw_Command_State_Type: {
				dcGetStructAndIncrement(State);

				switch(dc.state) {
					case STATE_POLYGONMODE: {
						int m;
						switch(dc.value) {
							case POLYGON_MODE_FILL: m = GL_FILL; break;
							case POLYGON_MODE_LINE: m = GL_LINE; break;
							case POLYGON_MODE_POINT: m = GL_POINT; break;
						}
						glPolygonMode(GL_FRONT_AND_BACK, m);
					} break;

					case STATE_LINEWIDTH: glLineWidth(dc.value); break;

					default: {} break;
				}
			} break;

			case Draw_Command_Enable_Type: {
				dcGetStructAndIncrement(Int);

				int m = stateSwitch(dc.state);
				glEnable(m);
			} break;			

			case Draw_Command_Disable_Type: {
				dcGetStructAndIncrement(Int);

				int m = stateSwitch(dc.state);
				glDisable(m);
			} break;	

			case Draw_Command_Rect_Type: {
				dcGetStructAndIncrement(Rect);
				int texture = dc.texture == -1 ? getTexture(TEXTURE_WHITE)->id : dc.texture;
				drawRect(dc.r, dc.uv, dc.color, texture, dc.texZ-1);
			} break;

			case Draw_Command_Text_Type: {
				dcGetStructAndIncrement(Text);
				drawText(dc.text, dc.font, dc.pos, dc.color, dc.vAlign, dc.hAlign, dc.shadow, dc.shadowColor);
			} break;

			case Draw_Command_Scissor_Type: {
				dcGetStructAndIncrement(Scissor);
				Rect r = dc.rect;
				Vec2 dim = rectGetDim(r);
				glScissor(r.min.x, r.min.y, dim.x, dim.y);
			} break;

			default: {} break;
		}
	}
}


struct Bomb {
	Vec3 pos;
	Vec3 size;
	Vec3 vel;
	Vec3 acc;

	Vec3 dir;
	bool active;
	Vec3 exploded;
};

struct Entity;
struct EntityList {
	Entity* e;
	int size;
};


struct Camera {
	Vec3 pos;
	Vec3 look;
	Vec3 up;
	Vec3 right;
};

Camera getCamData(Vec3 pos, Vec3 rot, Vec3 offset = vec3(0,0,0), Vec3 gUp = vec3(0,0,1), Vec3 startDir = vec3(0,1,0)) {
	Camera c;
	c.pos = pos + offset;
	c.look = startDir;
	rotateVec3(&c.look, rot.x, gUp);
	rotateVec3(&c.look, rot.y, normVec3(cross(gUp, c.look)));
	c.up = normVec3(cross(c.look, normVec3(cross(gUp, c.look))));
	c.right = normVec3(cross(gUp, c.look));
	c.look = -c.look;

	return c;
}


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



struct GuiInput {
	Vec2 mousePos;
	int mouseWheel;
	bool mouseClick, mouseDown;
	bool escape, enter, space, backSpace, del, home, end;
	bool left, right, up, down;
	bool shift, ctrl;
	char* charList;
	int charListSize;
};

char* guiColorStrings[] = {
	"panelColor",
	"regionColor",
	"sectionColor",
	"selectionColor",
	"resizeButtonColor",
	"textColor",
	"shadowColor",
	"hoverColorAdd",
	"switchColorOn",
	"switchColorOff",
	"sliderColor",
	"sliderHoverColorAdd",
	"cursorColor",
};

char* guiSettingStrings[] = {
	"offsets",
	"border",
	"minSize",
	"sliderSize",
	"sliderResetDistance",
	"fontOffset",
	"sectionOffset",
	"fontHeightOffset",
	"sectionIndent",
	"scrollBarWidth",
	"scrollBarSliderSize",
	"cursorWidth",
	"textShadow",
};

union GuiColors {
	struct {
		Vec4 panelColor;
		Vec4 regionColor;
		Vec4 sectionColor;
		Vec4 selectionColor;
		Vec4 resizeButtonColor;
		Vec4 textColor;
		Vec4 shadowColor;
		Vec4 hoverColorAdd;
		Vec4 switchColorOn;
		Vec4 switchColorOff;
		Vec4 sliderColor;
		Vec4 sliderHoverColorAdd;
		Vec4 cursorColor;
	};
	Vec4 e[13];
};

struct GuiSettings {
	Vec2i border;
	Vec2i offsets;
	Vec2i minSize;
	int sliderSize;
	float sliderResetDistance;
	float fontOffset;
	float sectionOffset;
	float fontHeightOffset;
	float sectionIndent;
	int scrollBarWidth;
	int scrollBarSliderSize;
	int cursorWidth;
	int textShadow;
};

struct Gui;
bool guiCreateSettingsFile();
void guiSave(Gui* gui, int element, int slot);
void guiLoad(Gui* gui, int element, int slot);
void guiSaveAll(Gui* gui, int slot);
void guiLoadAll(Gui* gui, int slot);

struct Gui {
	GuiColors colors;
	GuiSettings settings;
	Vec2 cornerPos;
	Vec2 panelStartDim;

	GuiInput input;
	Font* font;
	Vec2i screenRes;

	Vec2 startPos;
	float panelWidth;
	float fontHeight;

	int currentId;
	int activeId;
	int wantsToBeActiveId;
	int hotId;

	Vec2 currentPos;
	Vec2 currentDim;
	Vec2 lastPos;
	Vec2 lastDim;
	Vec2 lastMousePos;
	Vec2 mouseDelta;

	float mainScrollAmount;
	int mainScrollHeight;

	bool columnMode;
	float columnArray[10];
	float columnSize;
	int columnIndex;

	Rect scissorStack[8];
	int scissorStackSize;

	int scrollStackIndexX;
	int scrollStackIndexY[4];
	float scrollStack[4][4];
	float scrollStartY[4];
	float scrollEndY[4];
	bool showScrollBar[4];
	float scrollAnchor;
	Vec2 scrollMouse;
	float scrollValue;

	void* originalPointer;
	int textBoxIndex;
	char textBoxText[50];
	int textBoxTextSize;
	int textBoxTextCapacity;
	int selectionAnchor;

	bool secondMode;

	float heightStack[4];
	int heightStackIndex;

	void init(Rect panelRect, int saveSlot) {
		*this = {};

		if(guiCreateSettingsFile() || saveSlot == -1) {
			Vec3 mainColor = vec3(280,0.7f,0.3f);

			colors.panelColor 		= vec4(hslToRgb(mainColor), 1);
			colors.regionColor 		= vec4(hslToRgb(mainColor + vec3(0,0,-0.1f)), 1);
			colors.sectionColor 		= vec4(hslToRgb(mainColor + vec3(45,0,0)), 1);
			colors.resizeButtonColor 	= vec4(hslToRgb(mainColor + vec3(-45,-0.1f,0)), 1);
			colors.selectionColor 	= vec4(hslToRgb(mainColor + vec3(-180,0,0)), 1);

			colors.switchColorOn 		= vec4(0,0.3f,0,0);
			colors.switchColorOff 	= vec4(0.3f,0,0,0);

			colors.textColor 			= vec4(0.9f, 0.9f, 0.9f, 1);
			colors.hoverColorAdd 		= vec4(0.2f,0.2f,0.2f,0);
			colors.sliderHoverColorAdd= colors.hoverColorAdd;
			colors.sliderColor 		= vec4(0.6f,0.6f,0.6f,0.8f);
			colors.cursorColor 		= vec4(1,1,1,1);

			settings.sliderSize = 4;
			settings.sliderResetDistance = 100;
			settings.offsets = vec2i(1,1);
			settings.border = vec2i(4,-4);
			settings.minSize = vec2i(100,100);
			settings.fontHeightOffset = 1.0f;
			settings.fontOffset = 0.15f;
			settings.sectionOffset = 1.15f;
			settings.sectionIndent = 0.08f;
			settings.scrollBarWidth = 20;
			settings.scrollBarSliderSize = 30;
			settings.cursorWidth = 1;

			cornerPos = rectGetUL(panelRect);
			panelStartDim = rectGetDim(panelRect);
			panelStartDim.h += settings.border.h*2;
		} else {
			guiLoadAll(this, saveSlot);
		}
	}

	void heightPush(float heightOffset) { heightStack[++heightStackIndex] = heightOffset; }
	void heightPop() { heightStackIndex--; }

	void doScissor(Rect r) {
		dcScissor(r);
	}

	void setScissor(bool m) {
		if(m) dcEnable(STATE_SCISSOR);
		else dcDisable(STATE_SCISSOR);
	}

	Rect getCurrentScissor() {
		Rect result = scissorStack[scissorStackSize-1];
		return result;
	}

	Rect scissorPush(Rect newRect) {
		if(scissorStackSize == 0) {
			scissorStack[scissorStackSize++] = newRect;
			doScissor(scissorRect(newRect));

			return newRect;
		}

		if((newRect.max.x < newRect.min.x || newRect.max.y < newRect.min.y) || 
			newRect == rect(0,0,0,0)) {
			doScissor(scissorRect(rect(0,0,0,0)));
			scissorStack[scissorStackSize++] = rect(0,0,0,0);

			return rect(0,0,0,0);
		}

		Rect currentRect = getCurrentScissor();
		Rect intersection;
		rectGetIntersection(&intersection, newRect, currentRect);

		doScissor(scissorRect(intersection));

		scissorStack[scissorStackSize++] = intersection;
		return intersection;
	}

	// Rect scissorPushWidth(float width) {
	// 	if(width <= 0) {
	// 		Rect result = scissorPush(rect(0,0,0,0));
	// 		return result;
	// 	}
	// 	else {
	// 		Rect currentRect = getCurrentScissor();
	// 		currentRect 
	// 		Rect result = scissorPush();
	// 		return result;
	// 	}
	// }

	void scissorPop() {
		scissorStackSize--;
		scissorStackSize = clampMin(scissorStackSize, 0);
		if(scissorStackSize > 0) doScissor(scissorRect(scissorStack[scissorStackSize-1]));
	}

	Rect scissorRect(Rect r) {
		Rect scissorRect = {r.min.x, r.min.y+screenRes.h, r.max.x, r.max.y+screenRes.h};
		return scissorRect;
	}

	void drawText(char* text, int align = 1) {
		Rect region = getCurrentRegion();
		scissorPush(region);

		Vec2 textPos = rectGetCen(region) + vec2(0,fontHeight*settings.fontOffset);
		if(align == 0) textPos.x -= rectGetDim(region).w*0.5f;
		else if(align == 2) textPos.x += rectGetDim(region).w*0.5f;

		dcText(text, font, textPos, colors.textColor, align, 1, settings.textShadow, colors.shadowColor);		

		scissorPop();
	}

	void drawText(char* text, int align, Rect region) {
		scissorPush(region);

		Vec2 textPos = rectGetCen(region) + vec2(0,fontHeight*settings.fontOffset);
		if(align == 0) textPos.x -= rectGetDim(region).w*0.5f;
		else if(align == 2) textPos.x += rectGetDim(region).w*0.5f;

		dcText(text, font, textPos, colors.textColor, align, 1, settings.textShadow, colors.shadowColor);		

		scissorPop();
	}

	int getTextWidth(char* text) {
		Vec2 textDim = getTextDim(text, font);
		return textDim.w;
	}	

	void drawRect(Rect r, Vec4 color, bool scissor = false) {
		if(scissor) scissorPush(getCurrentRegion());
		dcRect(r, rect(0,0,1,1), color, (int)getTexture(TEXTURE_WHITE)->id);
		if(scissor) scissorPop();
	}

	void drawTextBox(Rect region, char* text, Vec4 bgColor, int align = 0) {
		drawRect(region, bgColor, false);
		drawText(text, align, region);
	}

	void start(GuiInput guiInput, Font* font, Vec2i res) {
		this->font = font;
		fontHeight = font->height;
		lastMousePos = input.mousePos;
		input = guiInput;
		input.mousePos.y *= -1;
		mouseDelta = input.mousePos - lastMousePos;
		screenRes = res;
		currentId = 0;
		scrollStackIndexX = 0;
		for(int i = 0; i < 4; i++) scrollStackIndexY[i] = 0;
		hotId = wantsToBeActiveId;
		wantsToBeActiveId = 0;

		for(int i = 0; i < arrayCount(heightStack); i++) heightStack[i] = 1;
		heightStackIndex = 0;

		Rect background = rectULDim(cornerPos, vec2(panelWidth+settings.border.x*2, -(lastPos.y-cornerPos.y)+lastDim.h - settings.border.y));

		incrementId();
		// drag window
		{
			Vec2 dragDelta = vec2(0,0);
			Vec2 oldPos = cornerPos;
			if(!input.ctrl && drag(background, &dragDelta)) {
				cornerPos += dragDelta;
			}

			// clamp(&cornerPos, rect(0, -res.y, res.x - rectGetDim(background).w+1, 0.5f));
			clamp(&cornerPos, rect(0, -res.y + rectGetDim(background).h, res.x - rectGetDim(background).w+1, 0.5f));
			background = rectAddOffset(background, cornerPos - oldPos);
		}

		// resize window
		Rect resizeRegion;
		{
			resizeRegion = rect(rectGetDR(background)-vec2(settings.border.x*2,0), rectGetDR(background)+vec2(0,-settings.border.y*2));
			Vec2 dragDelta = vec2(0,0);
			int oldMainScrollHeight = panelStartDim.h;
			float oldPanelWidth = panelStartDim.w;

			if(input.ctrl) {
				if(drag(background, &dragDelta)) {
					panelStartDim.h += -dragDelta.y;
					panelStartDim.w += dragDelta.x;
				}
			}
			{
				incrementId();
				if(drag(resizeRegion, &dragDelta)) {
					panelStartDim.h += -dragDelta.y;
					panelStartDim.w += dragDelta.x;
				}
			}


			// garbage
			if(dragDelta != vec2(0,0)) {
				Rect screenRect = rect(0,-screenRes.h,screenRes.w,0);
				dragDelta.x = clampMax(dragDelta.x, screenRect.max.x - background.max.x +1);
				dragDelta.y = clampMin(dragDelta.y, screenRect.min.y - background.min.y);
				panelStartDim.w = oldPanelWidth + dragDelta.x;
				panelStartDim.h = oldMainScrollHeight - dragDelta.y;
			}


		
			panelStartDim.h = clamp(panelStartDim.h, settings.minSize.y, screenRes.h+settings.border.h*2+1);
			panelStartDim.w = clamp(panelStartDim.w, settings.minSize.x, screenRes.w-settings.border.w*2+1);

			Vec2 dragAfterClamp = vec2(panelStartDim.w - oldPanelWidth, oldMainScrollHeight - panelStartDim.h);
			if(showScrollBar[0] == false) dragAfterClamp.y = 0;
			resizeRegion = rectAddOffset(resizeRegion, dragAfterClamp);
			background = rectExpand(background, 0, dragAfterClamp.y, dragAfterClamp.x, 0);

			clamp(&cornerPos, rect(0, -res.y + rectGetDim(background).h, res.x - rectGetDim(background).w+1, 0.5f));
		}

		setScissor(true);
		scissorPush(background);

		drawRect(background, colors.panelColor, false);
		drawRect(resizeRegion, colors.resizeButtonColor, false);

		startPos = cornerPos + settings.border;
		// currentPos = startPos + vec2(0, getDefaultYOffset());
		// getDefaultHeight();
		defaultHeight();
		currentPos = startPos + vec2(0, getDefaultYOffset());
		
		panelWidth = panelStartDim.w;
		mainScrollHeight = panelStartDim.h;

		// panel scrollbar
		{
			beginScroll(mainScrollHeight, &mainScrollAmount);
		}
	}

	void end() {
		// terrible hack
		currentPos.y = currentPos.y - (currentDim.h - getDefaultHeightEx());
		currentDim.h = getDefaultHeightEx();

		// panel scrollbar
		{
			endScroll();
		}

		scissorPop();
		setScissor(false);
	}

	float getDefaultHeightEx(){return fontHeight*settings.fontHeightOffset;};
	float getDefaultHeight() {
		float height = fontHeight*settings.fontHeightOffset;
		float stackValue = heightStack[heightStackIndex];

		// switch from multiplier to pixels if over threshhold
		if(stackValue > 10) height = stackValue;
		else height *= stackValue;

		return height;
	};
	float getDefaultWidth(){return panelWidth;};
	float getDefaultYOffset(){return currentDim.h+settings.offsets.h;};
	float getYOffset(){return currentDim.h+settings.offsets.h;};
	// float getDefaultYOffset(){return getDefaultHeight()+settings.offsets.h;};
	Vec2 getDefaultPos(){return vec2(startPos.x, currentPos.y - getDefaultYOffset());};
	void advancePosY() 	{currentPos.y -= getDefaultYOffset();};
	void defaultX() 	{currentPos.x = startPos.x;};
	void defaultHeight(){currentDim.h = getDefaultHeight();};
	void defaultWidth() {currentDim.w = getDefaultWidth();};

	void defaultValues() {
		advancePosY();
		defaultX();
		defaultHeight();
		defaultWidth();
	}

	void setLastPosition() {
		lastPos = currentPos;
		lastDim = currentDim;
	}

	void incrementId() {currentId++;};

	bool pre() {
		incrementId();

		if(columnMode) {
			if(columnIndex == 0) {
				defaultValues();
			} else {
				defaultHeight();
				defaultWidth();
				currentPos.x += lastDim.w + settings.offsets.w;
			}
			currentDim.w = columnArray[columnIndex];

			columnIndex++;
			if(columnIndex == columnSize) columnMode = false;
		} else {
			defaultValues();
		}

		setLastPosition();

		if(scissorStack[scissorStackSize-1] == rect(0,0,0,0)) return false;
		else if(currentDim.w <= 0) return false;
		else return true;
	}


	// no usage yet
	void post() {
	}

	void advanceY(float v) {
		if(v >= 5) currentPos.y -= v;
		else currentPos.y -= getDefaultYOffset()*v;
		// else currentPos.y -= getYOffset();
	}

	void div(float* c, int size) {
		int elementCount = size;
		float dynamicSum = 0;
		int flowCount = 0;
		float staticSum = 0;
		int staticCount = 0;
		for(int i = 0; i < size; i++) {
			float val = c[i];

			if(val == 0) { 			// float element
				flowCount++;
			} else if(val <= 1) { 	// dynamic element
				dynamicSum += val;
			} else { 				// static element
				staticSum += val;
				staticCount++;
			}
		}

		if(flowCount) {
			float flowVal = abs(dynamicSum-1)/(float)flowCount;
			for(int i = 0; i < size; i++)
				if(c[i] == 0) c[i] = flowVal;
		}

		float totalWidth = getDefaultWidth() - staticSum - settings.offsets.w*(elementCount-1);
		for(int i = 0; i < size; i++) {
			float val = c[i];
			if(val <= 1) {
				columnArray[i] = val * totalWidth;
			} else {
				columnArray[i] = val;
			}
		}

		columnSize = size;
		columnMode = true;
		columnIndex = 0;
	}
	void div(Vec2 v) {div(v.e, 2);};
	void div(Vec3 v) {div(v.e, 3);};
	void div(Vec4 v) {div(v.e, 4);};
	void div(float a, float b) {div(vec2(a,b));};
	void div(float a, float b, float c) {div(vec3(a,b,c));};
	void div(float a, float b, float c, float d) {div(vec4(a,b,c,d));};

	Rect getCurrentRegion() {
		Rect result = rectULDim(currentPos, currentDim);
		return result;
	}

	void empty() {
		pre();
		post();
	}

	void label(char* text, int align = 1, Vec4 bgColor = vec4(0,0,0,0)) {
		if(!pre()) return;

		if(bgColor != vec4(0,0,0,0)) drawRect(getCurrentRegion(), bgColor, false);
		drawText(text, align);
		post();
	}

	bool beginSection(char* text, bool* b) {
		heightPush(settings.sectionOffset);
		defaultWidth();
		div(vec2(getDefaultHeight(), 0)); switcher("", b); label(text, 1, colors.sectionColor);
		heightPop();

		float indent = panelWidth*settings.sectionIndent;
		startPos.x += indent*0.5f;
		panelWidth -= indent;

		return *b;
	}

	void endSection() {
		panelWidth = panelWidth*(1/(1-settings.sectionIndent));
		startPos.x -= panelWidth*settings.sectionIndent*0.5f;
	}

	void beginScroll(int scrollHeight, float* scrollAmount) {
		scrollEndY[scrollStackIndexX] = currentPos.y - scrollHeight - settings.offsets.h;

		Rect r = rectULDim(getDefaultPos(), vec2(panelWidth, scrollHeight));

		float scrollValue = scrollStack[scrollStackIndexX][scrollStackIndexY[scrollStackIndexX]];
		float diff = (scrollValue - scrollHeight - settings.offsets.y);
		if(diff > 0) {
			panelWidth -= settings.scrollBarWidth + settings.offsets.w;

			currentPos.y += *scrollAmount * diff;
			Rect scrollRect = r;
			scrollRect.min.x += panelWidth + settings.offsets.w;
			// r.min.x += panelWidth;

			Rect regCen = rectGetCenDim(scrollRect);
			float sliderHeight = regCen.dim.h - diff;
			sliderHeight = clampMin(sliderHeight, settings.scrollBarSliderSize);

			slider(scrollAmount, 0, 1, true, scrollRect, sliderHeight);

			currentPos.y += getDefaultYOffset();

			Rect scissorRect = r;
			scissorRect.max.x -= settings.offsets.w + settings.scrollBarWidth;
			scissorPush(scissorRect);
		} else {
			incrementId();
			scissorPush(r);
		}

		showScrollBar[scrollStackIndexX] = diff > 0 ? true : false;

		scrollStartY[scrollStackIndexX] = currentPos.y;

		scrollStackIndexX++;
	};

	void endScroll() {
		scrollStackIndexX--;

		scissorPop();

		scrollStack[scrollStackIndexX][scrollStackIndexY[scrollStackIndexX]++] = -((currentPos.y) - scrollStartY[scrollStackIndexX]);
		// float h = currentDim.h - getDefaultHeightEx();
		// scrollStack[scrollStackIndexX][scrollStackIndexY[scrollStackIndexX]++] = -((currentPos.y + currentDim.h) - scrollStartY[scrollStackIndexX]);
		// scrollStack[scrollStackIndexX][scrollStackIndexY[scrollStackIndexX]++] = -((currentPos.y + h) - scrollStartY[scrollStackIndexX]);

		if(showScrollBar[scrollStackIndexX]) {
			currentPos.y = scrollEndY[scrollStackIndexX];
			// currentPos.y = scrollEndY[scrollStackIndexX] + currentDim.h;
			panelWidth += settings.scrollBarWidth + settings.offsets.w;
		}

		setLastPosition();
	}

	bool getMouseOver(Vec2 mousePos, Rect region, bool noScissor = false) {
		bool overScissorRegion = noScissor ? true : pointInRect(mousePos, scissorStack[scissorStackSize-1]);
		bool mouseOver = pointInRect(mousePos, region) && overScissorRegion;
		return mouseOver;
	}

	bool getActive() {
		bool active = (activeId == currentId);
		return active;
	}

	bool getHot() {
		bool hot = (hotId == currentId);
		return hot;
	}

	bool setActive(bool mouseOver, int releaseType = 0) {
		if((hotId == currentId) && activeId == 0 && (mouseOver && input.mouseClick)) activeId = currentId;
		else if(activeId == currentId) {
			if( releaseType == 0 || 
				(releaseType == 1 && !input.mouseDown)) {
				activeId = 0;
				secondMode = false;
			}
		}

		if(mouseOver) {
			wantsToBeActiveId = max(wantsToBeActiveId, currentId);
		} 


		bool active = getActive();
		return active;
	}

	Vec4 getColorAdd(bool active, bool mouseOver, int type = 0) {
		if(type == 2) {
			Vec4 color = active || (mouseOver && activeId == 0) ? colors.hoverColorAdd : vec4(0,0,0,0);
			return color;
		}

		Vec4 colorAdd = type == 0 ? colors.hoverColorAdd : colors.sliderHoverColorAdd;
		Vec4 color;
		if(active) color = colorAdd*2;
		else if(mouseOver && activeId == 0) color = colorAdd;
		else color = {};

		return color;
	}

	bool drag(Rect region, Vec2* dragDelta, Vec4 color = vec4(0,0,0,0)) {
		// incrementId();

		bool mouseOver = getMouseOver(input.mousePos, region, true);
		bool active = setActive(mouseOver, 1);

		if(active) {
			*dragDelta = mouseDelta;
		}

		if(color != vec4(0,0,0,0)) drawRect(region, color);

		post();
		return active;
	}

	bool button(char* text, int switcher = 0, Vec4 bgColor = vec4(0,0,0,0)) {
		if(!pre()) return false;

		Rect region = getCurrentRegion();
		bool mouseOver = getMouseOver(input.mousePos, region);
		bool active = setActive(mouseOver);
		Vec4 colorAdd = getColorAdd(active, mouseOver);

		Vec4 finalColor = (bgColor==vec4(0,0,0,0) ? colors.regionColor:bgColor) + colorAdd;

		if(switcher) {
			if(switcher == 2) finalColor += colors.switchColorOn;
			else finalColor += colors.switchColorOff;
		}

		drawRect(region, finalColor);
		drawText(text);

		post();
		return active;
	}

	bool switcher(char* text, bool* value) {
		bool active = false;
		if(button(text, (int)(*value) + 1)) {
			*value = !(*value);
			active = true;
		}

		return active;
	}

	bool slider(void* value, float min, float max, bool typeFloat = true, Rect reg = rect(0,0,0,0), float sliderS = 0) {
		if(!pre()) return false;

		bool verticalSlider = sliderS == 0 ? false : true;
		Vec2 valueRange = verticalSlider ? vec2(max,min) : vec2(min,max);

		Rect region = reg == rect(0,0,0,0) ? getCurrentRegion() : reg;
		scissorPush(region);

		float sliderSize = verticalSlider ? sliderS : settings.sliderSize;
		if(typeFloat) sliderSize = verticalSlider ? sliderS : settings.sliderSize;
		else {
			sliderSize = rectGetDim(region).w * 1/(float)((int)roundInt(max)-(int)roundInt(min) + 1);
			sliderSize = clampMin(sliderSize, settings.sliderSize);
		}

		float calc = typeFloat ? *((float*)value) : *((int*)value);
		Vec2 sliderRange = verticalSlider ? vec2(region.min.y, region.max.y-sliderSize) : 
											vec2(region.min.x, region.max.x-sliderSize);
		float sliderPos = mapRangeClamp(calc, valueRange, sliderRange);
		Rect sliderRegion = verticalSlider ? rect(region.min.x, sliderPos, region.max.x, sliderPos+sliderSize) :
											 rect(sliderPos, region.min.y, sliderPos+sliderSize, region.max.x);
		
		bool activeBefore = getActive();
		bool mouseOverSlider = getMouseOver(input.mousePos, sliderRegion);
		bool active = setActive(mouseOverSlider, 1);
		Vec4 colorAdd = getColorAdd(active, mouseOverSlider, 1);

		if(!activeBefore && active) {
			scrollAnchor = verticalSlider ? input.mousePos.y - sliderPos : input.mousePos.x - sliderPos;
			scrollMouse = input.mousePos;
			scrollValue = typeFloat ? *(float*)value : *(int*)value;
		}

		bool resetSlider = false;
		if(active) {
			if(verticalSlider) {
				calc = mapRange(input.mousePos.y - scrollAnchor, sliderRange, valueRange);
				calc = clamp(calc, valueRange.y, valueRange.x);
			} else {
				calc = mapRangeClamp(input.mousePos.x - scrollAnchor, sliderRange, valueRange);
			}

			float dist = verticalSlider ? abs(scrollMouse.x - input.mousePos.x) : 
										  abs(scrollMouse.y - input.mousePos.y);
			if(dist > settings.sliderResetDistance) calc = scrollValue;

			if(!verticalSlider && typeFloat) {
				if(input.ctrl) calc = roundInt(calc);
				else if(input.shift) calc = roundFloat(calc, 10);
			}

			if(!typeFloat) calc = roundInt(calc);
			
			sliderPos = mapRangeClamp(calc, valueRange, sliderRange);
			sliderRegion = verticalSlider ? rect(region.min.x, sliderPos, region.max.x, sliderPos+sliderSize) :
											rect(sliderPos, region.min.y, sliderPos+sliderSize, region.max.x);
		}

		if(typeFloat) *(float*)value = calc;
		else *(int*)value = calc;

		// draw background
		drawRect(region, colors.regionColor);

		// draw slider
		drawRect(sliderRegion, colors.sliderColor + colorAdd);

		// draw text
		char* text;
		if(verticalSlider) text = "";
		if(typeFloat) text = fillString("%f", *((float*)value));
		else text = fillString("%i", *((int*)value));
		drawText(text);

		scissorPop();

		post();
		return active;
	}

	bool slider(int* value, float min, float max) {
		return slider(value, min, max, false);
	}

	bool slider2d(float v[2], float min, float max) {
		bool active1 = slider(&v[0], min, max);
		bool active2 = slider(&v[1], min, max);
		return (active1 || active2);
	}

	// really?
	bool slider2d(float v[2], float min, float max, float min2, float max2) {
		bool active1 = slider(&v[0], min, max);
		bool active2 = slider(&v[1], min2, max2);
		return (active1 || active2);
	}


	bool textBox(void* value, int type, int textSize, int textCapacity) {
		if(!pre()) return false;

		bool activeBefore = getActive();

		Rect region = getCurrentRegion();
		bool mouseOver = getMouseOver(input.mousePos, region);
		bool active = setActive(mouseOver, 2);
		Vec4 colorAdd = getColorAdd(active, mouseOver, 2);

		if(!activeBefore && active) {
			int textLength;
			if(type == 0) {
				textLength = textSize == 0 ? strLen((char*)value) : textSize;
				strCpy(textBoxText, (char*)value, textLength);
			} else if(type == 1) {
				intToStr(textBoxText, *((int*)value));
				textLength = strLen(textBoxText);
			} else {
				floatToStr(textBoxText, *((float*)value));
				textLength = strLen(textBoxText);
			}

			originalPointer = value;
			textBoxIndex = textLength;
			textBoxTextSize = textLength;
			textBoxTextCapacity = textCapacity;

			selectionAnchor = -1;
		}

		drawRect(region, colors.regionColor + colorAdd);

		if(active) {
			int oldTextBoxIndex = textBoxIndex;
			if(input.left) textBoxIndex--;
			if(input.right) textBoxIndex++;
			clampInt(&textBoxIndex, 0, textBoxTextSize);

			if(input.ctrl) {
				if(input.left) textBoxIndex = strFindBackwards(textBoxText, ' ', textBoxIndex);
				else if(input.right) {
					int foundIndex = strFind(textBoxText, ' ', textBoxIndex);
					if(foundIndex == 0) textBoxIndex = textBoxTextSize;
					else textBoxIndex = foundIndex - 1;
				}
			}
			if(input.home) textBoxIndex = 0;
			if(input.end) textBoxIndex = textBoxTextSize;

			if(selectionAnchor == -1) {
				if(input.shift && (oldTextBoxIndex != textBoxIndex)) selectionAnchor = oldTextBoxIndex;
			} else {
				if(input.shift) {
					if(textBoxIndex == selectionAnchor) selectionAnchor = -1;
				} else {
					if(input.left || input.right || input.home || input.end) {
						if(input.left) textBoxIndex = min(selectionAnchor, oldTextBoxIndex);
						else if(input.right) textBoxIndex = max(selectionAnchor, oldTextBoxIndex);
						selectionAnchor = -1;
					}
				}
			}

			for(int i = 0; i < input.charListSize; i++) {
				if(textBoxTextSize >= textBoxTextCapacity) break;
				char c = input.charList[i];
				if(type == 1 && ((c < '0' || c > '9') && c != '-')) break;
				if(type == 2 && ((c < '0' || c > '9') && c != '.' && c != '-')) break; 

				if(selectionAnchor != -1) {
					int index = min(selectionAnchor, textBoxIndex);
					int size = max(selectionAnchor, textBoxIndex) - index;
					strErase(textBoxText, index, size);
					textBoxTextSize -= size;
					textBoxIndex = index;
					selectionAnchor = -1;
				}

				strInsert(textBoxText, textBoxIndex, input.charList[i]);
				textBoxTextSize++;
				textBoxIndex++;
			}

			if(selectionAnchor != -1) {
				if(input.backSpace || input.del) {
					// code duplication
					int index = min(selectionAnchor, textBoxIndex);
					int size = max(selectionAnchor, textBoxIndex) - index;
					strErase(textBoxText, index, size);
					textBoxTextSize -= size;
					textBoxIndex = index;
					selectionAnchor = -1;
				}
			} else {
				if(input.backSpace && textBoxIndex > 0) {
					strRemove(textBoxText, textBoxIndex, textBoxTextSize);
					textBoxTextSize--;
					textBoxIndex--;
				}
				
				if(input.del && textBoxIndex < textBoxTextSize) {
					strRemove(textBoxText, textBoxIndex+1, textBoxTextSize);
					textBoxTextSize--;
				}
			}

			Rect regCen = rectGetCenDim(region);
			Vec2 textStart = regCen.cen - vec2(regCen.dim.w*0.5f,0);
			Vec2 cursorPos = textStart + vec2(getTextPos(textBoxText, textBoxIndex, font), 0);
			if(textBoxIndex == 0) cursorPos += vec2(1,0); // avoid scissoring cursor on far left

			Rect cursorRect = rectCenDim(cursorPos, vec2(settings.cursorWidth,font->height));

			if(selectionAnchor != -1) {
				int start = min(textBoxIndex, selectionAnchor);
				int end = max(textBoxIndex, selectionAnchor);

				Vec2 selectionStartPos = textStart + vec2(getTextPos(textBoxText, start, font), 0);
				Vec2 selectionEndPos = textStart + vec2(getTextPos(textBoxText, end, font), 0);
				float selectionWidth = selectionEndPos.x - selectionStartPos.x;

				Vec2 center = selectionStartPos+selectionWidth*0.5;
				Rect selectionRect = rectCenDim(selectionStartPos + vec2(selectionWidth*0.5f,0), vec2(selectionWidth,font->height));
				drawRect(selectionRect, colors.selectionColor, true);
			} 

			drawText(textBoxText, 0);
			drawRect(cursorRect, colors.cursorColor, true);

			if(input.enter) {
				if(type == 0) strCpy((char*)originalPointer, textBoxText, textBoxTextSize);
				else if(type == 1) *((int*)originalPointer) = strToInt(textBoxText);
				else *((float*)originalPointer) = strToFloat(textBoxText);

				activeId = 0;
			}

			if(input.mouseClick && !getMouseOver(input.mousePos, region)) activeId = 0;

			if(input.escape) activeId = 0;

		} else {
			if(type == 0) drawText((char*)value, 0);
			else { 
				char* buffer = getTString(50); // unfortunate
				strClear(buffer);
				if(type == 1) {
					intToStr(buffer, *((int*)value));
					drawText(buffer, 0);
				} else {
					floatToStr(buffer, *((float*)value));
					drawText(buffer, 0);
				}
			}
		}

		post();
		return active;
	}

	bool textBoxChar(char* text, int textSize = 0, int textCapacity = 0) {
		return textBox(text, 0, textSize, textCapacity);
	}
	bool textBoxInt(int* number) {
		return textBox(number, 1, 0, arrayCount(textBoxText));
	}
	bool textBoxFloat(float* number) {
		return textBox(number, 2, 0, arrayCount(textBoxText));
	}
};


struct SerializeData {
	char* name;
	int offset;
	int size;
};

void saveData(void* data, SerializeData sd, char* fileName, int fileOffset) {
	writeBufferSectionToFile((char*)(data) + sd.offset, fileName, fileOffset + sd.offset, sd.size);
}

void loadData(void* data, SerializeData sd, char* fileName, int fileOffset) {
	readFileSectionToBuffer((char*)(data) + sd.offset, fileName, fileOffset + sd.offset, sd.size);
}

const SerializeData settingData[] = {
	{"Colors", offsetof(Gui, colors), memberSize(Gui, colors)}, 
	{"Settings", offsetof(Gui, settings), memberSize(Gui, settings)}, 
	{"Layout", offsetof(Gui, cornerPos), memberSize(Gui, cornerPos) + memberSize(Gui, panelStartDim)}, 
};

const char* settingFile = "C:\\Projects\\Hmm\\data\\guiSettings.txt";
const int settingSlotSize = settingData[0].size + settingData[1].size + settingData[2].size;
const int settingSlots = 4;
const int settingFileSize = settingSlotSize*settingSlots;

bool guiCreateSettingsFile() {
	if(!fileExists((char*)settingFile)) {
		createFileAndOverwrite((char*)settingFile, settingFileSize);
		return true;
	}

	return false;
}

void guiSave(Gui* gui, int element, int slot) {
	saveData(gui, settingData[element], (char*)settingFile, settingSlotSize*slot);
}

void guiLoad(Gui* gui, int element, int slot) {
	loadData(gui, settingData[element], (char*)settingFile, settingSlotSize*slot);
}

void guiSaveAll(Gui* gui, int slot) {
	for(int i = 0; i < arrayCount(settingData); i++) guiSave(gui, i, slot);
}

void guiLoadAll(Gui* gui, int slot) {
	for(int i = 0; i < arrayCount(settingData); i++) guiLoad(gui, i, slot);
}


void guiSettings(Gui* gui) {
	static int maxStringWidth = 0;
	static bool staticInit = true;
	if(staticInit) {
		// guiCreateSettingsFile();

		staticInit = false;
		for(int i = 0; i < arrayCount(guiSettingStrings); i++) maxStringWidth = max(maxStringWidth, gui->getTextWidth(guiSettingStrings[i]));
		for(int i = 0; i < arrayCount(guiColorStrings); i++) maxStringWidth = max(maxStringWidth, gui->getTextWidth(guiColorStrings[i]));
	}


	int colorCount = arrayCount(gui->colors.e);

	static int slot = 0;
	gui->div(vec2(gui->getTextWidth("Slot: "),0)); gui->label("Slot: "); gui->slider(&slot, 0, settingSlots-1);


	for(int i = 0; i < arrayCount(settingData); i++) {
		char* string = settingData[i].name;
		gui->div(gui->getTextWidth(string),0,0); gui->label(string, 0); 
		if(gui->button("Save")) guiSave(gui, i, slot);
		if(gui->button("Load")) guiLoad(gui, i, slot);
	}

	gui->heightPush(2); gui->div(vec3(0,0,0)); 
		if(gui->button("SaveAll")) {
			guiSaveAll(gui, slot);
		}
		if(gui->button("LoadAll")) {
			guiLoadAll(gui, slot);
		} 
		if(gui->button("ResetColors")) {
			for(int i = 0; i < colorCount; i++) {
				gui->colors.e[i] = vec4(0,0,0,1);
			}
		}
	gui->heightPop();


	static int tw = maxStringWidth + 2;
	bool mainUpdated = false;
	Vec3 colorUpdate = vec3(0,0,0);
	for(int i = 0; i < colorCount; i++) {
		bool main = i == 0 ? true : false;

		gui->advanceY(5);

		gui->heightPush(2.5f);
			int colorBoxWidth = gui->getDefaultHeight();
			float divWidths[] = {tw, colorBoxWidth, 0, 0, 0};
			gui->div(divWidths, arrayCount(divWidths)); 

			gui->label(guiColorStrings[i],0); 
			if(gui->button("", 0, gui->colors.e[i]) && !main) {
				gui->colors.e[i] = vec4(0,0,0,1);
			}
		gui->heightPop();

		gui->heightPush(1.5f);
		Vec3 hslColor = rgbToHsl(gui->colors.e[i].xyz);

		if(mainUpdated) {
			Vec3 preColor = hslColor + colorUpdate;
			preColor.x = modFloat(preColor.x, 360);
			preColor.y = clamp(preColor.y, 0, 1);
			preColor.z = clamp(preColor.z, 0, 1);
			hslColor = preColor;
		}

		Vec3 oldHslColor = hslColor;
		bool hue = gui->slider(&hslColor.x,0,360);
		bool sat = gui->slider(&hslColor.y,0,1);
		bool lig = gui->slider(&hslColor.z,0,1);

		if(main && (hue || sat || lig)) {
			mainUpdated = true;
			colorUpdate = hslColor - oldHslColor;
		}

		gui->colors.e[i].xyz = hslToRgb(hslColor);
		gui->colors.e[i].w = 1;
		gui->heightPop();

		gui->div(divWidths, arrayCount(divWidths)); gui->label("",0); gui->label("",0); 
		bool r = gui->slider(&gui->colors.e[i].x, 0,1); 
		bool g = gui->slider(&gui->colors.e[i].y, 0,1); 
		bool b = gui->slider(&gui->colors.e[i].z, 0,1);
	}

	gui->advanceY(1);

	char** ss = &guiSettingStrings[0];
	int i = 0;
	GuiSettings* gs = &gui->settings;
	gui->div(tw,0,0); gui->label(ss[i++],0); gui->slider(&gs->offsets.x, 0, 20); gui->slider(&gs->offsets.y, 0, 20);
	gui->div(tw,0,0); gui->label(ss[i++],0); gui->slider(&gs->border.x, 0, 20), gui->slider(&gs->border.y, -20, 0);
	gui->div(tw,0,0); gui->label(ss[i++],0); gui->slider(&gs->minSize.x, 1, 400), gui->slider(&gs->minSize.y, 1, 400);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->sliderSize, 		1, 20);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->sliderResetDistance,20, 300);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->fontOffset, 		0, 1);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->sectionOffset, 		0, 4);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->fontHeightOffset, 	0.5, 2);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->sectionIndent, 		0, 0.5f);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->scrollBarWidth, 	2, 30);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->scrollBarSliderSize,1, 50);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->cursorWidth, 		1, 10);
	gui->div(tw,0); gui->label(ss[i++],0); gui->slider(&gs->textShadow, 		0, 10);
}



struct AppData {
	DebugState debugState;
	DrawCommandList commandListDebug;

	bool showHud;
	bool updateFrameBuffers;
	float guiAlpha;

	Gui gui;

	uint cubemapTextureId[16];
	uint cubemapSamplerId;
	int cubeMapCount;
	int cubeMapDrawIndex;

	SystemData systemData;
	Input input;
	WindowSettings wSettings;

	GraphicsState graphicsState;
	DrawCommandList commandListGui;
	DrawCommandList commandList2d;
	DrawCommandList commandList3d;

	DrawCommandList commandList;

	LONGLONG lastTimeStamp;
	float dt;
	float time;

	bool* treeNoise;

	EntityList entityList;
	Entity* player;
	Entity* cameraEntity;

	Camera activeCam;

	uint samplers[16];
	uint frameBuffers[16];
	uint renderBuffers[16];
	uint frameBufferTextures[16];

	float aspectRatio;
	float fieldOfView;
	float nearPlane;
	float farPlane;

	Vec2i curRes;
	int msaaSamples;
	Vec2i fboRes;
	bool useNativeRes;

	bool playerMode;
	bool pickMode;

	bool playerOnGround;
	int blockMenu[10];
	int blockMenuSelected;

	float bombFireInterval;
	bool bombButtonDown;
	float bombSpawnTimer;

	int selectionRadius;
	bool blockSelected;
	Vec3 selectedBlock;
	Vec3 selectedBlockFaceDir;

	VoxelNode* voxelHash[1024];
	int voxelHashSize;
	uchar* voxelCache[8];
	uchar* voxelLightingCache[8];

	MakeMeshThreadedData threadData[256];

	bool reloadWorld;

	GLuint voxelSamplers[3];
	GLuint voxelTextures[3];

	GLuint testBufferId;
	char* testBuffer;
	int testBufferSize;
};

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

enum Entity_Type {
	ET_Player = 0,
	ET_Camera,
	ET_Rocket,

	ET_Size,
};

struct Entity {
	int init;

	int type;
	int id;
	char name[16];

	Vec3 pos;
	Vec3 dir;
	Vec3 rot;
	float rotAngle;
	Vec3 dim;

	Vec3 camOff;

	Vec3 vel;
	Vec3 acc;

	int movementType;
	int spatial;

	bool deleted;
	bool isMoving;
	bool isColliding;

	bool exploded;

	bool playerOnGround;

	void* data;
};


Vec3 getRotationToVector(Vec3 start, Vec3 dest, float* angle) {
	Vec3 side = normVec3(cross(start, normVec3(dest)));
	*angle = dot(start, normVec3(dest));
	*angle = acos(*angle)*2;

	return side;
}	

void initEntity(Entity* e, int type, Vec3 pos, Vec3 dir, Vec3 dim, Vec3 camOff) {
	*e = {};
	e->init = true;
	e->type = type;
	e->pos = pos;
	e->dir = dir;
	e->dim = dim;
	e->camOff = camOff;
	
	e->rot = getRotationToVector(vec3(0,1,0), dir, &e->rotAngle);
	int stop = 234;
}

Entity* addEntity(EntityList* list, Entity* e) {
	bool foundSlot = false;
	Entity* freeEntity = 0;
	int id = 0;
	for(int i = 0; i < list->size; i++) {
		if(list->e[i].init == false) {
			freeEntity = &list->e[i];
			id = i;
			break;
		}
	}

	assert(freeEntity);

	*freeEntity = *e;
	freeEntity->id = id;

	return freeEntity;
}


// struct RocketEntity {
// };

// struct Player {
// }

#define VK_W 0x57
#define VK_A 0x41
#define VK_S 0x53
#define VK_D 0x44
#define VK_E 0x45
#define VK_Q 0x51

#define VK_T 0x54

#define KEYCODE_0 0x30
#define KEYCODE_1 0x31
#define KEYCODE_2 0x32
#define KEYCODE_3 0x33
#define KEYCODE_4 0x34

struct Particle {
	Vec3 pos;
	Vec3 vel;
	Vec3 acc;

	Vec4 color;
	Vec4 velColor;
	Vec4 accColor;

	Vec3 size;
	Vec3 velSize;
	Vec3 accSize;

	// Vec3 rot;
	// Vec3 velRot;
	// Vec3 accRot;

	float rot;
	float rot2;
	float velRot;
	float accRot;

	float dt;
	float timeToLive;
};

struct ParticleEmitter {
	Particle* particleList;
	int particleListSize;
	int particleListCount;

	Vec3 pos;
	float spawnRate;
	float timeToLive;
	float dt;
	float friction;
};

void particleEmitterUpdate(ParticleEmitter* e, float dt) {
	// push particles
	// e->dt += dt;
	// while(e->dt >= 0.1f) {
	// 	e->dt -= e->spawnRate;

	// 	if(e->particleListCount < e->particleListSize) {
	// 		Particle p = {};
	// 		p.pos = e->pos;
	// 		p.vel = normVec3(vec3(randomFloat(-1,1,0.01f), randomFloat(-1,1,0.01f), randomFloat(-1,1,0.01f))) * 10;
	// 		p.acc = vec3(0,0,0);

	// 		p.timeToLive = 1;

	// 		e->particleList[e->particleListCount++] = p;
	// 	}
	// }

	// update
	// float friction = 0.1f;
	float friction = e->friction;
	for(int i = 0; i < e->particleListCount; i++) {
		Particle* p = e->particleList + i;

		p->vel = p->vel + p->acc*dt;
		// p->vel = p->vel * pow(friction,dt);
		p->pos = p->pos - 0.5f*p->acc*dt*dt + p->vel*dt;

		p->velColor = p->velColor + p->accColor*dt;
		// p->velColor = p->velColor * pow(friction,dt);
		p->color = p->color - 0.5f*p->accColor*dt*dt + p->velColor*dt;

		p->velSize = p->velSize + p->accSize*dt;
		// p->velColor = p->velColor * pow(friction,dt);
		p->size = p->size - 0.5f*p->accSize*dt*dt + p->velSize*dt;

		p->velRot = p->velRot + p->accRot*dt;
		// p->velColor = p->velColor * pow(friction,dt);
		p->rot = p->rot - 0.5f*p->accRot*dt*dt + p->velRot*dt;

		p->dt += dt;
	}

	// remove dead
	for(int i = 0; i < e->particleListCount; i++) {
		Particle* p = e->particleList + i;

		if(p->dt >= p->timeToLive) {
			if(i == e->particleListCount-1) {
				e->particleListCount--;
				break;
			}

			e->particleList[i] = e->particleList[e->particleListCount-1];
			e->particleListCount--;
			i--;
		}
	}
}

// struct NVTokenSequence {
// 	// int a;

// 	int sagaswerghaweftawefeeeeee;

// //   // std::vector<GLintptr>  offsets;
// //   // std::vector<GLsizei>   sizes;
// //   // std::vector<GLuint>    states;
// //   // std::vector<GLuint>    fbos;

// //   // GLint* offsets[100];
// //   // GLsizei sizes[100];
// //   // GLuint states[100];
// //   // GLuint fbos[100];
//   // int size;
// //   char* asdf;
// };

struct NVTokenSequencee {
	GLintptr offsets[100];
	GLsizei sizes[100];
	GLuint states[100];
	GLuint fbos[100];
	int size;
};

struct StateIncarnation {
  uint  programIncarnation;
  uint  fboIncarnation;

  bool operator ==(const StateIncarnation& other) const
  {
    return memcmp(this,&other,sizeof(StateIncarnation)) == 0;
  }

  bool operator !=(const StateIncarnation& other) const
  {
    return memcmp(this,&other,sizeof(StateIncarnation)) != 0;
  }

  StateIncarnation()
    : programIncarnation(0)
    , fboIncarnation(0)
  {

  }
};

struct CmdList {
      // we introduce variables that track when we changed global state
      StateIncarnation  state;
      StateIncarnation  captured;

      // two state objects
      GLuint                stateobj_draw;
      GLuint                stateobj_draw_geo;

// #if ALLOW_EMULATION_LAYER
//       // for emulation
//       StateSystem       statesystem;
//       StateSystem::StateID  stateid_draw;
//       StateSystem::StateID  stateid_draw_geo;
// #endif

      // there is multiple ways to draw the scene
      // either via buffer, cmdlist object, or emulation
      GLuint          tokenBuffer;
      GLuint          tokenCmdList;
      // std::string     tokenData;
      char* tokenData;
      int tokenDataSize; 
      NVTokenSequencee tokenSequence;
      NVTokenSequencee tokenSequenceList;
      NVTokenSequencee tokenSequenceEmu;
};



#pragma optimize( "", off )
// #pragma optimize( "", on )

extern "C" APPMAINFUNCTION(appMain) {
	if(init) {
		SYSTEM_INFO info;
		GetSystemInfo(&info);

		char* baseAddress = (char*)gigaBytes(8);
	    VirtualAlloc(baseAddress, gigaBytes(40), MEM_RESERVE, PAGE_READWRITE);

		ExtendibleMemoryArray* pMemory = &appMemory->extendibleMemoryArrays[appMemory->extendibleMemoryArrayCount++];
		initExtendibleMemoryArray(pMemory, megaBytes(512), info.dwAllocationGranularity, baseAddress);

		ExtendibleBucketMemory* dMemory = &appMemory->extendibleBucketMemories[appMemory->extendibleBucketMemoryCount++];
		initExtendibleBucketMemory(dMemory, megaBytes(1), megaBytes(512), info.dwAllocationGranularity, baseAddress + gigaBytes(16));

		MemoryArray* tMemory = &appMemory->memoryArrays[appMemory->memoryArrayCount++];
		initMemoryArray(tMemory, megaBytes(30), baseAddress + gigaBytes(32));

		MemoryArray* tMemoryDebug = &appMemory->memoryArrays[appMemory->memoryArrayCount++];
		initMemoryArray(tMemoryDebug, megaBytes(30), baseAddress + gigaBytes(33));



		ExtendibleMemoryArray* debugMemory = &appMemory->extendibleMemoryArrays[appMemory->extendibleMemoryArrayCount++];
		initExtendibleMemoryArray(debugMemory, megaBytes(512), info.dwAllocationGranularity, baseAddress + gigaBytes(34));
	}

	MemoryBlock gMemory = {};
	gMemory.pMemory = &appMemory->extendibleMemoryArrays[0];
	gMemory.tMemory = &appMemory->memoryArrays[0];
	gMemory.dMemory = &appMemory->extendibleBucketMemories[0];
	gMemory.tMemoryDebug = &appMemory->memoryArrays[1];
	globalMemory = &gMemory;

	AppData* ad = (AppData*)globalMemory->pMemory->arrays[0].data;
	Input* input = &ad->input;
	SystemData* systemData = &ad->systemData;
	HWND windowHandle = systemData->windowHandle;
	WindowSettings* wSettings = &ad->wSettings;

	globalThreadQueue = threadQueue;
	globalGraphicsState = &ad->graphicsState;
	threadData = ad->threadData;
	globalDebugState = &ad->debugState;

	treeNoise = ad->treeNoise;

	for(int i = 0; i < 8; i++) {
		voxelCache[i] = ad->voxelCache[i];
		voxelLightingCache[i] = ad->voxelLightingCache[i];
	}

	if(init) {
		getPMemory(sizeof(AppData));
		*ad = {};

		ad->dt = 1/(float)60;

		initInput(&ad->input);
		
		wSettings->res.w = 1920;
		wSettings->res.h = 1080;
		wSettings->fullscreen = false;
		wSettings->fullRes.x = GetSystemMetrics(SM_CXSCREEN);
		wSettings->fullRes.y = GetSystemMetrics(SM_CYSCREEN);
		wSettings->style = (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU  | WS_MINIMIZEBOX  | WS_VISIBLE);
		initSystem(systemData, windowsData, 0, 0,0,0,0);

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

		// for(int i = 0; i < GL_NUM_EXTENSIONS; i++) {
		// 	char* s = (char*)glGetStringi(GL_EXTENSIONS, i);
		// 	printf("%s\n", s);
		// }

		// typedef int wglGetSwapIntervalEXTFunction(void);
		// wglGetSwapIntervalEXTFunction* wglGetSwapIntervalEXT;
		// typedef int wglSwapIntervalEXTFunction(void);
		// wglSwapIntervalEXTFunction* wglSwapIntervalEXT;

		// gl##name = (name##Function*)wglGetProcAddress("gl" #name);
		wglGetSwapIntervalEXT = (wglGetSwapIntervalEXTFunction*)wglGetProcAddress("wglGetSwapIntervalEXT");
		wglSwapIntervalEXT = (wglSwapIntervalEXTFunction*)wglGetProcAddress("wglSwapIntervalEXT");
		wglSwapIntervalEXT(1);

		int interval = wglGetSwapIntervalEXT();

		// @setup

		int timerSlots = 10000;
		ad->debugState.bufferSize = timerSlots;
		ad->debugState.timerBuffer = getPArray(TimerSlot, timerSlots);
		ad->debugState.savedTimerBuffer	= getPArray(TimerSlot, timerSlots);
		ad->debugState.stringMemory = getPArray(char, 10000);
		ad->debugState.stringMemorySize = 10000;
		ad->debugState.cycleIndex = 0;

		int clSize = kiloBytes(200);
		drawCommandListInit(&ad->commandListDebug, (char*)getPMemory(clSize), clSize);

		// ad->fieldOfView = 55;
		ad->fieldOfView = 60;
		ad->msaaSamples = 4;
		ad->fboRes = vec2i(0, 120);
		ad->useNativeRes = true;
		ad->nearPlane = 0.1f;
		// ad->farPlane = 2000;
		ad->farPlane = 3000;
		ad->showHud = true;

		ad->guiAlpha = 0.95f;

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

		ad->playerMode = true;
		ad->pickMode = true;
		ad->selectionRadius = 5;
		// input->captureMouse = true;
		input->captureMouse = false;

		*ad->blockMenu = {};
		ad->blockMenuSelected = 0;


		int treeRadius = 4;
		ad->treeNoise = (bool*)getPMemory(VOXEL_X*VOXEL_Y);
		zeroMemory(ad->treeNoise, VOXEL_X*VOXEL_Y);

		Rect bounds = rect(0, 0, 64, 64);
		Vec2* noiseSamples;
		// int noiseSamplesSize = blueNoise(bounds, 5, &noiseSamples);
		int noiseSamplesSize = blueNoise(bounds, treeRadius, &noiseSamples);
		// int noiseSamplesSize = 10;
		for(int i = 0; i < noiseSamplesSize; i++) {
			Vec2 s = noiseSamples[i];
			// Vec2i p = vec2i((int)(s.x/gridCell) * gridCell, (int)(s.y/gridCell) * gridCell);
			// drawRect(rectCenDim(vec2(p), vec2(5,5)), rect(0,0,1,1), vec4(1,0,1,1), ad->textures[0]);
			Vec2i index = vec2i(s);
			ad->treeNoise[index.y*VOXEL_X + index.x] = 1;
		}
		free(noiseSamples);

		ad->bombFireInterval = 0.1f;
		ad->bombButtonDown = false;

		ad->entityList.size = 1000;
		ad->entityList.e = (Entity*)getPMemory(sizeof(Entity)*ad->entityList.size);
		for(int i = 0; i < ad->entityList.size; i++) ad->entityList.e[i].init = false;

		Vec3 startDir = normVec3(vec3(1,0,0));

		Entity player;
		Vec3 playerDim = vec3(0.8f, 0.8f, 1.8f);
		float camOff = playerDim.z*0.5f - playerDim.x*0.25f;
		initEntity(&player, ET_Player, vec3(0,0,40), startDir, playerDim, vec3(0,0,camOff));
		player.rot = vec3(M_2PI,0,0);
		player.playerOnGround = false;
		ad->player = addEntity(&ad->entityList, &player);

		Entity freeCam;
		initEntity(&freeCam, ET_Camera, vec3(35,35,32), startDir, vec3(0,0,0), vec3(0,0,0));
		ad->cameraEntity = addEntity(&ad->entityList, &freeCam);

		uint vao = 0;
		glCreateVertexArrays(1, &vao);
		glBindVertexArray(vao);

		// setup textures
		for(int i = 0; i < TEXTURE_SIZE; i++) {
			if(USE_SRGB) globalGraphicsState->textures[i] = loadTextureFile(texturePaths[i], 1, GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE);
			else globalGraphicsState->textures[i] = loadTextureFile(texturePaths[i], 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		}

		// load cubemap
		glCreateTextures(GL_TEXTURE_CUBE_MAP_ARRAY, arrayCount(ad->cubemapTextureId), &ad->cubemapTextureId[0]);

		char* texturePaths[] = {
			 					// "..\\data\\skybox\\sb1.png",
								// "..\\data\\skybox\\sb2.png", 
								// "..\\data\\skybox\\sb3.jpg", 
								// "..\\data\\skybox\\sb4.png", 
								"..\\data\\skybox\\xoGVD3X.jpg", 
								// "..\\data\\skybox\\xoGVD3X.jpg", 
								// "C:\\Projects\\Hmm\\data\\skybox\\xoGVD3X.jpg", 
								};

		ad->cubeMapCount = arrayCount(texturePaths);

		// for(int textureIndex = 0; textureIndex < arrayCount(ad->cubemapTextureId); textureIndex++) {
		for(int textureIndex = 0; textureIndex < arrayCount(texturePaths); textureIndex++) {
			int texWidth, texHeight, n;
			uint* stbData = (uint*)stbi_load(texturePaths[textureIndex], &texWidth, &texHeight, &n, 4);

			int skySize = texWidth/(float)4;

			glTextureStorage3D(ad->cubemapTextureId[textureIndex], 5, GL_SRGB8_ALPHA8, skySize, skySize, 6);

			uint* skyTex = getTArray(uint, skySize*skySize);
			Vec2i texOffsets[] = {{2,1}, {0,1}, {1,0}, {1,2}, {1,1}, {3,1}};
			for(int i = 0; i < 6; i++) {
				Vec2i offset = texOffsets[i] * skySize;

				for(int x = 0; x < skySize; x++) {
					for(int y = 0; y < skySize; y++) {
						skyTex[y*skySize + x] = stbData[(offset.y+y)*texWidth + (offset.x+x)];
					}
				}

				glTextureSubImage3D(ad->cubemapTextureId[textureIndex], 0, 0, 0, i, skySize, skySize, 1, GL_RGBA, GL_UNSIGNED_BYTE, skyTex);
			}
			// glGenerateTextureMipmap(ad->cubemapTextureId);

			stbi_image_free(stbData);
		}




		// setup shaders and uniforms
		for(int i = 0; i < SHADER_SIZE; i++) {
			MakeShaderInfo* info = makeShaderInfo + i; 
			Shader* s = globalGraphicsState->shaders + i;

			s->program = createShader(info->vertexString, info->fragmentString, &s->vertex, &s->fragment);
			s->uniformCount = info->uniformCount;
			s->uniforms = getPArray(ShaderUniform, s->uniformCount);

			for(int i = 0; i < s->uniformCount; i++) {
				ShaderUniform* uni = s->uniforms + i;
				uni->type = info->uniformNameMap[i].type;	
				uni->vertexLocation = glGetUniformLocation(s->vertex, info->uniformNameMap[i].name);
				uni->fragmentLocation = glGetUniformLocation(s->fragment, info->uniformNameMap[i].name);
			}
		}



		// setup meshs
		for(int i = 0; i < MESH_SIZE; i++) {
			Mesh* mesh = getMesh(i);

			MeshMap* meshMap = meshArrays +i;

			glCreateBuffers(1, &mesh->bufferId);
			glNamedBufferData(mesh->bufferId, meshMap->size, meshMap->vertexArray, GL_STATIC_DRAW);
			mesh->vertCount = meshMap->size / sizeof(Vertex);
		}

		ad->testBufferSize = megaBytes(10);
		ad->testBuffer = getPArray(char, ad->testBufferSize);
		glCreateBuffers(1, &ad->testBufferId);
		glNamedBufferData(ad->testBufferId, ad->testBufferSize, ad->testBuffer, GL_STREAM_DRAW);



		ad->voxelSamplers[0] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR);
		ad->voxelSamplers[1] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR);
		ad->voxelSamplers[2] = createSampler(16.0f, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);

		glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, ad->voxelTextures);

		ad->samplers[0] = createSampler(16.0f, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);

		// voxel textures
		const int mipMapCount = 5;
		char* p = getTString(34);
		strClear(p);
		strAppend(p, "..\\data\\minecraft textures\\");

		char* fullPath = getTString(234);
		if(USE_SRGB) glTextureStorage3D(ad->voxelTextures[0], mipMapCount, GL_SRGB8_ALPHA8, 32, 32, BX_Size);
		else glTextureStorage3D(ad->voxelTextures[0], mipMapCount, GL_RGBA8, 32, 32, BX_Size);

		for(int layerIndex = 0; layerIndex < BX_Size; layerIndex++) {
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

			glTextureSubImage3D(ad->voxelTextures[0], 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);

			stbi_image_free(stbData);
		}
		glGenerateTextureMipmap(ad->voxelTextures[0]);


		float alphaCoverage[mipMapCount] = {};
		int size = 32;
		Vec4* pixels = (Vec4*)getTMemory(sizeof(Vec4)*size*size);
		for(int i = 0; i < mipMapCount; i++) {
			glGetTextureSubImage(ad->voxelTextures[0], i, 0,0,BX_Leaves, size, size, 1, GL_RGBA, GL_FLOAT, size*size*sizeof(Vec4), &pixels[0]);

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
			glGetTextureSubImage(ad->voxelTextures[0], i, 0,0,BX_Leaves, size, size, 1, GL_RGBA, GL_FLOAT, size*size*sizeof(Vec4), &pixels[0]);

			// float alphaScale = (size*size*alphaCoverage[0]) / (alphaCoverage[i]*size*size);
			float alphaScale = (alphaCoverage[0]) / (alphaCoverage[i]);

			for(int y = 0; y < size; y++) {
				for(int x = 1; x < size; x++) {
					pixels[y*size + x].a *= alphaScale;
					alphaCoverage2[i] += pixels[y*size + x].a;
				}
			}
		
			glTextureSubImage3D(ad->voxelTextures[0], i, 0, 0, BX_Leaves, size, size, 1, GL_RGBA, GL_FLOAT, pixels);

			alphaCoverage2[i] = alphaCoverage2[i] / (size*size);
			size /= 2;
		}


		if(USE_SRGB) glTextureStorage3D(ad->voxelTextures[1], 1, GL_SRGB8_ALPHA8, 32, 32, BX2_Size);
		else glTextureStorage3D(ad->voxelTextures[1], 1, GL_RGBA8, 32, 32, BX2_Size);
		for(int layerIndex = 0; layerIndex < BX2_Size; layerIndex++) {
			int x,y,n;
			unsigned char* stbData = stbi_load(textureFilePaths2[layerIndex], &x, &y, &n, 4);
			
			glTextureSubImage3D(ad->voxelTextures[1], 0, 0, 0, layerIndex, x, y, 1, GL_RGBA, GL_UNSIGNED_BYTE, stbData);
			stbi_image_free(stbData);
		}


		glCreateFramebuffers(5, ad->frameBuffers);
		glCreateRenderbuffers(2, ad->renderBuffers);
		glCreateTextures(GL_TEXTURE_2D, 6, ad->frameBufferTextures);
		GLenum result = glCheckNamedFramebufferStatus(ad->frameBuffers[0], GL_FRAMEBUFFER);

		return; // window operations only work after first frame?
	}



	if(second) {
		// setWindowProperties(windowHandle, wSettings->res.w, wSettings->res.h, -1920, 0);
		// setWindowProperties(windowHandle, wSettings->res.w, wSettings->res.h, 0, 0);
		setWindowStyle(windowHandle, wSettings->style);
		// setWindowMode(windowHandle, wSettings, WINDOW_MODE_FULLBORDERLESS);

		setWindowProperties(windowHandle, wSettings->res.w, wSettings->res.h, 300, 300);
		setWindowMode(windowHandle, wSettings, WINDOW_MODE_WINDOWED);


		ad->updateFrameBuffers = true;
	}

	if(reload) {
		loadFunctions();
		SetWindowLongPtr(systemData->windowHandle, GWLP_WNDPROC, (LONG_PTR)mainWindowCallBack);
	}

	TIMER_BLOCK_BEGIN(Main)

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

		ad->time += ad->dt;
	}
	// printf("%f \n", ad->dt);
	// ad->dt = 0.016f;



	// alloc drawcommandlist	
	int clSize = kiloBytes(1000);
	drawCommandListInit(&ad->commandListGui, (char*)getTMemory(clSize), clSize);
	drawCommandListInit(&ad->commandList3d, (char*)getTMemory(clSize), clSize);
	drawCommandListInit(&ad->commandList2d, (char*)getTMemory(clSize), clSize);
	drawCommandListInit(&ad->commandList, (char*)getTMemory(clSize), clSize);
	globalCommandList = &ad->commandList3d;


	{
		// TIMER_BLOCK_NAMED("Input");
		updateInput(&ad->input, isRunning, windowHandle);
	}

	{
		ExtendibleMemoryArray* debugMemory = &appMemory->extendibleMemoryArrays[1];
		ExtendibleMemoryArray* pMemory = globalMemory->pMemory;

		if(input->keysPressed[VK_F11]) {
			for(int i = 0; i < pMemory->index; i++) {
				if(debugMemory->index < i) getExtendibleMemoryArray(pMemory->slotSize, debugMemory);

				void* data = debugMemory->arrays[i].data;
				memCpy(data, pMemory->arrays[i].data, pMemory->slotSize);
			}
		}
		
		// if(input->keysPressed[VK_F12]) {
		// 	for(int i = 0; i < pMemory->index; i++) {
		// 		memCpy(pMemory->arrays[i].data, debugMemory->arrays[i].data, pMemory->slotSize);
		// 	}
		// }
	}

	if(input->keysPressed[VK_F1]) {
		int mode;
		if(wSettings->fullscreen) mode = WINDOW_MODE_WINDOWED;
		else mode = WINDOW_MODE_FULLBORDERLESS;
		setWindowMode(windowHandle, wSettings, mode);

		ad->updateFrameBuffers = true;
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

	if(input->keysPressed[VK_F3]) {
		static bool switchMonitor = false;

		setWindowMode(windowHandle, wSettings, WINDOW_MODE_WINDOWED);

		if(!switchMonitor) setWindowProperties(windowHandle, 1, 1, 1920, 0);
		else setWindowProperties(windowHandle, 1920, 1080, -1920, 0);
		switchMonitor = !switchMonitor;

		setWindowMode(windowHandle, wSettings, WINDOW_MODE_FULLBORDERLESS);

		ad->updateFrameBuffers = true;
	}

	if(ad->updateFrameBuffers) {
		ad->updateFrameBuffers = false;
		ad->aspectRatio = wSettings->aspectRatio;
		
		ad->fboRes.x = ad->fboRes.y*ad->aspectRatio;

		if(ad->useNativeRes) ad->curRes = wSettings->currentRes;
		else ad->curRes = ad->fboRes;

		Vec2i s = ad->curRes;


		glNamedRenderbufferStorageMultisample(ad->renderBuffers[0], ad->msaaSamples, GL_RGBA8, s.w, s.h);
		glNamedFramebufferRenderbuffer(ad->frameBuffers[0], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ad->renderBuffers[0]);

		glNamedRenderbufferStorageMultisample(ad->renderBuffers[1], ad->msaaSamples, GL_DEPTH_STENCIL, s.w, s.h);
		glNamedFramebufferRenderbuffer(ad->frameBuffers[0], GL_DEPTH_STENCIL_ATTACHMENT,  GL_RENDERBUFFER, ad->renderBuffers[1]);

		glDeleteTextures(1, &ad->frameBufferTextures[0]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[0]);
		glTextureStorage2D(ad->frameBufferTextures[0], 1, GL_RGBA8, s.w, s.h);
		glNamedFramebufferTexture(ad->frameBuffers[1], GL_COLOR_ATTACHMENT0, ad->frameBufferTextures[0], 0);

		glDeleteTextures(1, &ad->frameBufferTextures[3]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[3]);
		glTextureStorage2D(ad->frameBufferTextures[3], 1, GL_DEPTH24_STENCIL8, s.w, s.h);
		glNamedFramebufferTexture(ad->frameBuffers[1], GL_DEPTH_STENCIL_ATTACHMENT, ad->frameBufferTextures[3], 0);


		Vec2 reflectionRes = vec2(s);

		glDeleteTextures(1, &ad->frameBufferTextures[1]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[1]);
		glTextureStorage2D(ad->frameBufferTextures[1], 1, GL_RGBA8, reflectionRes.w, reflectionRes.h);
		glNamedFramebufferTexture(ad->frameBuffers[2], GL_COLOR_ATTACHMENT0, ad->frameBufferTextures[1], 0);

		glDeleteTextures(1, &ad->frameBufferTextures[2]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[2]);
		glTextureStorage2D(ad->frameBufferTextures[2], 1, GL_DEPTH24_STENCIL8, reflectionRes.w, reflectionRes.h);
		glNamedFramebufferTexture(ad->frameBuffers[2], GL_DEPTH_STENCIL_ATTACHMENT, ad->frameBufferTextures[2], 0);



		glDeleteTextures(1, &ad->frameBufferTextures[4]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[4]);
		// glTextureStorage2D(ad->frameBufferTextures[4], 1, GL_RGBA8, s.w, s.h);
		glTextureStorage2D(ad->frameBufferTextures[4], 1, GL_RGBA8, wSettings->currentRes.w, wSettings->currentRes.h);
		glNamedFramebufferTexture(ad->frameBuffers[3], GL_COLOR_ATTACHMENT0, ad->frameBufferTextures[4], 0);


		glDeleteTextures(1, &ad->frameBufferTextures[5]);
		glCreateTextures(GL_TEXTURE_2D, 1, &ad->frameBufferTextures[5]);
		glTextureStorage2D(ad->frameBufferTextures[5], 1, GL_RGBA8, wSettings->currentRes.w, wSettings->currentRes.h);
		// glTextureStorage2D(ad->frameBufferTextures[5], 1, GL_SRGB8_ALPHA8, wSettings->currentRes.w, wSettings->currentRes.h);
		glNamedFramebufferTexture(ad->frameBuffers[4], GL_COLOR_ATTACHMENT0, ad->frameBufferTextures[5], 0);



		GLenum result = glCheckNamedFramebufferStatus(ad->frameBuffers[0], GL_FRAMEBUFFER);
		GLenum result2 = glCheckNamedFramebufferStatus(ad->frameBuffers[1], GL_FRAMEBUFFER);
		GLenum result3 = glCheckNamedFramebufferStatus(ad->frameBuffers[2], GL_FRAMEBUFFER);
	}



	// 2d camera controls
	// Vec3* cam = &ad->camera;	
	// if(input->mouseButtonDown[0]) {
	// 	cam->x += input->mouseDeltaX*(cam->z/wSettings->currentRes.w);
	// 	cam->y -= input->mouseDeltaY*((cam->z/wSettings->currentRes.h)/ad->aspectRatio);
	// }

	// if(input->mouseWheel) {
	// 	float zoom = cam->z;
	// 	zoom -= input->mouseWheel/(float)1;
	// 	cam->z = zoom;
	// }

	Entity* player = ad->player;
	Entity* camera = ad->cameraEntity;

	if(input->keysPressed[VK_F4]) {
		if(ad->playerMode) {
			camera->pos = player->pos + player->camOff;
			camera->dir = player->dir;
			camera->rot = player->rot;
			camera->rotAngle = player->rotAngle;
		}
		ad->playerMode = !ad->playerMode;
	}

	if(input->mouseWheel) {
		ad->blockMenuSelected += -input->mouseWheel;
		ad->blockMenuSelected = mod(ad->blockMenuSelected, arrayCount(ad->blockMenu));
	}

	if(input->keysPressed[KEYCODE_0]) ad->blockMenuSelected = 9;
	for(int i = 0; i < 9; i++) {
		if(input->keysPressed[KEYCODE_0 + i+1]) ad->blockMenuSelected = i;
	}

	if(ad->playerMode) {
		camera->vel = vec3(0,0,0);
	} else {
		player->vel = vec3(0,0,0);
	}

	if(!ad->playerMode && input->keysPressed[VK_SPACE]) {
		player->pos = camera->pos;
		player->dir = camera->dir;
		player->rot = camera->rot;
		player->rotAngle = camera->rotAngle;
		player->vel = camera->vel;
		ad->playerMode = true;
		input->keysPressed[VK_SPACE] = false;
		input->keysDown[VK_SPACE] = false;
	}

	// spawn bomb
	bool spawnBomb = false;
	if(input->mouseButtonPressed[2]) {
		spawnBomb = true;
		ad->bombSpawnTimer = 0;
	}

	if(input->mouseButtonDown[2]) {
		ad->bombSpawnTimer += ad->dt;
	}

	if(ad->bombSpawnTimer >= ad->bombFireInterval) {
		spawnBomb = true;
		ad->bombSpawnTimer = 0;
	}

	if(spawnBomb) {
		Entity b;
		Vec3 bombPos = ad->activeCam.pos + ad->activeCam.look*4;
		initEntity(&b, ET_Rocket, bombPos, ad->activeCam.look, vec3(0.5f), vec3(0,0,0));
		b.vel = ad->activeCam.look*300;
		b.acc = ad->activeCam.look*200;
		b.isMoving = true;

		addEntity(&ad->entityList, &b);
	}

	// make sure the meshs around the player are loaded at startup
	if(second) {
		// Vec2i pPos = coordToMesh(ad->activeCam.pos);
		Vec2i pPos = coordToMesh(ad->player->pos);
		// for(int i = 0; i < 2; i++) {
			for(int y = -1; y < 2; y++) {
				for(int x = -1; x < 2; x++) {
					Vec2i coord = pPos - vec2i(x,y);

					VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
					makeMesh(m, ad->voxelHash, ad->voxelHashSize);
				}
			}

			threadQueueComplete(globalThreadQueue);
		// }
	}	

	// @update entities
	// TIMER_BLOCK_BEGIN_NAMED(entities, "Upd Entities");
	for(int i = 0; i < ad->entityList.size; i++) {
		Entity* e = &ad->entityList.e[i];
		if(!e->init) continue;

		float dt = ad->dt;
		Vec3 up = vec3(0,0,1);

		switch(e->type) {
			case ET_Player: {
				if(ad->playerMode == false) continue;

				Camera cam = getCamData(e->pos, e->rot);
				e->acc = vec3(0,0,0);

				bool rightLock = true;
				float runBoost = 1.5f;
				float speed = 30;

				if((!fpsMode && input->mouseButtonDown[1]) || fpsMode) {
					float turnRate = ad->dt*0.3f;
					e->rot.y += turnRate * input->mouseDeltaY;
					e->rot.x += turnRate * input->mouseDeltaX;

					float margin = 0.00001f;
					clamp(&e->rot.y, -M_PI+margin, M_PI-margin);

					e->rot.x = modFloat(e->rot.x, (float)M_PI*4);
				}

				if( input->keysDown[VK_W] || input->keysDown[VK_A] || input->keysDown[VK_S] || 
					input->keysDown[VK_D]) {

					if(rightLock || input->keysDown[VK_CONTROL]) cam.look = cross(up, cam.right);

					Vec3 acceleration = vec3(0,0,0);
					if(input->keysDown[VK_SHIFT]) speed *= runBoost;
					if(input->keysDown[VK_W]) acceleration +=  normVec3(cam.look);
					if(input->keysDown[VK_S]) acceleration += -normVec3(cam.look);
					if(input->keysDown[VK_D]) acceleration +=  normVec3(cam.right);
					if(input->keysDown[VK_A]) acceleration += -normVec3(cam.right);
					e->acc += normVec3(acceleration)*speed;
				}

				e->acc.z = 0;

				if(ad->playerMode) {
					// if(input->keysPressed[VK_SPACE]) {
					if(input->keysDown[VK_SPACE]) {
						if(player->playerOnGround) {
							player->vel += up*7.0f;
							player->playerOnGround = false;
						}
					}
				}

				float gravity = 20.0f;
				if(!e->playerOnGround) e->acc += -up*gravity;
				e->vel = e->vel + e->acc*dt;
				float friction = 0.01f;
				e->vel.x *= pow(friction,dt);
				e->vel.y *= pow(friction,dt);
				// e->vel *= 0.9f;

				if(e->playerOnGround) e->vel.z = 0;

				bool playerGroundCollision = false;
				bool playerCeilingCollision = false;
				bool playerSideCollision = false;

				if(e->vel != vec3(0,0,0)) {
					Vec3 pPos = e->pos;
					Vec3 pSize = e->dim;

					Vec3 nPos = pPos + -0.5f*e->acc*dt*dt + e->vel*dt;

					int collisionCount = 0;
					bool collision = true;
					while(collision) {

						// get mesh coords that touch the player box
						Rect3 box = rect3CenDim(nPos, pSize);
						Vec3i voxelMin = coordToVoxel(box.min);
						Vec3i voxelMax = coordToVoxel(box.max+1);

						Vec3 collisionBox;
						collision = false;
						float minDistance = 100000;

							// check collision with the voxel thats closest
						for(int x = voxelMin.x; x < voxelMax.x; x++) {
							for(int y = voxelMin.y; y < voxelMax.y; y++) {
								for(int z = voxelMin.z; z < voxelMax.z; z++) {
									Vec3i coord = vec3i(x,y,z);
									uchar* block = getBlockFromVoxel(ad->voxelHash, ad->voxelHashSize, coord);

									if(*block > 0) {
										Vec3 cBox = voxelToVoxelCoord(coord);
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
							break;	
						}
					}

					float stillnessThreshold = 0.0001f;
					if(valueBetween(e->vel.z, -stillnessThreshold, stillnessThreshold)) {
						e->vel.z = 0;
					}

					if(playerCeilingCollision) {
						e->vel.z = 0;
					}

					if(playerSideCollision) {
						float sideFriction = 0.0010f;
						e->vel.x *= pow(sideFriction,dt);
						e->vel.y *= pow(sideFriction,dt);
					}

					if(collisionCount > 5) {
						e->vel = vec3(0,0,0);
					}

					e->pos = nPos;
				}

				// raycast for touching ground
				if(ad->playerMode) {
					if(playerGroundCollision && e->vel.z <= 0) {
						e->playerOnGround = true;
						e->vel.z = 0;
					}

					if(e->playerOnGround) {
						Vec3 pos = e->pos;
						Vec3 size = e->dim;
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
							gp -= up*raycastThreshold;

							Vec3 block = coordToVoxelCoord(gp);
							uchar* blockType = getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, gp);

							if(*blockType > 0) {
								groundCollision = true;
								break;
							}
						}

						if(groundCollision) {
							if(e->vel.z <= 0) e->playerOnGround = true;
						} else {
							e->playerOnGround = false;
						}
					}
				}


			} break;

			case ET_Camera: {
				Camera cam = getCamData(e->pos, e->rot);

				e->acc = vec3(0,0,0);

				bool rightLock = false;
				float runBoost = 2.0f;
				float speed = 150;
				if(input->keysDown[VK_T]) speed = 1000;

				if((!fpsMode && input->mouseButtonDown[1]) || fpsMode) {
					float turnRate = ad->dt*0.3f;
					e->rot.y += turnRate * input->mouseDeltaY;
					e->rot.x += turnRate * input->mouseDeltaX;

					float margin = 0.00001f;
					clamp(&e->rot.y, -M_PI+margin, M_PI-margin);
				}

				if( input->keysDown[VK_W] || input->keysDown[VK_A] || input->keysDown[VK_S] || 
					input->keysDown[VK_D] || input->keysDown[VK_E] || input->keysDown[VK_Q]) {

					if(rightLock || input->keysDown[VK_CONTROL]) cam.look = cross(up, cam.right);

					Vec3 acceleration = vec3(0,0,0);
					if(input->keysDown[VK_SHIFT]) speed *= runBoost;
					if(input->keysDown[VK_W]) acceleration +=  normVec3(cam.look);
					if(input->keysDown[VK_S]) acceleration += -normVec3(cam.look);
					if(input->keysDown[VK_D]) acceleration +=  normVec3(cam.right);
					if(input->keysDown[VK_A]) acceleration += -normVec3(cam.right);
					if(input->keysDown[VK_E]) acceleration +=  normVec3(up);
					if(input->keysDown[VK_Q]) acceleration += -normVec3(up);
					e->acc += normVec3(acceleration)*speed;
				}

				e->vel = e->vel + e->acc*dt;
				float friction = 0.01f;
				e->vel = e->vel * pow(friction,dt);

				if(e->vel != vec3(0,0,0)) {
					e->pos = e->pos - 0.5f*e->acc*dt*dt + e->vel*dt;
				}

			} break;

			case ET_Rocket: {
				// float gravity = 20.0f;
				float gravity = 1.0f;

				e->acc += -up*gravity;
				e->vel = e->vel + e->acc*dt;

				float friction = 0.01f;
				e->vel.x *= pow(friction,dt);
				e->vel.y *= pow(friction,dt);
				e->vel.z *= pow(friction,dt);

				if(e->vel != vec3(0,0,0)) {
					Vec3 pPos = e->pos;
					Vec3 pSize = e->dim;

					Vec3 nPos = pPos + -0.5f*e->acc*dt*dt + e->vel*dt;

					bool collision = false;

					Rect3 box = rect3CenDim(nPos, pSize);
					Vec3i voxelMin = coordToVoxel(box.min);
					Vec3i voxelMax = coordToVoxel(box.max+1);

					collision = false;
					Vec3 collisionBox;

					for(int x = voxelMin.x; x < voxelMax.x; x++) {
						for(int y = voxelMin.y; y < voxelMax.y; y++) {
							for(int z = voxelMin.z; z < voxelMax.z; z++) {
								Vec3i coord = vec3i(x,y,z);
								uchar* block = getBlockFromVoxel(ad->voxelHash, ad->voxelHashSize, coord);

								if(*block > 0) {
									collisionBox = voxelToVoxelCoord(coord);
									collision = true;
									goto forGoto;
								}
							}
						}
					} forGoto:

					if(collision) {
						e->init = false;
						e->exploded = true;
					}

					Vec2i updateMeshList[8];
					int updateMeshListSize = 0;

					if(e->exploded) {
						Vec3 startPos = collisionBox;
						float sRad = 10;

						// float resolution = M_2PI;
						// float vStep = 0.5f;
						float resolution = M_PI;
						float vStep = 0.75f;

						int itCount = sRad*resolution;
						for(int it = 0; it < itCount+1; it++) {
							float off = degreeToRadian(it * (360/(float)itCount));
							Vec3 dir = rotateVec3(normVec3(vec3(0,1,0)), off, vec3(1,0,0));
							float off2 = sin(off/(float)2)*sRad;

							float rad = (dir*sRad).y;
							for(int i = 0; i < 2; i++) {
								Vec3 pos;
								if(i == 0) pos = startPos + vec3(0,off2,0);
								else pos = startPos + vec3(0,-off2,0);

								int itCount = rad*resolution;
								for(int it = 0; it < itCount+1; it++) {
									float off = degreeToRadian(it * (360/(float)itCount));
									Vec3 dir = rotateVec3(normVec3(vec3(1,0,0)), off, vec3(0,-1,0));
									Vec3 p = pos + dir*rad;

									float cubeSize = 1.0f;

									// dcCube({coordToVoxelCoord(pos + dir*rad), vec3(cubeSize), vec4(1,0.5f,0,1), 0, vec3(0,0,0)});
									// dcCube({coordToVoxelCoord(pos - dir*rad), vec3(cubeSize), vec4(1,0.5f,0,1), 0, vec3(0,0,0)});

									*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, pos+dir*rad) = 0; 
									*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos+dir*rad) = globalLumen; 
									*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, pos-dir*rad) = 0; 
									*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos-dir*rad) = globalLumen; 

									for(int it = 0; it < 2; it++) {
										bool found = false;
										Vec2i mc;
										if(it == 0) mc = coordToMesh(pos+dir*rad);
										else mc = coordToMesh(pos-dir*rad);
										for(int i = 0; i < updateMeshListSize; i++) {
											if(updateMeshList[i] == mc) {
												found = true;
												break;
											}
										}
										if(!found) {
											updateMeshList[updateMeshListSize++] = mc;
										}

									}

									float off2 = sin(off/(float)2)*rad;
									for(float z = 0; z < (dir*rad).x; z += vStep) {
										// dcCube({coordToVoxelCoord(pos + vec3(off2,0, z)), vec3(cubeSize), vec4(0,0.5f,1,1), 0, vec3(0,0,0)});
										// dcCube({coordToVoxelCoord(pos + vec3(off2,0,-z)), vec3(cubeSize), vec4(0,0.5f,1,1), 0, vec3(0,0,0)});
										// dcCube({coordToVoxelCoord(pos - vec3(off2,0,z)), vec3(cubeSize), vec4(0,0.5f,1,1), 0, vec3(0,0,0)});
										// dcCube({coordToVoxelCoord(pos - vec3(off2,0,-z)), vec3(cubeSize), vec4(0,0.5f,1,1), 0, vec3(0,0,0)});

										*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize,    pos + vec3(off2,0, z)) = 0; 
										*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos + vec3(off2,0, z)) = globalLumen; 
										*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize,    pos + vec3(off2,0,-z)) = 0; 
										*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos + vec3(off2,0,-z)) = globalLumen; 
										*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize,    pos - vec3(off2,0, z)) = 0; 
										*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos - vec3(off2,0, z)) = globalLumen; 
										*getBlockFromCoord(ad->voxelHash, ad->voxelHashSize,    pos - vec3(off2,0,-z)) = 0; 
										*getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, pos - vec3(off2,0,-z)) = globalLumen; 
									}
								}
							}
						}

						for(int i = 0; i < updateMeshListSize; i++) {
							VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, updateMeshList[i]);
							m->upToDate = false;
							m->meshUploaded = false;
							m->modifiedByUser = true;
						}
					}

					e->pos = nPos;

					dcCube(e->pos, e->dim, vec4(1,0.5f,0,1), 0, vec3(0,0,0));
				}
			} break;

			default: {

			};
		}
	}
	// TIMER_BLOCK_END(entities);

	if(ad->playerMode) {
		ad->activeCam = getCamData(ad->player->pos, ad->player->rot, ad->player->camOff);
	} else {
		ad->activeCam = getCamData(ad->cameraEntity->pos, ad->cameraEntity->rot, ad->cameraEntity->camOff);
	}

	// selecting blocks and modifying them
	if(ad->playerMode) {
		ad->blockSelected = false;

		// get block in line
		Vec3 startDir = ad->activeCam.look;
		Vec3 startPos = player->pos + player->camOff;

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

			Vec3 blockCoords = voxelToVoxelCoord(coordToVoxel(newPos));

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

				uchar* blockType = getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, block);
				Vec3 temp = voxelToVoxelCoord(coordToVoxel(block));

				// Vec4 c;
				// if(i == 0) c = vec4(1,0,1,1);
				// else c = vec4(0,0,1,1);
				// glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				// drawCube(&ad->pipelineIds, temp, vec3(1.0f,1.0f,1.0f), c, 0, vec3(0,0,0));

				// if(blockType > 0) {
				if(*blockType > 0) {
					Vec3 iBox = voxelToVoxelCoord(coordToVoxel(block));
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

			if(ad->playerMode && fpsMode) {
				VoxelMesh* vm = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordToMesh(intersectionBox));

				uchar* block = getBlockFromCoord(ad->voxelHash, ad->voxelHashSize, intersectionBox);
				uchar* lighting = getLightingFromCoord(ad->voxelHash, ad->voxelHashSize, intersectionBox);

				bool mouse1 = input->mouseButtonPressed[0];
				bool mouse2 = input->mouseButtonPressed[1];
				bool placeBlock = (!fpsMode && ad->pickMode && mouse1) || (fpsMode && mouse1);
				bool removeBlock = (!fpsMode && !ad->pickMode && mouse1) || (fpsMode && mouse2);

				if(placeBlock || removeBlock) {
					vm->upToDate = false;
					vm->meshUploaded = false;
					vm->modifiedByUser = true;

					// if block at edge of mesh, we have to update the mesh on the other side too
					Vec2i currentCoord = coordToMesh(intersectionBox);
					for(int i = 0; i < 4; i++) {
						Vec3 offset;
						if(i == 0) offset = vec3(1,0,0);
						else if(i == 1) offset = vec3(-1,0,0);
						else if(i == 2) offset = vec3(0,1,0);
						else if(i == 3) offset = vec3(0,-1,0);

						Vec2i mc = coordToMesh(intersectionBox + offset);
						if(mc != currentCoord) {
							VoxelMesh* edgeMesh = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, mc);
							edgeMesh->upToDate = false;
							edgeMesh->meshUploaded = false;
							edgeMesh->modifiedByUser = true;
						}
					}
				}

				if(placeBlock) {
					Vec3 boxToCamDir = startPos - intersectionBox;
					Vec3 sideBlock = coordToVoxelCoord(intersectionBox + faceDir);
					Vec3i voxelSideBlock = coordToVoxel(sideBlock);

					// CodeDuplication:
					// get mesh coords that touch the player box
					Rect3 box = rect3CenDim(player->pos, player->dim);
					Vec3i voxelMin = coordToVoxel(box.min);
					Vec3i voxelMax = coordToVoxel(box.max+1);
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
						uchar* sideBlockType = getBlockFromVoxel(ad->voxelHash, ad->voxelHashSize, voxelSideBlock);
						uchar* sideBlockLighting = getLightingFromVoxel(ad->voxelHash, ad->voxelHashSize, voxelSideBlock);

						*sideBlockType = ad->blockMenu[ad->blockMenuSelected];
						*sideBlockLighting = 0;
					}
				} else if(removeBlock) {
					if(*block > 0) {
						*block = 0;
						*lighting = globalLumen;
					}
				}
			}
		}
	}

	// opengl init

	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

	const int count = 10;
	GLenum sources;
	GLenum types;
	GLuint ids;
	GLenum severities;
	GLsizei lengths;

	int bufSize = 1000;
	char* messageLog = getTString(bufSize);

	uint fetchedLogs = 1;
	while(fetchedLogs = glGetDebugMessageLog(count, bufSize, &sources, &types, &ids, &severities, &lengths, messageLog)) {
		if(severities == GL_DEBUG_SEVERITY_NOTIFICATION) continue;

		if(severities == GL_DEBUG_SEVERITY_HIGH) printf("HIGH: \n");
		else if(severities == GL_DEBUG_SEVERITY_MEDIUM) printf("MEDIUM: \n");
		else if(severities == GL_DEBUG_SEVERITY_LOW) printf("LOW: \n");
		else if(severities == GL_DEBUG_SEVERITY_NOTIFICATION) printf("NOTE: \n");

		printf("\t%s \n", messageLog);
	}

	// glEnable(GL_FRAMEBUFFER_SRGB);
	glEnable(GL_CULL_FACE);

	// Vec3 skyColor = vec3(0.90f, 0.90f, 0.95f);
	// Vec3 skyColor = vec3(0.95f);
	Vec3 skyColor = vec3(0.90f);
	// Vec3 fogColor = vec3(0.75f, 0.85f, 0.95f);
	// Vec3 fogColor = vec3(0.43f,0.38f,0.44f);
	Vec3 fogColor = vec3(0.43f,0.38f,0.44f);
	fogColor.x = powf(fogColor.x, (float)2.2f);
	fogColor.y = powf(fogColor.y, (float)2.2f);
	fogColor.z = powf(fogColor.z, (float)2.2f);

	// for tech showcase
	#ifdef STBVOX_CONFIG_LIGHTING_SIMPLE
	skyColor = skyColor * vec3(0.3f);pushUniform
	fogColor = fogColor * vec3(0.3f);
	#endif 

	glViewport(0,0, ad->curRes.x, ad->curRes.y);
	// glClearColor(0,0,0, 1.0f);
	// glClearColor(skyColor.x, skyColor.y, skyColor.z, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer (GL_FRAMEBUFFER, ad->frameBuffers[4]);
	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindFramebuffer (GL_FRAMEBUFFER, ad->frameBuffers[1]);
	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[0]);
	glClearColor(skyColor.x, skyColor.y, skyColor.z, 1.0f);
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
	// glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendEquation(GL_FUNC_ADD);

	lookAt(ad->activeCam.pos, -ad->activeCam.look, ad->activeCam.up, ad->activeCam.right);
	perspective(degreeToRadian(ad->fieldOfView), ad->aspectRatio, ad->nearPlane, ad->farPlane);

	Mat4 view, proj; 
	viewMatrix(&view, ad->activeCam.pos, -ad->activeCam.look, ad->activeCam.up, ad->activeCam.right);
	projMatrix(&proj, degreeToRadian(ad->fieldOfView), ad->aspectRatio, ad->nearPlane, ad->farPlane);

	globalGraphicsState->textureUnits[0] = ad->voxelTextures[0];
	globalGraphicsState->textureUnits[1] = ad->voxelTextures[1];
	globalGraphicsState->samplerUnits[0] = ad->voxelSamplers[0];
	globalGraphicsState->samplerUnits[1] = ad->voxelSamplers[1];
	globalGraphicsState->samplerUnits[2] = ad->voxelSamplers[2];

	setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, fogColor);



	// draw cubemap
	bindShader(SHADER_CUBEMAP);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glBindTextures(0, 1, &ad->cubemapTextureId[ad->cubeMapDrawIndex]);
	glBindSamplers(0, 1, ad->samplers);

	Vec3 skyBoxRot;
	if(ad->playerMode) skyBoxRot = ad->player->rot;
	else skyBoxRot = ad->cameraEntity->rot;
	skyBoxRot.x += M_PI;

	Camera skyBoxCam = getCamData(vec3(0,0,0), skyBoxRot, vec3(0,0,0), vec3(0,1,0), vec3(0,0,1));

	Mat4 viewMat; viewMatrix(&viewMat, skyBoxCam.pos, -skyBoxCam.look, skyBoxCam.up, skyBoxCam.right);
	Mat4 projMat; projMatrix(&projMat, degreeToRadian(ad->fieldOfView), ad->aspectRatio, 0.001f, 2);
	pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_VIEW, viewMat.e);
	pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_PROJ, projMat.e);

	pushUniform(SHADER_CUBEMAP, 2, CUBEMAP_UNIFORM_CLIPPLANE, false);

	glDepthMask(false);
	glFrontFace(GL_CCW);
	glDrawArrays(GL_TRIANGLES, 0, 6*6);
	glFrontFace(GL_CW);
	glDepthMask(true);
	glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);


	// #if 1



	// static float worldTimer = 0;
	// worldTimer += ad->dt;

	// if(worldTimer >= 1) {
	// 	worldTimer = 0;

	// 	int radius = VIEW_DISTANCE/VOXEL_X;

	// 	for(int y = -radius; y < radius; y++) {
	// 		for(int x = -radius; x < radius; x++) {
	// 			Vec2i coord = vec2i(y, x);
	// 			// Vec2i coord = vec2i(0,0);	
	// 			VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
	// 			m->upToDate = false;
	// 			m->meshUploaded = false;
	// 			m->generated = false;
	// 		}
	// 	}
	// }

	// bool worldLoaded = false;
	// while(worldLoaded == false) {
	// 	int radius = VIEW_DISTANCE/VOXEL_X;

	// 	worldLoaded = true;

	// 	for(int y = -radius; y < radius; y++) {
	// 		for(int x = -radius; x < radius; x++) {
	// 			Vec2i coord = vec2i(y, x);
	// 			VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);

	// 			if(!m->meshUploaded) {
	// 				makeMesh(m, ad->voxelHash, ad->voxelHashSize);

	// 				worldLoaded = false;
	// 			}
	// 		}
	// 	}
	// }

	if(ad->reloadWorld) {	
		ad->reloadWorld = false;

		if(threadQueueFinished(threadQueue)) {
			int radius = VIEW_DISTANCE/VOXEL_X;

			Vec3 pp = ad->activeCam.pos;
			Vec2i worldPos = coordToMesh(pp);

			for(int y = worldPos.y-radius; y < worldPos.y+radius; y++) {
				for(int x = worldPos.x-radius; x < worldPos.x+radius; x++) {
					Vec2i coord = vec2i(x, y);
					VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
					m->upToDate = false;
					m->meshUploaded = false;
					m->generated = false;

					makeMesh(m, ad->voxelHash, ad->voxelHashSize);
				}
			}
			// return;
		} 
	}


	#if 0
	// @worldgen
	if(reload) {
		int radius = VIEW_DISTANCE/VOXEL_X;

		for(int y = -radius; y < radius; y++) {
			for(int x = -radius; x < radius; x++) {
				Vec2i coord = vec2i(y, x);
				// Vec2i coord = vec2i(0,0);	
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coord);
				// m->upToDate = false;
				// m->meshUploaded = false;
				// m->generated = false;
			}
		}
		return;
	}
	#endif

	// TIMER_BLOCK_BEGIN_NAMED(world, "Upd World");

	Vec2i* coordList = (Vec2i*)getTMemory(sizeof(Vec2i)*2000);
	int coordListSize = 0;

	int meshGenerationCount = 0;
	int radCounter = 0;
	int triangleCount = 0;
	int drawCounter = 0;

	Vec2i pPos = coordToMesh(ad->activeCam.pos);
	int radius = VIEW_DISTANCE/VOXEL_X;

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
			Vec3 cp = ad->activeCam.pos;
			Vec3 cl = ad->activeCam.look;
			Vec3 cu = ad->activeCam.up;
			Vec3 cr = ad->activeCam.right;

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

	SortPair* sortList = (SortPair*)getTMemory(sizeof(SortPair)*coordListSize);
	int sortListSize = 0;

	for(int i = 0; i < coordListSize; i++) {
		Vec2 c = meshToMeshCoord(coordList[i]).xy;
		float distanceToCamera = lenVec2(ad->activeCam.pos.xy - c);
		sortList[sortListSize++] = {distanceToCamera, i};
	}

	radixSortPair(sortList, sortListSize);

	for(int i = 0; i < sortListSize-1; i++) {
		assert(sortList[i].key <= sortList[i+1].key);
	}

	// TIMER_BLOCK_END(world);

	{
		// TIMER_BLOCK_NAMED("D World");
		// draw world without water
		{
			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
				drawVoxelMesh(m, 2);
			}
		}

		// draw stencil
		{
			glEnable(GL_STENCIL_TEST);
			glStencilMask(0xFF);
			glClear(GL_STENCIL_BUFFER_BIT);
			glStencilFunc(GL_ALWAYS, 1, 0xFF);
			glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
			glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
			glDepthMask(GL_FALSE);

			glEnable(GL_CLIP_DISTANCE0);
			glEnable(GL_CLIP_DISTANCE1);

			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CLIPPLANE, true);
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CPLANE1, 0,0,1,-WATER_LEVEL_HEIGHT);
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CPLANE2, 0,0,-1,WATER_LEVEL_HEIGHT);

			pushUniform(SHADER_VOXEL, 1, VOXEL_UNIFORM_ALPHATEST, 0.5f);

			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
				drawVoxelMesh(m, 1);
			}

			glDisable(GL_CLIP_DISTANCE0);
			glDisable(GL_CLIP_DISTANCE1);
			glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
			glDepthMask(GL_TRUE);
		}

		// draw reflection
		{	
			glStencilMask(0x00);
			glStencilFunc(GL_EQUAL, 1, 0xFF);

			glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[2]);
			glClearColor(0,0,0,0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

			Vec2i reflectionRes = ad->curRes;
			glBlitNamedFramebuffer (ad->frameBuffers[0], ad->frameBuffers[2], 
				0,0, ad->curRes.x, ad->curRes.y,
				0,0, ad->curRes.x, ad->curRes.y,
				                   // 0,0, reflectionRes.x-1, reflectionRes.y-1,
				                   // 0,0, ad->curRes.x*0.5f, ad->curRes.y*0.5f,
				                   // GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT,
				GL_STENCIL_BUFFER_BIT,
				GL_NEAREST);

			glEnable(GL_CLIP_DISTANCE0);
			// glEnable(GL_CLIP_DISTANCE1);
			glEnable(GL_DEPTH_TEST);
			glFrontFace(GL_CCW);

				// draw cubemap reflection
				bindShader(SHADER_CUBEMAP);
				glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
				glBindTextures(0, 1, &ad->cubemapTextureId[ad->cubeMapDrawIndex]);
				glBindSamplers(0, 1, ad->samplers);

				Vec3 skyBoxRot;
				if(ad->playerMode) skyBoxRot = ad->player->rot;
				else skyBoxRot = ad->cameraEntity->rot;
				skyBoxRot.x += M_PI;

				Camera skyBoxCam = getCamData(vec3(0,0,0), skyBoxRot, vec3(0,0,0), vec3(0,1,0), vec3(0,0,1));

				Mat4 viewMat; viewMatrix(&viewMat, skyBoxCam.pos, -skyBoxCam.look, skyBoxCam.up, skyBoxCam.right);
				Mat4 projMat; projMatrix(&projMat, degreeToRadian(ad->fieldOfView), ad->aspectRatio, 0.001f, 2);
				pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_VIEW, viewMat.e);
				pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_PROJ, projMat.e);

				pushUniform(SHADER_CUBEMAP, 2, CUBEMAP_UNIFORM_CLIPPLANE, true);
				pushUniform(SHADER_CUBEMAP, 0, CUBEMAP_UNIFORM_CPLANE1, 0,0,-1,WATER_LEVEL_HEIGHT);

				glDepthMask(false);
				// glFrontFace(GL_CCW);
				glDrawArrays(GL_TRIANGLES, 0, 6*6);
				// glFrontFace(GL_CW);
				glDepthMask(true);
				glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

			setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, fogColor, vec3(0,0,WATER_LEVEL_HEIGHT*2 + 0.01f), vec3(1,1,-1));
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CLIPPLANE, true);
			pushUniform(SHADER_VOXEL, 0, VOXEL_UNIFORM_CPLANE1, 0,0,-1,WATER_LEVEL_HEIGHT);

			for(int i = 0; i < sortListSize; i++) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
				drawVoxelMesh(m, 2);
			}
			for(int i = sortListSize-1; i >= 0; i--) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
				drawVoxelMesh(m, 1);
			}

			glFrontFace(GL_CW);
			glDisable(GL_CLIP_DISTANCE0);
			// glDisable(GL_CLIP_DISTANCE1);
			glDisable(GL_STENCIL_TEST);
		}

		// draw reflection texture	
		{ 
			// 	// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			// 	// glBlendFunc(GL_SRC_COLOR, GL_ONE);
			// 	// glBlendFunc(GL_SRC_COLOR, GL_ONE_MINUS_SRC_ALPHA);
			// 	// glBlendFunc(GL_DST_COLOR, GL_ZERO);
			// 	// glBlendFunc(GL_SRC_COLOR, GL_ONE);

			// 	// glBlendFunc(GL_DST_COLOR, GL_ZERO);
			// 	// glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);
			// 	// glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);

			// 	// glBlendFunc(GL_ONE, GL_DST_COLOR);
			// 	// glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);
			// 	// glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);

			// 	// GL_DST_COLOR, GL_ZERO

			// 	// void glBlendFuncSeparate(	GLenum srcRGB,
			// 	//  	GLenum dstRGB,
			// 	//  	GLenum srcAlpha,
			// 	//  	GLenum dstAlpha);

			// // glBlendFuncSeparate(GL_ONE_MINUS_DST_COLOR, GL_ZERO, GL_ONE, GL_ONE);
			// // glBlendFuncSeparate(GL_ONE_MINUS_DST_COLOR, GL_ZERO, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			// glBlendFuncSeparate(GL_ONE_MINUS_DST_COLOR, GL_ZEROd, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


			// 	glBlendEquation(GL_FUNC_ADD);
			// 	// glBlendEquation(GL_FUNC_SUBTRACT);
			// 	// glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
			// 	// glBlendEquation(GL_MIN);
			// 	// glBlendEquation(GL_MAX);

			glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[0]);
			glDisable(GL_DEPTH_TEST);

			bindShader(SHADER_QUAD);
			drawRect(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0), rect(0,1,1,0), vec4(1,1,1,reflectionAlpha), ad->frameBufferTextures[1]);

			glEnable(GL_DEPTH_TEST);

			// 	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			// 	glBlendEquation(GL_FUNC_ADD);
		}

		// draw water
		{
			setupVoxelUniforms(vec4(ad->activeCam.pos, 1), 0, 1, 2, view, proj, fogColor);
			pushUniform(SHADER_VOXEL, 1, VOXEL_UNIFORM_ALPHATEST, 0.5f);

			for(int i = sortListSize-1; i >= 0; i--) {
				VoxelMesh* m = getVoxelMesh(ad->voxelHash, ad->voxelHashSize, coordList[sortList[i].index]);
				drawVoxelMesh(m, 1);
			}
		}
	}


	// Vec3 off = vec3(0.5f, 0.5f, 0.5f);
	// Vec3 s = vec3(1.01f, 1.01f, 1.01f);

	// for(int i = 0; i < 10; i++) dcCube({vec3(i*10,0,0) + off, s, vec4(0,1,1,1), 0, vec3(1,2,3)});
	// for(int i = 0; i < 10; i++) dcCube({vec3(0,i*10,0) + off, s, vec4(0,1,1,1), 0, vec3(1,2,3)});
	// for(int i = 0; i < 10; i++) dcCube({vec3(0,0,i*10) + off, s, vec4(0,1,1,1), 0, vec3(1,2,3)});

	dcState(STATE_LINEWIDTH, 3);
	if(!ad->playerMode) {
		Camera cam = getCamData(ad->player->pos, ad->player->rot);
		Vec3 pCamPos = player->pos + player->camOff;
		float lineLength = 0.5f;

		dcLine(pCamPos, pCamPos + cam.look*lineLength, vec4(1,0,0,1));
		dcLine(pCamPos, pCamPos + cam.up*lineLength, vec4(0,1,0,1));
		dcLine(pCamPos, pCamPos + cam.right*lineLength, vec4(0,0,1,1));

		dcState(STATE_POLYGONMODE, POLYGON_MODE_LINE);
		dcCube(player->pos, player->dim, vec4(1,1,1,1), 0, vec3(0,0,0));
		dcState(STATE_POLYGONMODE, POLYGON_MODE_FILL);
	} else {
		if(ad->blockSelected) {
			dcDisable(STATE_CULL);
			Vec3 vs[4];
			getPointsFromQuadAndNormal(ad->selectedBlock + ad->selectedBlockFaceDir*0.5f*1.01f, ad->selectedBlockFaceDir, 1, vs);
			dcQuad(vs[0], vs[1], vs[2], vs[3], vec4(1,1,1,0.025f));
			dcEnable(STATE_CULL);

			dcState(STATE_POLYGONMODE, POLYGON_MODE_LINE);
			dcCube(ad->selectedBlock, vec3(1.01f), vec4(0.9f), 0, vec3(0,0,0));
			dcState(STATE_POLYGONMODE, POLYGON_MODE_FILL);
		}
	}



	bindShader(SHADER_CUBE);

	Vec3 ep = vec3(0,0,80);

	static ParticleEmitter emitter;
	static bool emitterInit = true; 
	if(emitterInit) {
		emitter = {};
		// emitter.particleListSize = 1024;
		// emitter.particleListSize = 100000;
		emitter.particleListSize = 100000;
		emitter.particleList = getPArray(Particle, emitter.particleListSize);
		// emitter.spawnRate = 0.0001f;
		// emitter.spawnRate = 0.001f;
		emitter.spawnRate = 0.005f;
		// emitter.spawnRate = 0.0001f;

		emitter.pos = vec3(0,0,70);
		emitter.friction = 0.5f;

		emitterInit = false;
	}

	static float dt = 0;
	// dt += ad->dt;
	emitter.pos = ep + vec3(sin(dt),0,0);
	// drawCube(emitter.pos, vec3(0.5f), vec4(0,0,0,0.2f), 0, vec3(0,0,0));

	// if(0)
	{
		ParticleEmitter* e = &emitter;
		float dt = ad->dt;

		e->dt += dt;
		while(e->dt >= 0.1f) {
			e->dt -= e->spawnRate;

			if(e->particleListCount < e->particleListSize) {
				Particle p = {};

				p.pos = e->pos;
				Vec3 dir = normVec3(vec3(randomFloat(-1,1,0.01f), randomFloat(-1,1,0.01f), randomFloat(-1,1,0.01f)));
				// p.vel = dir * 1.0f;
				p.vel = dir * 5.0f;
				// p.acc = dir*0.2f;
				// p.acc = -dir*0.2f;
				p.acc = -dir*1.0f;

				// p.color = vec4(0.8f, 0.1f, 0.6f, 1.0f);
				// p.accColor = vec4(-0.15f,0,0.15f,-0.05f);
				p.color = vec4(0.8f, 0.8f, 0.1f, 1.0f);
				p.accColor = vec4(+0.10f,-0.15f,0,-0.05f);


				// p.size = vec3(0.1f);
				p.size = vec3(0.1f, 0.1f, 0.005f);

				p.rot = 0;
				p.rot2 = degreeToRadian(randomInt(0,360));
				p.velRot = 20.0f;
				p.accRot = -4.0f;

				p.timeToLive = 5;
				// p.timeToLive = randomFloat(2.0f,6.0f, 0.01f);

				e->particleList[e->particleListCount++] = p;
			}
		}
	}

	particleEmitterUpdate(&emitter, ad->dt);

	// glDisable(GL_CULL_FACE);

	// Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_UV, quadUVs, 4);
	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, true);

	// uint tex[2] = {getTexture(TEXTURE_CIRCLE)->id, 0};
	// glBindTextures(0,2,tex);

	// for(int i = 0; i < emitter.particleListCount; i++) {
	// 	Particle* p = emitter.particleList + i;

	// 	Vec3 normal = normVec3(p->vel);

	// 	float size = 0.1f;
	// 	Vec4 color = p->color;
	// 	Vec3 base = p->pos;

	// 	Vec3 dir1 = normVec3(cross(normal, vec3(1,0,0)));
	// 	rotateVec3(&dir1, p->rot2, normal);
	// 	rotateVec3(&normal, p->rot, dir1);
	// 	Vec3 dir2 = normVec3(cross(normal, dir1));

	// 	dir1 *= size*0.5f;
	// 	dir2 *= size*0.5f;

	// 	Vec3 verts[4] = {base + dir1+dir2, base + dir1+(-dir2), base + (-dir1)+(-dir2), base + (-dir1)+dir2};

	// 	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VERTICES, verts[0].e, arrayCount(verts));
	// 	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, &color);

	// 	glDrawArrays(GL_QUADS, 0, arrayCount(verts));
	// }
	// glEnable(GL_CULL_FACE);



	// glDisable(GL_CULL_FACE);

	// // Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
	// // pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_UV, quadUVs, 4);
	// // pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, true);

	// uint tex[2] = {getTexture(TEXTURE_CIRCLE)->id, 0};
	// glBindTextures(0,2,tex);

	// Mesh* mesh = getMesh(MESH_QUAD);
	// glBindBuffer(GL_ARRAY_BUFFER, mesh->bufferId);

	// glVertexAttribPointer(0, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)0);
	// glEnableVertexAttribArray(0);
	// glVertexAttribPointer(1, 2, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3)));
	// glEnableVertexAttribArray(1);
	// glVertexAttribPointer(2, 3, GL_FLOAT, 0, sizeof(Vertex), (void*)(sizeof(Vec3) + sizeof(Vec2)));
	// glEnableVertexAttribArray(2);

	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODE, false);

	// for(int i = 0; i < emitter.particleListCount; i++) {
	// 	Particle* p = emitter.particleList + i;

	// 	Vec3 normal = normVec3(p->vel);

	// 	float size = 0.1f;
	// 	Vec4 color = p->color;
	// 	Vec3 base = p->pos;

	// 	Vec3 dir1 = normVec3(cross(normal, vec3(1,0,0)));
	// 	rotateVec3(&dir1, p->rot2, normal);
	// 	rotateVec3(&normal, p->rot, dir1);
	// 	Vec3 dir2 = normVec3(cross(normal, dir1));

	// 	dir1 *= size*0.5f;
	// 	dir2 *= size*0.5f;

	// 	// Vec3 verts[4] = {base + dir1+dir2, base + dir1+(-dir2), base + (-dir1)+(-dir2), base + (-dir1)+dir2};

	// 	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_VERTICES, verts[0].e, arrayCount(verts));
	// 	// pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, &color);

	// 	// glDrawArrays(GL_QUADS, 0, arrayCount(verts));



	// 	Mat4 model = modelMatrix(p->pos, p->size, p->rot, normal);
	// 	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_MODEL, model.e);
	// 	pushUniform(SHADER_CUBE, 0, CUBE_UNIFORM_COLOR, color.e);

	// 	glDrawArrays(GL_QUADS, 0, mesh->vertCount);

	// }
	// glEnable(GL_CULL_FACE);




	#if 1


		bindShader(SHADER_PARTICLE);
		pushUniform(SHADER_PARTICLE, 0, PARTICLE_UNIFORM_VIEW, view);
		pushUniform(SHADER_PARTICLE, 0, PARTICLE_UNIFORM_PROJ, proj);

		uint tex[2] = {getTexture(TEXTURE_CIRCLE)->id, 0};
		glBindTextures(0,1,tex);
		glBindSamplers(0,1,ad->samplers);


		static float timer = 1;
		timer += ad->dt;

		// if(timer >= 1) {

			int bufferOffset = 0;

			for(int i = 0; i < emitter.particleListCount; i++) {
				Particle* p = emitter.particleList + i;
				Vec3 normal = normVec3(p->vel);

				Mat4 model = modelMatrix(p->pos, p->size, p->rot, normal);
				// Mat4 model = modelMatrix(p->pos, vec3(p->size.x, p->size.x, p->size.x), p->rot, normal);
				rowToColumn(&model);

				memCpy(ad->testBuffer + bufferOffset, model.e, sizeof(model)); bufferOffset += sizeof(model);
				memCpy(ad->testBuffer + bufferOffset, p->color.e, sizeof(p->color)); bufferOffset += sizeof(p->color);
			}

			// printf("%i \n", emitter.particleListCount);


			timer = 0;
			glNamedBufferData(ad->testBufferId, ad->testBufferSize, 0, GL_STREAM_DRAW);
			glNamedBufferSubData(ad->testBufferId, 0, bufferOffset, ad->testBuffer);
		// }

		// glNamedBufferData(ad->testBufferId, ad->testBufferSize, 0, GL_STREAM_DRAW);
		// glNamedBufferSubData(ad->testBufferId, 0, bufferOffset, ad->testBuffer);

		Vec3 verts[] = {vec3(-0.5f, -0.5f, 0),vec3(-0.5f, 0.5f, 0),vec3(0.5f, 0.5f, 0),vec3(0.5f, -0.5f, 0)};
		glProgramUniform3fv(getShader(SHADER_PARTICLE)->vertex, 0, 4, verts[0].e);
		
		Vec2 quadUVs[] = {{0,0}, {0,1}, {1,1}, {1,0}};
		glProgramUniform2fv(getShader(SHADER_PARTICLE)->vertex, 4, 4, quadUVs[0].e);

		glBindBuffer(GL_ARRAY_BUFFER, ad->testBufferId);

		glEnableVertexAttribArray(8);
		glVertexAttribPointer(8, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)(sizeof(Mat4)));

		glEnableVertexAttribArray(9);
		glVertexAttribPointer(9, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)0);
		glEnableVertexAttribArray(10);
		glVertexAttribPointer(10, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)(sizeof(Vec4)*1));
		glEnableVertexAttribArray(11);
		glVertexAttribPointer(11, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)(sizeof(Vec4)*2));
		glEnableVertexAttribArray(12);
		glVertexAttribPointer(12, 4, GL_FLOAT, 0, sizeof(Mat4)+sizeof(Vec4), (void*)(sizeof(Vec4)*3));

		glVertexAttribDivisor(8, 1);
		glVertexAttribDivisor(9, 1);
		glVertexAttribDivisor(10, 1);
		glVertexAttribDivisor(11, 1);
		glVertexAttribDivisor(12, 1);

		glDisable(GL_CULL_FACE);
		// glDrawArrays(GL_QUADS, 0, emitter.particleListCount * 4);
		glDrawArraysInstanced(GL_QUADS, 0, 4, emitter.particleListCount);
		// glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, emitter.particleListCount);
		glEnable(GL_CULL_FACE);

	#endif 



		// dcRect({rectCenDim(400,-400,200,200), rect(0,0,1,1), rColor, getTexture(TEXTURE_WHITE)->id});
	// dcRect({rectCenDim(});
	// dcRect({rectCenDim(});

		// dcText({fillString("Pos  : (%f,%f,%f)", PVEC3(ad->activeCam.pos)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow});

	// Vec2 tPos = vec2(900,-300);
	// char* text = "This is a Test String!";
	// // Font* f = getFont(FONT_ARIAL);
	// Font* f = getFont(FONT_ARIAL, 40);
	// dcText({text, f, tPos, vec4(1,1,0,1), 0, 2, 1});
	// // dcRect({rectCenDim(tPos-vec2(0,10), vec2(1,20)), rect(0,0,1,1), vec4(1,0,0,1), getTexture(TEXTURE_WHITE)->id});

	// float xOff = getTextPos(text, 5, f);
	// dcRect({rectCenDim(tPos + vec2(xOff,-20), vec2(1,40)), rect(0,0,1,1), vec4(1,0,0,1), getTexture(TEXTURE_WHITE)->id});

	globalCommandList = &ad->commandList2d;

	if(input->keysPressed[VK_F5]) ad->showHud = !ad->showHud;

	if(ad->showHud) {
		int fontSize = 18;
		int pi = 0;
		// Vec4 c = vec4(1.0f,0.2f,0.0f,1);
		Vec4 c = vec4(1.0f,0.4f,0.0f,1);
		Vec4 c2 = vec4(0,0,0,1);
		Font* font = getFont(FONT_CONSOLAS, fontSize);
		// Font* font = getFont(FONT_CALIBR, fontSize);
		int shadow = 1;
		// float shadow = 0;
		float xo = 6;
		int ali = 2;

		Vec2i tp = ad->wSettings.currentRes - vec2i(xo, 0);
		#define PVEC3(v) v.x, v.y, v.z
		#define PVEC2(v) v.x, v.y
		dcText(fillString("Pos  : (%f,%f,%f)", PVEC3(ad->activeCam.pos)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
		dcText(fillString("Look : (%f,%f,%f)", PVEC3(ad->activeCam.look)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
		dcText(fillString("Up   : (%f,%f,%f)", PVEC3(ad->activeCam.up)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
		dcText(fillString("Right: (%f,%f,%f)", PVEC3(ad->activeCam.right)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
		dcText(fillString("Rot  : (%f,%f)", 	PVEC2(player->rot)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
		dcText(fillString("Vec  : (%f,%f,%f)", PVEC3(player->vel)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
		dcText(fillString("Acc  : (%f,%f,%f)", PVEC3(player->acc)), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
		dcText(fillString("Draws: (%i)", 		drawCounter), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);
		dcText(fillString("Quads: (%i)", 		triangleCount), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);

		dcText(fillString("Threads: (%i, %i)",	threadQueue->completionCount, threadQueue->completionGoal), font, vec2(tp.x,-fontSize*pi++), c, ali, 2, shadow);




		globalCommandList = &ad->commandListGui;

		bool initSections = false;

		Gui* gui = &ad->gui;
		static bool setupGui = true;
		if(second) {
			// gui->init(rectCenDim(vec2(0,1), vec2(300,800)));
			// gui->init(rectCenDim(vec2(1300,1), vec2(300,500)));
			gui->init(rectCenDim(vec2(1300,1), vec2(300, wSettings->currentRes.h)), 0);
			setupGui = false;
		}
		GuiInput gInput = { vec2(input->mousePos), input->mouseWheel, input->mouseButtonPressed[0], input->mouseButtonDown[0], 
							input->keysPressed[VK_ESCAPE], input->keysPressed[VK_RETURN], input->keysPressed[VK_SPACE], input->keysPressed[VK_BACK], input->keysPressed[VK_DELETE], input->keysPressed[VK_HOME], input->keysPressed[VK_END], 
							input->keysPressed[VK_LEFT], input->keysPressed[VK_RIGHT], input->keysPressed[VK_UP], input->keysPressed[VK_DOWN], 
							input->keysDown[VK_SHIFT], input->keysDown[VK_CONTROL], input->inputCharacters, input->inputCharacterCount};
		// gui->start(gInput, getFont(FONT_CONSOLAS, fontSize), wSettings->currentRes);
		gui->start(gInput, getFont(FONT_CALIBRI, fontSize), wSettings->currentRes);

		static bool sectionGuiSettings = false;
		if(gui->beginSection("GuiSettings", &sectionGuiSettings)) {
			guiSettings(gui);
		} gui->endSection();

		static bool sectionSettings = initSections;
		if(gui->beginSection("Settings", &sectionSettings)) {
			gui->div(vec2(0,0)); if(gui->button("Compile")) shellExecute("C:\\Projects\\Hmm\\code\\buildWin32.bat");
								if(gui->button("Up Buffers")) ad->updateFrameBuffers = true;
			gui->div(vec2(0,0)); gui->label("GuiAlpha", 0); gui->slider(&ad->guiAlpha, 0.1f, 1);
			gui->div(vec2(0,0)); gui->label("FoV", 0); gui->slider(&ad->fieldOfView, 1, 180);
			gui->div(vec2(0,0)); gui->label("MSAA", 0); gui->slider(&ad->msaaSamples, 1, 8);
			gui->switcher("Native Res", &ad->useNativeRes);
			gui->div(0,0,0); gui->label("FboRes", 0); gui->slider(&ad->fboRes.x, 150, ad->curRes.x); gui->slider(&ad->fboRes.y, 150, ad->curRes.y);
			gui->div(0,0,0); gui->label("NFPlane", 0); gui->slider(&ad->nearPlane, 0.01, 2); gui->slider(&ad->farPlane, 1000, 5000);
		} gui->endSection();

		static bool sectionWorld = initSections;
		if(gui->beginSection("World", &sectionWorld)) { 
			if(gui->button("Reload World") || input->keysPressed[VK_TAB]) ad->reloadWorld = true;
			
			gui->div(vec2(0,0)); gui->label("CubeMap", 0); gui->slider(&ad->cubeMapDrawIndex, 0, ad->cubeMapCount);

			gui->div(vec2(0,0)); gui->label("RefAlpha", 0); gui->slider(&reflectionAlpha, 0, 1);
			gui->div(vec2(0,0)); gui->label("Light", 0); gui->slider(&globalLumen, 0, 255);
			gui->div(0,0,0); gui->label("MinMax", 0); gui->slider(&WORLD_MIN, 0, 255); gui->slider(&WORLD_MAX, 0, 255);
			gui->div(vec2(0,0)); gui->label("WaterLevel", 0); 
				if(gui->slider(&waterLevelValue, 0, 0.2f)) WATER_LEVEL_HEIGHT = lerp(waterLevelValue, WORLD_MIN, WORLD_MAX);

			gui->div(vec2(0,0)); gui->label("WFreq", 0); gui->slider(&worldFreq, 0.0001f, 0.02f);
			gui->div(vec2(0,0)); gui->label("WDepth", 0); gui->slider(&worldDepth, 1, 10);
			gui->div(vec2(0,0)); gui->label("MFreq", 0); gui->slider(&modFreq, 0.001f, 0.1f);
			gui->div(vec2(0,0)); gui->label("MDepth", 0); gui->slider(&modDepth, 1, 10);
			gui->div(vec2(0,0)); gui->label("MOffset", 0); gui->slider(&modOffset, 0, 1);
			gui->div(vec2(0,0)); gui->label("PowCurve", 0); gui->slider(&worldPowCurve, 1, 6);
			gui->div(0,0,0,0); gui->slider(heightLevels+0,0,1); gui->slider(heightLevels+1,0,1);
								  gui->slider(heightLevels+2,0,1); gui->slider(heightLevels+3,0,1);
		} gui->endSection();

		static bool sectionTest = false;
		if(gui->beginSection("Test", &sectionTest)) {
			static int scrollHeight = 200;
			static int scrollElements = 13;
			static float scrollVal = 0;
			gui->div(vec2(0,0)); gui->slider(&scrollHeight, 0, 2000); gui->slider(&scrollElements, 0, 100);

			gui->beginScroll(scrollHeight, &scrollVal); {
				for(int i = 0; i < scrollElements; i++) {
					gui->button(fillString("Element: %i.", i));
				}
			} gui->endScroll();

			static int textCapacity = 50;
			static char* text = getPArray(char, textCapacity);
			static bool DOIT = true;
			if(DOIT) {
				DOIT = false;
				strClear(text);
				strCpy(text, "This is a really long sentence!");
			}
			gui->div(vec2(0,0)); gui->label("Text Box:", 0); gui->textBoxChar(text, 0, textCapacity);

			static int textNumber = 1234;
			gui->div(vec2(0,0)); gui->label("Int Box:", 0); gui->textBoxInt(&textNumber);

			static float textFloat = 123.456f;
			gui->div(vec2(0,0)); gui->label("Float Box:", 0); gui->textBoxFloat(&textFloat);

		} gui->endSection();

		gui->end();

		globalCommandList = &ad->commandList2d;
	}




	// @menu
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
			// dcRect({rectCenDim(pos, iconSize*1.2f*iconOff), rect(0,0,1,1), vec4(vec3(0.1f),trans), 1});
			dcRect(rectCenDim(pos, iconSize*1.2f*iconOff), rect(0,0,1,1), vec4(0,0,0,0.85f), 1);

			uint textureId = ad->voxelTextures[0];
			dcRect(rectCenDim(pos, iconSize*iconOff), rect(0,0,1,1), color, (int)textureId, texture1Faces[i+1][0]+1);
		}
	}

	bindShader(SHADER_CUBE);

	executeCommandList(&ad->commandList3d);

	// ortho(rectCenDim(cam->x,cam->y, cam->z, cam->z/ad->aspectRatio));
	// bindShader(SHADER_QUAD);
	// drawRect(rectCenDim(0, 0, 0.01f, 100), rect(0,0,1,1), vec4(0.4f,1,0.4f,1), ad->textures[0]);
	// drawRect(rectCenDim(0, 0, 100, 0.01f), rect(0,0,1,1), vec4(0.4f,0.4f,1,1), ad->textures[0]);

	// drawRect(rectCenDim(0, 0, 5, 5), rect(0,0,1,1), vec4(1,1,1,1), ad->textures[2]);
	// drawRect(rectCenDim(0, 0, 5, 5), rect(0,0,1,1), vec4(1,1,1,1), 3);

	// drawRect(rect(2,2,4,4), rect(0,0,1,1), vec4(1,1,0,1), 2);

	ortho(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0));
	glDisable(GL_DEPTH_TEST);
	bindShader(SHADER_QUAD);

	glBlitNamedFramebuffer(ad->frameBuffers[0], ad->frameBuffers[1],
		0,0, ad->curRes.x, ad->curRes.y,
		0,0, ad->curRes.x, ad->curRes.y,
		                   // GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
		GL_COLOR_BUFFER_BIT,
		                   // GL_NEAREST);
		GL_LINEAR);

	glDisable(GL_DEPTH_TEST);
	bindShader(SHADER_QUAD);
	glBindFramebuffer (GL_FRAMEBUFFER, ad->frameBuffers[3]);
	glViewport(0,0, wSettings->currentRes.x, wSettings->currentRes.y);
	drawRect(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0), rect(0,1,1,0), vec4(1), ad->frameBufferTextures[0]);

	executeCommandList(&ad->commandList2d);




	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);

	glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[4]);
	executeCommandList(&ad->commandListGui);
	{
		TIMER_BLOCK_NAMED("asdf");
		executeCommandList(&ad->commandListDebug);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[3]);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendEquation(GL_FUNC_ADD);

	drawRect(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0), rect(0,1,1,0), vec4(1,1,1,ad->guiAlpha), ad->frameBufferTextures[5]);





	if(USE_SRGB) glEnable(GL_FRAMEBUFFER_SRGB);
	glBindFramebuffer (GL_FRAMEBUFFER, 0);
	bindShader(SHADER_QUAD);
	glBindSamplers(0, 1, ad->samplers);

	drawRect(rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0), rect(0,1,1,0), vec4(1), ad->frameBufferTextures[4]);
	if(USE_SRGB) glDisable(GL_FRAMEBUFFER_SRGB);




	#if 0
	stbvox_mesh_maker mm;
	stbvox_init_mesh_maker(&mm);
	stbvox_input_description* inputDesc = stbvox_get_input_description(&mm);
	*inputDesc = {};

	int meshBufferCapacity = 100;
	char* meshBuffer = (char*)getTMemory(meshBufferCapacity);
	int texBufferCapacity = meshBufferCapacity/4;
	char* texBuffer = (char*)getTMemory(texBufferCapacity);

	int meshBufferTransCapacity = 100;
	char* meshBufferTrans = (char*)getTMemory(meshBufferTransCapacity);
	int texBufferTransCapacity = meshBufferTransCapacity/4;
	char* texBufferTrans = (char*)getTMemory(texBufferTransCapacity);

	stbvox_set_buffer(&mm, 0, 0, meshBuffer, meshBufferCapacity);
	stbvox_set_buffer(&mm, 1, 0, meshBufferTrans, meshBufferTransCapacity);
	stbvox_set_buffer(&mm, 0, 1, texBuffer, texBufferCapacity);
	stbvox_set_buffer(&mm, 1, 1, texBufferTrans, texBufferTransCapacity);

	inputDesc->block_tex2 = texture2;
	inputDesc->block_tex1_face = texture1Faces;
	inputDesc->block_geometry = geometry;
	inputDesc->block_selector = meshSelection;

	uchar color[BT_Size];
	for(int i = 0; i < BT_Size; i++) color[i] = STBVOX_MAKE_COLOR(blockColor[i], 1, 0);
	inputDesc->block_color = color;

	unsigned char tLerp[50] = {};
	// for(int i = 1; i < arrayCount(tLerp)-1; i++) tLerp[i] = 6;
	// tLerp[10] = 4;
	inputDesc->block_texlerp = tLerp;



	stbvox_set_input_stride(&mm, 3*3,3);
	stbvox_set_input_range(&mm, 0,0,0, 1,1,1);

	uchar voxels[3][3][3] = {};
	voxels[1][1][1] = BT_Water;

	uchar lighting[3][3][3] = {};
	for(int i = 0; i < 27; i++) *((&lighting[0][0][0])+i) = 255;
	lighting[1][1][1] = 0;

	inputDesc->blocktype = &voxels[1][1][1];
	inputDesc->lighting = &lighting[1][1][1];

	int success = stbvox_make_mesh(&mm);


	// Vec3 pos = ad->activeCam.pos + ad->activeCam.look * 20;
	// stbvox_set_mesh_coordinates(&mm, pos.x, pos.y, pos.z);
	stbvox_set_mesh_coordinates(&mm, 0,0,30);

	float transform[3][3];
	stbvox_get_transform(&mm, transform);

	int quadCount = stbvox_get_quad_count(&mm, 0);
	int quadCountTrans = stbvox_get_quad_count(&mm, 1);

	int bufferSizePerQuad = stbvox_get_buffer_size_per_quad(&mm, 0);
	int textureBufferSizePerQuad = stbvox_get_buffer_size_per_quad(&mm, 1);


	static bool setup = true;
	static uint textureId, textureTransId, bufferId, texBufferId, bufferTransId, texBufferTransId;
	if(setup) {
		glCreateBuffers(1, &bufferId);
		glCreateBuffers(1, &bufferTransId);
		glCreateBuffers(1, &texBufferId);
		glCreateTextures(GL_TEXTURE_BUFFER, 1, &textureId);
		glCreateBuffers(1, &texBufferTransId);
		glCreateTextures(GL_TEXTURE_BUFFER, 1, &textureTransId);
		setup = false;
	}

	glNamedBufferData(bufferId, bufferSizePerQuad*quadCount, meshBuffer, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, bufferId);
	glNamedBufferData(texBufferId, textureBufferSizePerQuad*quadCount, texBuffer, GL_STATIC_DRAW);
	glTextureBuffer(textureId, GL_RGBA8UI, texBufferId);
	glNamedBufferData(bufferTransId, bufferSizePerQuad*quadCountTrans, meshBufferTrans, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, bufferTransId);
	glNamedBufferData(texBufferTransId, textureBufferSizePerQuad*quadCountTrans, texBufferTrans, GL_STATIC_DRAW);
	glTextureBuffer(textureTransId, GL_RGBA8UI, texBufferTransId);






	GLuint transformUniform1 = glGetUniformLocation(globalGraphicsState->pipelineIds.voxelVertex, "transform");
	glProgramUniform3fv(globalGraphicsState->pipelineIds.voxelVertex, transformUniform1, 3, transform[0]);
	GLuint transformUniform2 = glGetUniformLocation(globalGraphicsState->pipelineIds.voxelFragment, "transform");
	glProgramUniform3fv(globalGraphicsState->pipelineIds.voxelFragment, transformUniform2, 3, transform[0]);



	glBindBuffer(GL_ARRAY_BUFFER, bufferId);
	int vaLoc = glGetAttribLocation(globalGraphicsState->pipelineIds.voxelVertex, "attr_vertex");
	glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 4, (void*)0);
	glEnableVertexAttribArray(vaLoc);

	globalGraphicsState->textureUnits[2] = textureId;
	glBindTextures(0,16,globalGraphicsState->textureUnits);
	glBindSamplers(0,16,globalGraphicsState->samplerUnits);
	bindShader(globalGraphicsState->pipelineIds.programVoxel);

	glDrawArrays(GL_QUADS, 0, quadCount*4);



	glBindBuffer(GL_ARRAY_BUFFER, bufferTransId);
	vaLoc = glGetAttribLocation(globalGraphicsState->pipelineIds.voxelVertex, "attr_vertex");
	glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 4, (void*)0);
	glEnableVertexAttribArray(vaLoc);

	globalGraphicsState->textureUnits[2] = textureTransId;
	glBindTextures(0,16,globalGraphicsState->textureUnits);
	glBindSamplers(0,16,globalGraphicsState->samplerUnits);
	bindShader(globalGraphicsState->pipelineIds.programVoxel);
		

	glDrawArrays(GL_QUADS, 0, quadCountTrans*4);

	#endif 





	if(USE_SRGB) glEnable(GL_FRAMEBUFFER_SRGB);
	#if 0
	bindShader(SHADER_QUAD);

	// Vec2 size = vec2(800, 800/ad->aspectRatio);
	Vec2 size = vec2(ad->curRes);
	Rect tb = rectCenDim(vec2(ad->curRes.x - size.x*0.5f, -size.y*0.5f), size);
	// Texture* st = getTexture(TEXTURE_TEST);
	// Texture* st = getTexture(TEXTURE_TEST);

	Texture* st = &getFont(FONT_CONSOLAS, 20)->tex;
	Vec2i screenCenter = ad->wSettings.currentRes;
	drawRect(rectCenDim(vec2(screenCenter.x, -screenCenter.y)/2, vec2(st->dim)), rect(0,0,1,1), vec4(1,1,1,1), st->id);


	// glDisable(GL_FRAMEBUFFER_SRGB);

	// Vec2 size = vec2(800, 800/ad->aspectRatio);
	// // Vec2 size = vec2(ad->curRes);
	// Rect tb = rectCenDim(vec2(ad->curRes.x - size.x*0.5f, -size.y*0.5f), size);
	// drawRect(tb, rect(0,0,1,1), vec4(0,0,0,1), ad->textures[TEXTURE_WHITE]);
	// drawRect(tb, rect(0,0,1,1), vec4(1,1,1,1), ad->frameBufferTextures[1]);
	#endif
	if(USE_SRGB) glDisable(GL_FRAMEBUFFER_SRGB);


	if(second) {
		GLenum glError = glGetError(); printf("GLError: %i\n", glError);
	}

	clearTMemory();

	TIMER_BLOCK_END(Main)

	swapBuffers(&ad->systemData);
	glFinish();




	// collate timings 
	{		
		globalCommandList = &ad->commandListDebug;
		ad->commandListDebug.count = 0;
		ad->commandListDebug.bytes = 0;

		int globalTimingsCount = __COUNTER__;

		DebugState* ds = globalDebugState;
		ds->timerInfoCount = globalTimingsCount;
		ds->stringMemoryIndex = 0;

		int bufferIndex = ds->bufferIndex;
		Timings* timings = ds->timings[ds->cycleIndex];
		zeroMemory(timings, ds->timerInfoCount*sizeof(Timings));

		ds->cycleIndex = (ds->cycleIndex + 1)%arrayCount(ds->timings);

		ds->bufferIndex = 0;

		int fontHeight = 18;

		if(reload) {
			for(int i = 0; i < arrayCount(ds->timerInfos); i++) ds->timerInfos[i].initialised = false;
			return;
		}

		// collate timing buffer
		TimerStatistic stats[16] = {};
		int index = 0;

		for(int i = 0; i < bufferIndex; ++i) {
			TimerSlot* slot = ds->timerBuffer + i;

			if(slot->type == TIMER_TYPE_BEGIN) {
				stats[index].cycles = slot->cycles;
				stats[index].timerIndex = slot->timerIndex;
				index++;
			}

			if(slot->type == TIMER_TYPE_END) {
				index--;
				Timings* timing = timings + stats[index].timerIndex;
				timing->cycles += slot->cycles - stats[index].cycles;
				timing->hits++;
			}
		}

		for(int i = 0; i < ds->timerInfoCount; i++) {
			Timings* t = timings + i;
			t->cyclesOverHits = t->hits > 0 ? (u64)(t->cycles/t->hits) : 0; 
		}

		Statistic statistics[32] = {};
		for(int timerIndex = 0; timerIndex < ds->timerInfoCount; timerIndex++) {
			Statistic* stat = statistics + timerIndex;
			beginStatistic(stat);

			for(int i = 0; i < arrayCount(ds->timings); i++) {
				Timings* t = &ds->timings[i][timerIndex];
				updateStatistic(stat, t->cyclesOverHits);
			}

			endStatistic(stat);
		}

		Font* debugFont = getFont(FONT_CALIBRI, 18);

		// draw timing info
		float cyclesPerFrame = (float)((3*((float)1/60))*1024*1024*1024);
		fontHeight = 18;
		Vec2 textPos = vec2(550, -fontHeight);
		int infoCount = ds->timerInfoCount;

		bool initSections = false;

		if(second) {
			ds->gui = getPStruct(Gui);
			// ds->gui->init(rectCenDim(vec2(1300,1), vec2(400, wSettings->currentRes.h)), -1);
			ds->gui->init(rectCenDim(vec2(1300,1), vec2(300, wSettings->currentRes.h)), 3);
		}
		GuiInput gInput = { vec2(input->mousePos), input->mouseWheel, input->mouseButtonPressed[0], input->mouseButtonDown[0], 
							input->keysPressed[VK_ESCAPE], input->keysPressed[VK_RETURN], input->keysPressed[VK_SPACE], input->keysPressed[VK_BACK], input->keysPressed[VK_DELETE], input->keysPressed[VK_HOME], input->keysPressed[VK_END], 
							input->keysPressed[VK_LEFT], input->keysPressed[VK_RIGHT], input->keysPressed[VK_UP], input->keysPressed[VK_DOWN], 
							input->keysDown[VK_SHIFT], input->keysDown[VK_CONTROL], input->inputCharacters, input->inputCharacterCount};
		Gui* gui = ds->gui;
		gui->start(gInput, getFont(FONT_CALIBRI, fontHeight), wSettings->currentRes);

		static bool statsSection = false;
		static bool graphSection = false;
		gui->div(0.2f,0.2f,0); gui->switcher("Stats", &statsSection); gui->switcher("Graph", &graphSection); gui->empty();


		if(statsSection) {
			int barWidth = 1;
			int barCount = arrayCount(ds->timings);
			float sectionWidths[] = {0,0,0,0,0,0,0,0, barWidth*barCount};

			char* headers[] = {"File", "Function", "Description", "Cycles", "Hits", "C/H", "Avg. Cycl.", "Total Time", ""};
			gui->div(sectionWidths, arrayCount(sectionWidths));
			for(int i = 0; i < arrayCount(sectionWidths); i++) gui->label(headers[i],1);

			for(int i = 0; i < infoCount; i++) {
				TimerInfo* tInfo = ds->timerInfos + i;
				Timings* timing = timings + i;

				float cycleCountPercent = (float)timing->cycles/cyclesPerFrame;
				char * percentString = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += 30;
				percentString = floatToStr(percentString, cycleCountPercent*100, 3);

				int debugStringSize = 0;
				char* buffer = 0;

				gui->div(sectionWidths, arrayCount(sectionWidths)); 

				debugStringSize = 30;
				buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%s", tInfo->file + 21);
				gui->label(buffer,0);

				debugStringSize = 30;
				buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%s", tInfo->function);
				gui->label(buffer,0);

				debugStringSize = 30;
				buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%s", tInfo->name);
				gui->label(buffer,0);

				debugStringSize = 30;
				buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%I64uc", timing->cycles);
				gui->label(buffer,2);

				debugStringSize = 30;
				buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%u", timing->hits);
				gui->label(buffer,2);

				debugStringSize = 30;
				buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%I64u", timing->cyclesOverHits);
				gui->label(buffer,2);

				debugStringSize = 30;
				buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%.0fc", statistics[i].avg);
				gui->label(buffer,2);

				debugStringSize = 30;
				buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
				_snprintf_s(buffer, debugStringSize, debugStringSize, "%s%%", percentString);
				gui->label(buffer,2);

				gui->empty();
				Rect r = gui->getCurrentRegion();
				float rheight = gui->getDefaultHeight();

				float xOffset = 0;
				for(int statIndex = 0; statIndex < barCount; statIndex++) {
					Statistic* stat = statistics + i;
					u64 coh = ds->timings[statIndex][i].cyclesOverHits;

					float height = mapRangeClamp(coh, stat->min, stat->max, 1, rheight);
					Vec2 rmin = r.min + vec2(xOffset,-2);
					float colorOffset = mapRange(coh, stat->min, stat->max, 0, 1);
					// dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,0,1-colorOffset,1));
					dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,1-colorOffset,0,1));

					xOffset += barWidth;
				}
			}
		}

		// gui->button("asddsf");

		// float xOffset = 0;
		// for(int statIndex = 0; statIndex < bufferIndex; statIndex++) {
		// 	Statistic* stat = statistics + ds->cycleIndex;
		// 	u64 coh = ds->timings[statIndex][ds->cycleIndex].cyclesOverHits;

		// 	int debugStringSize = 30;
		// 	char* buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
		// 	_snprintf_s(buffer, debugStringSize, debugStringSize, "%I64uc ", coh);
		// 	gui->label(buffer,0);

		// 	// float height = mapRangeClamp(coh, stat->min, stat->max, 1, rheight);
		// 	// Vec2 rmin = r.min + vec2(xOffset,-2);
		// 	// float colorOffset = mapRange(coh, stat->min, stat->max, 0, 1);
		// 	// dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,0,1-colorOffset,1));
		// 	// dcRect(rectMinDim(rmin, vec2(barWidth, height)), vec4(colorOffset,1-colorOffset,0,1));

		// 	// xOffset += barWidth;
		// }





		// save timer buffer
		if(input->keysPressed[VK_F9]) {
			if(!ds->frozenGraph) {
				memCpy(ds->savedTimerBuffer, ds->timerBuffer, bufferIndex*sizeof(TimerSlot));
				ds->savedBufferIndex = bufferIndex;
				memCpy(ds->savedTimings, timings, ds->timerInfoCount*sizeof(Timings));
			}

			ds->frozenGraph = !ds->frozenGraph;
		}


		if(graphSection) {
			static Vec2 trans = vec2(10,0);
			static float zoom = 1;

			gui->div(vec2(0.1f,0)); 
			if(gui->button("Reset")) {
				trans = vec2(10,0);
				zoom = 1;
			}
			gui->empty();
			gui->heightPush(3);
			gui->empty();
			
			Rect bgRect = gui->getCurrentRegion();
			Vec2 dragDelta = vec2(0,0);
			gui->drag(bgRect, &dragDelta, vec4(0,0,0,0));

			trans.x += dragDelta.x;
			// trans.x = clampMax(trans.x, 10);
			gui->heightPop();

			if(gui->input.mouseWheel) {
				zoom *= 1 + gui->input.mouseWheel*0.1f;
				// zoom += gui->input.mouseWheel*10;
			}


			Timings* graphTimings = timings;
			TimerSlot* graphTimerBuffer = ds->timerBuffer;
			u64 graphBufferIndex = bufferIndex;

			if(ds->frozenGraph) {
				graphTimings = ds->savedTimings;
				graphTimerBuffer = ds->savedTimerBuffer;
				graphBufferIndex = ds->savedBufferIndex;
			}

			float graphWidth = rectGetDim(bgRect).w;
			Vec2 startPos = rectGetUL(bgRect);

			if(true) {
				u64 baseCycleCount = graphTimerBuffer[0].cycles;
				u64 startCycleCount = 0;
				u64 endCycleCount = cyclesPerFrame;

				float orthoLeft = 0 + trans.x;
				float orthoRight = graphWidth*zoom + trans.x;

				Rect bgRect = rectULDim(startPos, vec2(graphWidth, fontHeight*3));
				dcRect(bgRect, vec4(1,1,1,0.2f));

				float lineWidth = 1;
				int lineCount = 10;
				for(int i = 0; i < lineCount+1; i++) {
					float linePos = ((graphWidth-20)/(float)lineCount) * i;
					linePos *= zoom;
					linePos += trans.x;
					linePos += bgRect.min.x;
					Vec4 color = vec4(0.7f,0.7f,0.7f,1);
					float lw = lineWidth;
					if(i == 0 || i == lineCount) {
						color = vec4(1,1,1,1);
						lw = lineWidth * 3;
					}
					dcRect(rect(linePos, bgRect.min.y, linePos+lw, bgRect.max.y), color);
				} 

				startPos -= vec2(0, fontHeight);
				index = 0;
				for(int i = 0; i < graphBufferIndex; ++i) {
					TimerSlot* slot = graphTimerBuffer + i;

					if(slot->type == TIMER_TYPE_BEGIN) {

						Timings* t = graphTimings + slot->timerIndex;
						TimerInfo* tInfo = ds->timerInfos + slot->timerIndex;

						float xoff = mapRange(slot->cycles - baseCycleCount, startCycleCount, endCycleCount, orthoLeft, orthoRight);
						float barWidth = mapRange(t->cycles, startCycleCount, endCycleCount, 0, graphWidth*zoom);
						Vec2 pos = startPos + vec2(xoff,index*-fontHeight);
						// Vec2 pos = startPos + vec2(xoff,0);
						
						Rect r = rect(pos, pos + vec2(barWidth,fontHeight));
						float cOff = slot->timerIndex/(float)ds->timerInfoCount;
						Vec4 c = vec4(1-cOff, 0, cOff, 1);

						int debugStringSize = 50;
						char* buffer = &ds->stringMemory[ds->stringMemoryIndex]; ds->stringMemoryIndex += debugStringSize+1;
						_snprintf_s(buffer, debugStringSize, debugStringSize, "%s %s", tInfo->function, tInfo->name);

						gui->drawRect(r, vec4(0,0,0,1));
						gui->drawTextBox(rect(r.min+vec2(1,1), r.max-vec2(1,1)), buffer, c);

						index++;
					}

					if(slot->type == TIMER_TYPE_END) {
						index--;
					}
				}

			}
		}

		gui->end();

		globalCommandList = 0;
	}


	if(*isRunning == false) {
		guiSave(&ad->gui, 2, 0);
		if(globalDebugState->gui) guiSave(globalDebugState->gui, 2, 3);
	}

	{
		ExtendibleMemoryArray* debugMemory = &appMemory->extendibleMemoryArrays[1];
		ExtendibleMemoryArray* pMemory = globalMemory->pMemory;
		
		if(input->keysPressed[VK_F12]) {
			for(int i = 0; i < pMemory->index; i++) {
				memCpy(pMemory->arrays[i].data, debugMemory->arrays[i].data, pMemory->slotSize);
			}
		}
	}

}

POSTMAINFUNCTION(postMain) {
	
}

// #pragma optimize( "", off)
#pragma optimize( "", on ) 

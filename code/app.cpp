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
- Data Package - Streaming
- Expand Font functionality
- Gui

- create simpler windows.h
- remove c runtime library, implement sin, cos...
- savestates, hotrealoading
- pre and post functions to appMain
- memory dynamic expansion
- collision, walking
- select block and delete
- 32x32 voxel chunks


//-------------------------------------
//               BUGS
//-------------------------------------
- Font doesn't draw int
- look has to be negative to work
- app stops reacting to input after a while, hotloading still works though

*/



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
		GLOP(void, FramebufferTexture, GLenum target, GLenum attachment, GLuint texture, GLint level)





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
	const vec3 cube[] = vec3[] (
	    vec3(-0.5f,-0.5f,-0.5f), 
	    vec3( 0.5f,-0.5f,-0.5f), 
	    vec3(-0.5f, 0.5f,-0.5f), 
	    vec3( 0.5f,-0.5f,-0.5f), 
	    vec3( 0.5f, 0.5f,-0.5f),
	    vec3(-0.5f, 0.5f,-0.5f),
	    vec3(-0.5f,-0.5f, 0.5f), 
	    vec3(-0.5f, 0.5f, 0.5f), 
	    vec3( 0.5f,-0.5f, 0.5f), 
	    vec3( 0.5f,-0.5f, 0.5f), 
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3( 0.5f, 0.5f, 0.5f),
	    vec3(-0.5f, 0.5f,-0.5f), 
	    vec3( 0.5f, 0.5f,-0.5f), 
	    vec3(-0.5f, 0.5f, 0.5f), 
	    vec3( 0.5f, 0.5f,-0.5f), 
	    vec3( 0.5f, 0.5f, 0.5f),
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3(-0.5f,-0.5f,-0.5f),
	    vec3(-0.5f,-0.5f, 0.5f), 
	    vec3( 0.5f,-0.5f,-0.5f), 
	    vec3( 0.5f,-0.5f,-0.5f), 
	    vec3(-0.5f,-0.5f, 0.5f),
	    vec3( 0.5f,-0.5f, 0.5f),
	    vec3(-0.5f,-0.5f,-0.5f), 
	    vec3(-0.5f, 0.5f,-0.5f), 
	    vec3(-0.5f,-0.5f, 0.5f), 
	    vec3(-0.5f, 0.5f,-0.5f), 
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3(-0.5f,-0.5f, 0.5f),
	    vec3( 0.5f,-0.5f,-0.5f), 
	    vec3( 0.5f,-0.5f, 0.5f), 
	    vec3( 0.5f, 0.5f,-0.5f), 
	    vec3( 0.5f, 0.5f,-0.5f), 
	    vec3( 0.5f,-0.5f, 0.5f),
	    vec3( 0.5f, 0.5f, 0.5f)
	);

	out gl_PerVertex { vec4 gl_Position; };
	out vec4 Color;

	uniform mat4x4 model;
	uniform mat4x4 view;
	uniform mat4x4 proj;
	// uniform mat4x4 projViewModel;

	uniform vec4 setColor;

	void main() {
		float c = gl_VertexID;
		// Color = vec4(1/(c/36),1/(c/7),1/(c/2),1);
		// Color = vec4(c/36,c/36,c/36,1);
		// setColor = vec4(1,1,1,1);
		// vec4 co = vec4(1,1,1,1);
		Color = setColor;
	
		vec4 pos = vec4(cube[gl_VertexID], 1);
	
		gl_Position = proj*view*model*pos;
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


const char* vertexShaderPrimitive = GLSL (
	out gl_PerVertex { vec4 gl_Position; };
	out vec4 Color;

	uniform mat4x4 view;
	uniform mat4x4 proj;
	uniform vec4 setColor;

	uniform vec3 vertices[4];

	void main() {
		Color = setColor;

		vec4 pos = vec4(vertices[gl_VertexID], 1);
		gl_Position = proj*view*pos;
	}
);

const char* fragmentShaderPrimitive = GLSL (
	in vec4 Color;
	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	void main() {
		color = Color;
	}
);


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

	uint programPrimitive;
	uint primitiveVertex;
	uint primitiveFragment;
	// uint primitiveVertexModel;
	uint primitiveVertexView;
	uint primitiveVertexProj;
	uint primitiveVertexColor;
	uint primitiveVertexVertices;

	uint programVoxel;
	uint voxelVertex;
	uint voxelFragment;
};

void drawRect(PipelineIds* ids, Rect r, Rect uv, Vec4 color, int texture) {
	uint uniformLocation;
	Rect cd = rectGetCenDim(r);
	glProgramUniform4f(ids->quadVertex, ids->quadVertexMod, cd.min.x, cd.min.y, cd.max.x, cd.max.y);
	glProgramUniform4f(ids->quadVertex, ids->quadVertexUV, uv.min.x, uv.max.x, uv.max.y, uv.min.y);
	glProgramUniform4f(ids->quadVertex, ids->quadVertexColor, color.r, color.g, color.b, color.a);
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

void ortho(PipelineIds* ids, Rect r) {
	r = rectGetCenDim(r);
	glProgramUniform4f(1, ids->quadVertexCamera, r.cen.x, r.cen.y, r.dim.w, r.dim.h);
}

void lookAt(PipelineIds* ids, Vec3 pos, Vec3 look, Vec3 up, Vec3 right) {
	Mat4 view;
	viewMatrix(&view, pos, look, up, right);

	glProgramUniformMatrix4fv(ids->cubeVertex, ids->cubeVertexView, 1, 1, view.e);
}

void perspective(PipelineIds* ids, float fov, float aspect, float n, float f) {
	Mat4 proj;
	projMatrix(&proj, fov, aspect, n, f);

	glProgramUniformMatrix4fv(ids->cubeVertex, ids->cubeVertexProj, 1, 1, proj.e);
}

struct Font {
	char* fileBuffer;
	Vec2i size;
	int glyphStart, glyphCount;
	stbtt_bakedchar* cData;
	uint texId;
	int height;
};

void drawText(PipelineIds* ids, Vec2 pos, char* text, Vec4 color, Font* font, int vAlign, int hAlign) {
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
		drawRect(ids, rect(q.x0, q.y0, q.x1, q.y1), rect(q.s0,q.t0,q.s1,q.t1), color, 3);
	}
}

void drawTextA(PipelineIds* ids, Vec2 pos, Vec4 color, Font* font, int vAlign, int hAlign, char* text, ... ) {
	va_list vl;
	va_start(vl, text);

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

		if(t == '%') {
			i++;
			t = text[i];
			if(t != '%') {
				char* valueBuffer = getTString(20);
				Vec2 oPos = vec2(pos.x + (vAlign*0.5f*textDim.w), pos.y + (hAlign*0.5f*textDim.h));

				if(t == 'i') {
					int v = va_arg(vl, int);
					intToStr(valueBuffer, v);
				}

				if(t == 'f') {
					float v = va_arg(vl, double);
					floatToStr(valueBuffer, v, 2);
				}

				drawText(ids, oPos, valueBuffer, color, font, vAlign, hAlign);
				t = text[++i];
				Vec2 dim = stbtt_GetTextDim(font->cData, font->height, font->glyphStart, valueBuffer);
				pos.x += dim.w;
			}
		}

		stbtt_aligned_quad q;
		stbtt_GetBakedQuad(font->cData, font->size.w, font->size.h, t-font->glyphStart, &pos.x, &pos.y, &q, 1);
		drawRect(ids, rect(q.x0, q.y0, q.x1, q.y1), rect(q.s0,q.t0,q.s1,q.t1), color, 3);
	}
}

void drawCube(PipelineIds* ids, Vec3 trans, Vec3 scale, Vec4 color, float degrees, Vec3 rot) {
	// Quat xRot = quat(dt, normVec3(vec3(1,0.7f,0.5f)));
	// Quat yRot = quat(0, vec3(0,1,0));
	// Quat zRot = quat(0, vec3(0,0,1));
	// quatRotationMatrix(&rm, zRot*yRot*xRot);

	Mat4 sm; scaleMatrix(&sm, scale);
	Mat4 rm; quatRotationMatrix(&rm, quat(degrees, rot));
	Mat4 tm; translationMatrix(&tm, trans);
	Mat4 model = tm*rm*sm;

	glProgramUniformMatrix4fv(ids->cubeVertex, ids->cubeVertexModel, 1, 1, model.e);
	glProgramUniform4f(ids->cubeVertex, ids->cubeVertexColor, color.r, color.g, color.b, color.a);

	glDrawArraysInstancedBaseInstance(GL_TRIANGLES, 0, 36, 1, 0);
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

void setupVoxelUniforms(uint vertexShader, uint fragmentShader, Vec4 camera, uint texUnit1, uint texUnit2, uint faceUnit, Mat4 ambient, Mat4 view, Mat4 proj) {

	int texUnit[2] = {texUnit1, texUnit2};

	for (int i=0; i < STBVOX_UNIFORM_count; ++i) {
		stbvox_uniform_info sui;
		if (stbvox_get_uniform_info(&sui, i)) {
			if(i == STBVOX_UNIFORM_transform) continue;

			for(int shaderStage = 0; shaderStage < 2; shaderStage++) {
				GLint location;
				GLuint program;
				if(shaderStage == 0) {
					location = glGetUniformLocation(vertexShader, sui.name);
					program = vertexShader;
				} else {
					location = glGetUniformLocation(fragmentShader, sui.name);
					program = fragmentShader;
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
							data = ambient.e;
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
	GLint modelViewUni = glGetUniformLocation(vertexShader, "model_view");
	glProgramUniformMatrix4fv(vertexShader, modelViewUni, 1, 1, finalMat.e);
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



// #define VIEW_DISTANCE 1024 // 16
// #define VIEW_DISTANCE 512  // 8
#define VIEW_DISTANCE 256 // 4
// #define VIEW_DISTANCE 128 // 2

// #define VIEW_DISTANCE 64*10
// #define VIEW_DISTANCE 64*11
// #define VIEW_DISTANCE 64*12
// #define VIEW_DISTANCE 64*13
// #define VIEW_DISTANCE 64*14
// #define VIEW_DISTANCE 64*15
// #define VIEW_DISTANCE 64*16

#define VOXEL_X 64
#define VOXEL_Y 64
#define VOXEL_Z 254
#define VOXEL_SIZE VOXEL_X*VOXEL_Y*VOXEL_Z
#define VC_X 66
#define VC_Y 66
#define VC_Z 256
uchar voxelCache[VC_X][VC_Y][VC_Z];
uchar voxelLightingCache[VC_X][VC_Y][VC_Z];

struct VoxelMesh {
	bool generated;
	bool upToDate;

	Vec2i coord;
	uchar* voxels;
	uchar* lighting;

	float transform[3][3];
	int quadCount;

	char* meshBuffer;
	int meshBufferSize;
	int meshBufferCapacity;
	uint bufferId;

	char* texBuffer;
	int texBufferSize;
	int texBufferCapacity;
	uint textureId;
	uint texBufferId;
};

void initVoxelMesh(VoxelMesh* m, Vec2i coord) {
	*m = {};
	m->coord = coord;

	m->meshBufferCapacity = kiloBytes(150);
	m->meshBuffer = (char*)getPMemory(m->meshBufferCapacity);

	m->texBufferCapacity = m->meshBufferCapacity/4;
	m->texBuffer = (char*)getPMemory(m->texBufferCapacity);

	m->voxels = (uchar*)getPMemory(VOXEL_SIZE);
	m->lighting = (uchar*)getPMemory(VOXEL_SIZE);

	glCreateBuffers(1, &m->bufferId);

	if(STBVOX_CONFIG_MODE == 1) {
		glCreateBuffers(1, &m->texBufferId);
		glCreateTextures(GL_TEXTURE_BUFFER, 1, &m->textureId);
	}
}

VoxelMesh* getVoxelMesh(VoxelMesh* vms, int* vmsSize, Vec2i coord) {

	// find mesh at coordinate
	int index = -1;
	for(int i = 0; i < *vmsSize; i++) {
		VoxelMesh* vm = vms + i;
		if(vm->coord == coord) {
			index = i;
			break;
		}
	}

	// initialise mesh with coordinate
	if(index == -1) {
		index = (*vmsSize)++;
		VoxelMesh* m = vms + index;

		initVoxelMesh(m, coord);
	}

	VoxelMesh* m = vms + index;
	return m;
}

void generateVoxelMesh(VoxelMesh* m) {
	Vec2i coord = m->coord;

	if(!m->generated) {
		zeroMemory(m->voxels, VOXEL_SIZE);
		memSet(m->lighting, 255, VOXEL_SIZE);

		Vec3i min = vec3i(0,0,0);
		Vec3i max = vec3i(VOXEL_X,VOXEL_Y,VOXEL_Z);

	    for(int y = min.y; y < max.y; y++) {
	    	for(int x = min.x; x < max.x; x++) {
	    		int gx = (coord.x*VOXEL_X)+x;
	    		int gy = (coord.y*VOXEL_Y)+y;

	    		float perlin = perlin2d(gx+4000, gy+4000, 0.015f, 4);
	    		float height = perlin*50;
	    		// float height = 5;

	    		int blockType = randomInt(8,10);
	    		for(int z = 0; z < height; z++) {
	    			m->voxels[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = blockType;
	    			m->lighting[x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z] = 0;
	    		}
	    	}
	    }

		m->generated = true;
	}
}


void makeMesh(VoxelMesh* m, VoxelMesh* vms, int* vmsSize) {
	zeroMemory(voxelCache, VC_X*VC_Y*VC_Z);
	zeroMemory(voxelLightingCache, VC_X*VC_Y*VC_Z);

	// gather voxel data in radius and copy to cache
	Vec2i coord = m->coord;
	for(int y = -1; y < 2; y++) {
		for(int x = -1; x < 2; x++) {
			Vec2i c = coord + vec2i(x,y);
			VoxelMesh* lm = getVoxelMesh(vms, vmsSize, c);
			generateVoxelMesh(lm);

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
						voxelCache[x+lPos.x][y+lPos.y][z+1] = lm->voxels[(x+mPos.x)*VOXEL_Y*VOXEL_Z + (y+mPos.y)*VOXEL_Z + z];
						voxelLightingCache[x+lPos.x][y+lPos.y][z+1] = lm->lighting[(x+mPos.x)*VOXEL_Y*VOXEL_Z + (y+mPos.y)*VOXEL_Z + z];
					}
				}
			}

			// make floor solid
			for(int y = 0; y < VC_Y; y++) {
				for(int x = 0; x < VC_X; x++) {
					voxelCache[x][y][0] = 1;
				}
			}
		}
	}


	stbvox_mesh_maker mm;
	stbvox_init_mesh_maker(&mm);
	stbvox_input_description* inputDesc = stbvox_get_input_description(&mm);
	*inputDesc = {};

	stbvox_set_buffer(&mm, 0, 0, m->meshBuffer, m->meshBufferCapacity);

	if(STBVOX_CONFIG_MODE == 1) {
		stbvox_set_buffer(&mm, 0, 1, m->texBuffer, m->texBufferCapacity);
	}

	int count = stbvox_get_buffer_count(&mm);

	unsigned char tex2[256];
	for(int i = 0; i < arrayCount(tex2)-1; i++) tex2[1+i] = i;

	inputDesc->block_tex2 = (unsigned char*)tex2;

	stbvox_set_input_stride(&mm, VC_Y*VC_Z,VC_Z);
	stbvox_set_input_range(&mm, 0,0,0, VOXEL_X, VOXEL_Y, VOXEL_Z);

	inputDesc->blocktype = &voxelCache[1][1][1];
	inputDesc->lighting = &voxelLightingCache[1][1][1];

	stbvox_set_default_mesh(&mm, 0);
	int success = stbvox_make_mesh(&mm);

	stbvox_set_mesh_coordinates(&mm, coord.x*VOXEL_X, coord.y*VOXEL_Y,0);

	stbvox_get_transform(&mm, m->transform);
	float bounds [2][3]; stbvox_get_bounds(&mm, bounds);
	m->quadCount = stbvox_get_quad_count(&mm, 0);

	int bufferSizePerQuad = stbvox_get_buffer_size_per_quad(&mm, 0);
	int textureBufferSizePerQuad = stbvox_get_buffer_size_per_quad(&mm, 1);



	glNamedBufferData(m->bufferId, bufferSizePerQuad*m->quadCount, m->meshBuffer, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, m->bufferId);

	if(STBVOX_CONFIG_MODE == 1) {
		glNamedBufferData(m->texBufferId, textureBufferSizePerQuad*m->quadCount, m->texBuffer, GL_STATIC_DRAW);
		glTextureBuffer(m->textureId, GL_RGBA8UI, m->texBufferId);
	}

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


#define voxelArray(x, y, z) x*VOXEL_Y*VOXEL_Z + y*VOXEL_Z + z


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

	Mesh mesh;
	PipelineIds pipelineIds;
	uint programs[16];
	uint textures[16];
	int texCount;
	uint samplers[16];

	uint frameBuffers[16];
	uint renderBuffers[16];
	uint frameBufferTextures[2];

	WindowSettings wSettings;
	Vec3 camera;
	Vec3 camPos;
	Vec3 camLook;
	Vec2 camRot;
	float aspectRatio;
	float fieldOfView;

	Font fontArial;

	Vec2i curRes;
	int msaaSamples;
	Vec2i fboRes;
	bool useNativeRes;

	Vec3 playerPos;
	Vec3 playerLook;
	Vec2 playerRot;
	Vec3 playerSize;
	float playerCamZOffset;
	bool playerMode;
	bool pickMode;

	Vec3 playerVel;
	Vec3 playerAcc;



	float transform[3][3];
	int quadCount;
	uint shader;
	uint voxelVertex;
	uint voxelFragment;
	uint bufferId;
	char* meshBuffer;
	int meshBufferSize;
	GLuint voxelSamplers[3];
	GLuint voxelTextures[3];

	unsigned char* voxelBlocks;

	GLuint voxelFaceTextures;
	GLuint voxelFaceSamplers;

	uint texBufferId;
	char* texBuffer;
	int texBufferSize;

	GLuint textureUnits[16];
	GLuint samplerUnits[16];

	VoxelMesh* vMeshs;
	int vMeshsSize;
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
		initSystem(systemData, windowsData, 0, 0,0,0,0);

		ad->msaaSamples = 4;
		ad->fieldOfView = 50;
		ad->fboRes = vec2i(0, 120);
		ad->useNativeRes = true;

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
		ids->cubeVertexModel = glGetUniformLocation(ids->cubeVertex, "model");
		ids->cubeVertexView = glGetUniformLocation(ids->cubeVertex, "view");
		ids->cubeVertexProj = glGetUniformLocation(ids->cubeVertex, "proj");
		ids->cubeVertexColor = glGetUniformLocation(ids->cubeVertex, "setColor");
		ad->programs[1] = ids->programCube;

		ids->programPrimitive = createShader(vertexShaderPrimitive, fragmentShaderPrimitive, &ids->primitiveVertex, &ids->primitiveFragment);
		ids->primitiveVertexView = glGetUniformLocation(ids->primitiveVertex, "view");
		ids->primitiveVertexProj = glGetUniformLocation(ids->primitiveVertex, "proj");
		ids->primitiveVertexColor = glGetUniformLocation(ids->primitiveVertex, "setColor");
		ids->primitiveVertexVertices = glGetUniformLocation(ids->primitiveVertex, "vertices");
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





		GLenum glError = glGetError(); printf("GLError: %i\n", glError);

		ad->shader = createShader(stbvox_get_vertex_shader(), stbvox_get_fragment_shader(), &ad->voxelVertex, &ad->voxelFragment);

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

		texId = ad->voxelTextures[0];
		glTextureStorage3D(texId, 6, internalFormat, width, height, texCount);
		for(int tc = 0; tc < texCount; tc++) {
			unsigned char* stbData;
			int x,y,n;

			strClear(fullPath);
			strAppend(fullPath, p);
			strAppend(fullPath, files[tc]);
			stbData = stbi_load(fullPath, &x, &y, &n, 4);

			if(x == width && y == height) {
				glTextureSubImage3D(texId, 0, 0, 0, tc, x, y, 1, format, GL_UNSIGNED_BYTE, stbData);
				glGenerateTextureMipmap(texId);
			}

			stbi_image_free(stbData);
		}



		glCreateFramebuffers(2, ad->frameBuffers);
		glCreateRenderbuffers(2, ad->renderBuffers);
		glCreateTextures(GL_TEXTURE_2D, 2, ad->frameBufferTextures);
		// GLenum result = glCheckNamedFramebufferStatus(ad->frameBuffers[0], GL_FRAMEBUFFER);



		int vMeshSize = sizeof(VoxelMesh)*2000;
		ad->vMeshs = (VoxelMesh*)getPMemory(vMeshSize);
		zeroMemory(ad->vMeshs, vMeshSize);
		ad->vMeshsSize = 0;

		ad->playerPos = vec3(35.5f,35.3f,30);
		ad->playerLook = vec3(0,1,0);
		ad->playerSize = vec3(0.8f, 0.8f, 1.8f);
		ad->playerCamZOffset = ad->playerSize.z*0.5f - ad->playerSize.x*0.5f;

		ad->playerMode = true;
		ad->pickMode = false;

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

	updateInput(&ad->input, isRunning, windowHandle);

	PipelineIds* ids = &ad->pipelineIds;

	if(input->keysPressed[VK_F1]) {
		int mode;
		if(wSettings->fullscreen) mode = WINDOW_MODE_WINDOWED;
		else mode = WINDOW_MODE_FULLBORDERLESS;
		setWindowMode(windowHandle, wSettings, mode);

		updateAfterWindowResizing(wSettings, &ad->aspectRatio, ad->frameBuffers[0], ad->frameBuffers[1], ad->renderBuffers[0], ad->renderBuffers[1], &ad->frameBufferTextures[0], ad->msaaSamples, &ad->fboRes, &ad->curRes, ad->useNativeRes);
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
		ad->playerMode = !ad->playerMode;
	}

	if(input->keysPressed[VK_F5]) {
		ad->pickMode = !ad->pickMode;
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
	// Vec3 playerDelta = vec3(0,0,0);

	// ad->camPos = ad->playerPos + vec3(0,0,ad->playerCamZOffset);
	// ad->camLook = ad->playerLook;

	Vec3 cLook, cRight, cUp;
	getCamData(ad->camLook, ad->camRot, gUp, &cLook, &cRight, &cUp);

	// cLook = pLook;
	// cRight = pRight;
	// cUp = pUp;

	ad->playerAcc = vec3(0,0,0);

	if(!ad->playerMode) {
		// 3d camera controls
		if(input->mouseButtonDown[1]) {
			float dt = 0.005f;
			ad->camRot.y += dt * input->mouseDeltaY;
			ad->camRot.x += dt * input->mouseDeltaX;

			float margin = 0.00001f;
			clamp(&ad->camRot.y, -M_PI+margin, M_PI-margin);
		}

		if( input->keysDown[VK_W] || input->keysDown[VK_A] || input->keysDown[VK_S] || 
			input->keysDown[VK_D] || input->keysDown[VK_E] || input->keysDown[VK_Q]) {

			Vec3 look = cLook;
			if(input->keysDown[VK_CONTROL]) look = cross(gUp, cRight);

			float speed = 0.1f;
			if(input->keysDown[VK_SHIFT]) speed = 0.5f;
			if(input->keysDown[VK_W]) ad->camPos += normVec3(look)*speed;
			if(input->keysDown[VK_S]) ad->camPos += -normVec3(look)*speed;
			if(input->keysDown[VK_D]) ad->camPos += normVec3(cRight)*speed;
			if(input->keysDown[VK_A]) ad->camPos += -normVec3(cRight)*speed;
			if(input->keysDown[VK_E]) ad->camPos += normVec3(gUp)*speed;
			if(input->keysDown[VK_Q]) ad->camPos += -normVec3(gUp)*speed;
		}
	} else {
		ad->camPos = ad->playerPos + vec3(0,0,ad->playerCamZOffset);
		ad->camLook = ad->playerLook;

		cLook = pLook;
		cRight = pRight;
		cUp = pUp;

		if(input->mouseButtonDown[1]) {
			float dt = 0.005f;
			ad->playerRot.y += dt * input->mouseDeltaY;
			ad->playerRot.x += dt * input->mouseDeltaX;

			float margin = 0.00001f;
			clamp(&ad->playerRot.y, -M_PI+margin, M_PI-margin);
		}

		if( input->keysDown[VK_W] || input->keysDown[VK_A] || input->keysDown[VK_S] || 
			input->keysDown[VK_D] || input->keysDown[VK_E] || input->keysDown[VK_Q]) {

			Vec3 look = pLook;
			if(input->keysDown[VK_CONTROL]) look = cross(gUp, pRight);

			// float speed = 0.01f;
			float speed = 1.0f;
			if(input->keysDown[VK_SHIFT]) speed = 3.0f;
			if(input->keysDown[VK_W]) ad->playerAcc += normVec3(look)*speed;
			if(input->keysDown[VK_S]) ad->playerAcc += -normVec3(look)*speed;
			if(input->keysDown[VK_D]) ad->playerAcc += normVec3(pRight)*speed;
			if(input->keysDown[VK_A]) ad->playerAcc += -normVec3(pRight)*speed;
			if(input->keysDown[VK_E]) ad->playerAcc += normVec3(gUp)*speed;
			if(input->keysDown[VK_Q]) ad->playerAcc += -normVec3(gUp)*speed;
		}
	}

	float dt = 0.166f;
	ad->playerVel += ad->playerAcc*dt;
	ad->playerVel *= 0.9f;

	if(ad->playerVel != vec3(0,0,0)) {
		Vec3 pPos = ad->playerPos;
		Vec3 pSize = ad->playerSize;

		// Vec3 nPos = pPos + playerDelta;
		// Vec3 nPos = pPos + playerDelta;
		Vec3 nPos = pPos + 0.5f*ad->playerAcc*dt*dt + ad->playerVel*dt;

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
						Vec2i meshCoord = getMeshCoordFromVoxelCoord(coord);
						VoxelMesh* m = getVoxelMesh(ad->vMeshs, &ad->vMeshsSize, meshCoord);
						Vec3i localC = getLocalVoxelCoord(coord);
						int blockType = m->voxels[voxelArray(localC.x,localC.y,localC.z)];

						if(blockType > 0) {
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
					}
				}

				float error = 0.00001f;
				nPos += dir*(-minDistance + error);
			}
		}

		ad->playerPos = nPos;
	}


	glViewport(0,0, ad->curRes.x, ad->curRes.y);
	// glClearColor(0.3f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// glDepthRange(-1.0,1.0);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CW);



	glEnable(GL_MULTISAMPLE);
	glBindFramebuffer(GL_FRAMEBUFFER, ad->frameBuffers[0]);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);



#if 1



	glEnable(GL_CULL_FACE);
	// glDisable(GL_CULL_FACE);
	// glDisable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearDepth(1);
	glDepthMask(GL_TRUE);
	glDisable(GL_SCISSOR_TEST);
	glClearColor(0.6f,0.7f,0.9f,0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glFrontFace(GL_CW);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.5);






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
	lColor *= 50;
	Vec3 light[2] = { vec3(0,0,(amp/2)+start + sin(dtl)*amp), lColor };
	int loc = glGetUniformLocation(ad->voxelFragment, "light_source");
	glProgramUniform3fv(ad->voxelFragment, loc, 2, (GLfloat*)light);
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
	al.e2[3][0] = 0.6f, al.e2[3][1] = 0.7f, al.e2[3][2] = 0.9f;
	// al.e2[3][3] = 1.0f / (view_distance - MESH_CHUNK_SIZE_X);
	// al.e2[3][3] *= al.e2[3][3];
	al.e2[3][3] = (float)1.0f/(VIEW_DISTANCE - VOXEL_X);
	al.e2[3][3] *= al.e2[3][3];

	ambientLighting = al;



	Mat4 view, proj; 
	viewMatrix(&view, ad->camPos, -cLook, cUp, cRight);
	projMatrix(&proj, degreeToRadian(ad->fieldOfView), ad->aspectRatio, 0.1f, 2000);

	setupVoxelUniforms(ad->voxelVertex, ad->voxelFragment, vec4(ad->camPos, 1), 0, 0, 2, ambientLighting, view, proj);





	int meshGenerationCount = 0;
	int triangleCount = 0;

	Vec2i playerMeshCoord = getMeshCoordFromGlobalCoord(ad->camPos);
	int radius = VIEW_DISTANCE/VOXEL_X;

	VoxelMesh* vms = ad->vMeshs;
	int* vmsSize = &ad->vMeshsSize;

	// Vec2i min = vec2i(playerMeshCoord.x-radius, playerMeshCoord.y-radius);
	// Vec2i max = vec2i(playerMeshCoord.x+radius, playerMeshCoord.y+radius);
	// for(int x = min.x; x < max.x+1; x++) {
		// for(int y = min.y; y < max.y+1; y++) {

	Vec2i pPos = vec2i(playerMeshCoord.x, playerMeshCoord.y);

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

			Vec2i coord = lPos;
			// Vec2i coord = vec2i(x,y);

			VoxelMesh* m = getVoxelMesh(vms, vmsSize, coord);
			if(!m->upToDate) {
				if(meshGenerationCount < 1) {
					makeMesh(m, vms, vmsSize);
					m->upToDate = true;

					meshGenerationCount++;
				}
			}



			Vec3 cp = ad->camPos;
			Vec3 cl = cLook;
			Vec3 cu = cUp;
			Vec3 cr = cRight;

			float ar = ad->aspectRatio;
			float fov = degreeToRadian(ad->fieldOfView);
			float ne = 0.1f;
			float fa = 2000;

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
			// if(true) {
				if(STBVOX_CONFIG_MODE == 0) {
					// interleaved buffer - 2 uints in a row -> 8 bytes stride
					glBindBuffer(GL_ARRAY_BUFFER, m->bufferId);
					int vaLoc = glGetAttribLocation(ad->voxelVertex, "attr_vertex");
					glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 8, (void*)0);
					glEnableVertexAttribArray(vaLoc);
					int fLoc = glGetAttribLocation(ad->voxelVertex, "attr_face");
					glVertexAttribIPointer(fLoc, 4, GL_UNSIGNED_BYTE, 8, (void*)4);
					glEnableVertexAttribArray(fLoc);

				} else {
					glBindBuffer(GL_ARRAY_BUFFER, m->bufferId);
					int vaLoc = glGetAttribLocation(ad->voxelVertex, "attr_vertex");
					glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 4, (void*)0);
					glEnableVertexAttribArray(vaLoc);
				}

				GLuint transformUniform1 = glGetUniformLocation(ad->voxelVertex, "transform");
				glProgramUniform3fv(ad->voxelVertex, transformUniform1, 3, m->transform[0]);
				GLuint transformUniform2 = glGetUniformLocation(ad->voxelFragment, "transform");
				glProgramUniform3fv(ad->voxelFragment, transformUniform2, 3, m->transform[0]);

				ad->textureUnits[0] = ad->voxelTextures[0];
				ad->textureUnits[1] = ad->voxelTextures[1];
				ad->textureUnits[2] = m->textureId;
				ad->samplerUnits[0] = ad->voxelSamplers[0];
				ad->samplerUnits[1] = ad->voxelSamplers[1];
				ad->samplerUnits[2] = ad->voxelSamplers[2];

				glBindTextures(0,16,ad->textureUnits);
				glBindSamplers(0,16,ad->samplerUnits);

				glBindProgramPipeline(ad->shader);

				glDrawArrays(GL_QUADS, 0, m->quadCount*4);
				triangleCount += m->quadCount*4;
			}
		}
	}

	if(second) {
		GLenum glError = glGetError(); printf("GLError: %i\n", glError);
	}

	// // glDisableVertexAttrribArray(0);
	// glBindBuffer(GL_ARRAY_BUFFER, 0);
	// glActiveTexture(GL_TEXTURE0);

	// glUseProgram(0);

	// // glDisable(GL_BLEND);
	// // glDisable(GL_CULL_FACE);
	// // glDisable(GL_DEPTH_TEST);

	// glDisable(GL_TEXTURE_2D);





#endif


	if(second) {
		GLenum glError = glGetError(); printf("GLError: %i\n", glError);
		int stop = 2;
	}



	lookAt(&ad->pipelineIds, ad->camPos, -cLook, cUp, cRight);
	perspective(&ad->pipelineIds, degreeToRadian(ad->fieldOfView), ad->aspectRatio, 0.1f, 2000);
	glBindProgramPipeline(ad->pipelineIds.programCube);

	// static float dt = 0;
	// dt += 0.01f;
	// drawCube(&ad->pipelineIds, vec3(5,5,-5), vec3(6,2,1), dt, normVec3(vec3(0.9f,0.6f,0.2f)));

	Vec3 off = vec3(0.5f, 0.5f, 0.5f);
	Vec3 s = vec3(1.01f, 1.01f, 1.01f);

	// for(int i = -10; i < 10; i++) drawCube(&ad->pipelineIds, vec3(i*10,0,0) + off, s, 0, normVec3(vec3(0.9f,0.6f,0.2f)));
	// for(int i = -10; i < 10; i++) drawCube(&ad->pipelineIds, vec3(0,i*10,0) + off, s, 0, normVec3(vec3(0.9f,0.6f,0.2f)));
	// for(int i = -10; i < 10; i++) drawCube(&ad->pipelineIds, vec3(0,0,i*10) + off, s, 0, normVec3(vec3(0.9f,0.6f,0.2f)));


	// for(int i = 0; i < 10; i++) drawCube(&ad->pipelineIds, vec3(i*10,0,0) + off, s, vec4(0,1,1,1), 0, vec3(0,0,0));
	// for(int i = 0; i < 10; i++) drawCube(&ad->pipelineIds, vec3(0,i*10,0) + off, s, vec4(0,1,1,1), 0, vec3(0,0,0));
	// for(int i = 0; i < 10; i++) drawCube(&ad->pipelineIds, vec3(0,0,i*10) + off, s, vec4(0,1,1,1), 0, vec3(0,0,0));


	#ifdef STBVOX_CONFIG_LIGHTING_SIMPLE
	drawCube(&ad->pipelineIds, light[0], vec3(3,3,3), vec4(1,1,1,1), 0, vec3(0,0,0));
	#endif


	glDisable(GL_CULL_FACE);

	glEnable(GL_CULL_FACE);



	Vec3 cursorPos;
	Vec3 cursorDir;

	// if(input->mouseButtonPressed[0] && ad->playerMode) {
	if(true) {
		// playerPos;
		// playerLook;
		// playercamoffset;

		cursorDir = pLook;
		cursorPos = ad->playerPos + vec3(0,0,ad->playerCamZOffset);

		// Vec3 globalCoord = cursorPos + cursorDir*3;

		// VoxelMesh* vm = getVoxelMesh(ad->vMeshs, &ad->vMeshsSize, getMeshCoordFromGlobalCoord(globalCoord));
		// Vec3i localCoord = getLocalVoxelCoord(getVoxelCoordFromGlobalCoord(globalCoord));
		// int blockType = vm->voxels[voxelArray(localCoord.x, localCoord.y, localCoord.z)];

		// Vec3 blockCoords = getGlobalCoordFromVoxelCoord(getVoxelCoordFromGlobalCoord(globalCoord));

		// drawCube(&ad->pipelineIds, blockCoords, vec3(1,1,1), vec4(0,0,1,1), 0, vec3(0,0,0));



		// Vec3 blockCoords;
		// int blockType;

		// if(blockType > 0) {
			// drawCube(&ad->pipelineIds, blockCoords, vec3(1.01f,1.01f,1.01f), vec4(0,0,1,1), 0, vec3(0,0,0));
		// }

		// get block in line
		Vec3 startPos = cursorPos;
		Vec3 startDir = cursorDir;

		Vec3 newPos = startPos;
		int biggestAxis = maxReturnIndex(abs(startDir.x), abs(startDir.y), abs(startDir.z));

		bool intersection = false;
		Vec3 intersectionBox;

		int length = 20;
		for(int i = 0; i < length; i++) {
			newPos = newPos + normVec3(startDir);

			Vec3 coords[5];
			int coordsSize = 0;

			coords[coordsSize++] = newPos;

			Vec3 blockCoords = getGlobalCoordFromVoxelCoord(getVoxelCoordFromGlobalCoord(newPos));
			for(int axis = 0; axis < 3; axis++) {
				if(axis == biggestAxis) continue;
				
				for(int i = 0; i < 2; i++) {
					Vec3 dir = vec3(0,0,0);
			
					if(i == 0) dir.e[axis] = 1;
					else dir.e[axis] = -1;

					coords[coordsSize++] = blockCoords + dir;

					// drawCube(&ad->pipelineIds, blockCoords + dir, vec3(1.0f,1.0f,1.0f), vec4(1,0,1,1), 0, vec3(0,0,0));
				}
			}

			for(int i = 0; i < coordsSize; i++) {
				Vec3 block = coords[i];

				VoxelMesh* vm = getVoxelMesh(ad->vMeshs, &ad->vMeshsSize, getMeshCoordFromGlobalCoord(block));
				Vec3i localCoord = getLocalVoxelCoord(getVoxelCoordFromGlobalCoord(block));
				int blockType = vm->voxels[voxelArray(localCoord.x, localCoord.y, localCoord.z)];

				if(blockType > 0) {
					intersectionBox = getGlobalCoordFromVoxelCoord(getVoxelCoordFromGlobalCoord(block));
					intersection = boxRaycast(startPos, startDir, rect3CenDim(intersectionBox, vec3(1,1,1)));

					if(intersection) {
						glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
						drawCube(&ad->pipelineIds, intersectionBox, vec3(1.01f,1.01f,1.01f), vec4(0.9f), 0, vec3(0,0,0));

						intersection = true;
						break;						
					}
				}
			}

			if(intersection) break;
		}


		if(input->mouseButtonPressed[0] && ad->playerMode) {
			Vec3 block = intersectionBox;

			VoxelMesh* vm = getVoxelMesh(ad->vMeshs, &ad->vMeshsSize, getMeshCoordFromGlobalCoord(block));
			Vec3i localCoord = getLocalVoxelCoord(getVoxelCoordFromGlobalCoord(block));

			if(ad->pickMode) {
				vm->voxels[voxelArray(localCoord.x, localCoord.y, localCoord.z)] = 2;
				// vm->lighting[voxelArray(localCoord.x, localCoord.y, localCoord.z)] = 255;
			} else {
				vm->voxels[voxelArray(localCoord.x, localCoord.y, localCoord.z)] = 0;			
				vm->lighting[voxelArray(localCoord.x, localCoord.y, localCoord.z)] = 255;			
			}

			generateVoxelMesh(vm);
			makeMesh(vm, ad->vMeshs, &ad->vMeshsSize);
		}
	}






	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	drawCube(&ad->pipelineIds, ad->playerPos, ad->playerSize, vec4(1,1,1,1), 0, vec3(0,0,0));



	// glEnable(GL_CULL_FACE);

	view;
	viewMatrix(&view, ad->camPos, -cLook, cUp, cRight);
	glProgramUniformMatrix4fv(ids->primitiveVertex, ids->primitiveVertexView, 1, 1, view.e);
	proj;
	projMatrix(&proj, degreeToRadian(ad->fieldOfView), ad->aspectRatio, 0.1f, 2000);
	glProgramUniformMatrix4fv(ids->primitiveVertex, ids->primitiveVertexProj, 1, 1, proj.e);
	glBindProgramPipeline(ad->pipelineIds.programPrimitive);

	glDisable(GL_CULL_FACE);
	glEnable(GL_LINE_SMOOTH);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glLineWidth(3);

	// ad->camPos = vec3(44.81f, 34.44f, 29.27f);
	// ad->camLook = vec3(0.90f,0, -0.45f);

	Vec3 pCam = ad->playerPos + vec3(0,0,ad->playerCamZOffset);
	Vec3 pCamDir = pLook;
	Vec3 pCamRight = pRight;
	Vec3 pCamUp = pUp;

	// Vec3 verts[4] = {};
	// verts[0] = pCam; verts[1] = pCam + pCamDir*1;
	// glProgramUniform3fv(ids->primitiveVertex, ids->primitiveVertexVertices, 4, verts[0].e);
	// glProgramUniform4f(ids->primitiveVertex, ids->primitiveVertexColor, 1,0,0,1);
	// glDrawArrays(GL_LINES, 0, 2);

	// verts[0] = pCam; verts[1] = pCam + pCamRight*1;
	// glProgramUniform3fv(ids->primitiveVertex, ids->primitiveVertexVertices, 4, verts[0].e);
	// glProgramUniform4f(ids->primitiveVertex, ids->primitiveVertexColor, 0,1,0,1);
	// glDrawArrays(GL_LINES, 0, 2);

	// verts[0] = pCam; verts[1] = pCam + pCamUp*1;
	// glProgramUniform3fv(ids->primitiveVertex, ids->primitiveVertexVertices, 4, verts[0].e);
	// glProgramUniform4f(ids->primitiveVertex, ids->primitiveVertexColor, 0,0,1,1);
	// glDrawArrays(GL_LINES, 0, 2);


	Vec3 verts[4] = {};
	verts[0] = cursorPos; verts[1] = cursorPos + cursorDir*30;
	glProgramUniform3fv(ids->primitiveVertex, ids->primitiveVertexVertices, 4, verts[0].e);
	glProgramUniform4f(ids->primitiveVertex, ids->primitiveVertexColor, 1,0,0,1);
	glDrawArrays(GL_LINES, 0, 2);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LINE_SMOOTH);





	glBindFramebuffer (GL_READ_FRAMEBUFFER, ad->frameBuffers[0]);
	// glReadBuffer      (GL_COLOR_ATTACHMENT0);
	glBindFramebuffer (GL_DRAW_FRAMEBUFFER, ad->frameBuffers[1]);
	// glDrawBuffer      (GL_BACK);

	glBlitFramebuffer (0,0, ad->curRes.x, ad->curRes.y,
	                   0,0, ad->curRes.x, ad->curRes.y,
	                   // GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
	                   GL_COLOR_BUFFER_BIT,
	                   // GL_NEAREST);
	                   GL_LINEAR);


	glBindFramebuffer (GL_FRAMEBUFFER, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);


	glViewport(0,0, wSettings->currentRes.x, wSettings->currentRes.y);

	ortho(&ad->pipelineIds, rect(0,0,1,1));
	glBindProgramPipeline(ad->pipelineIds.programQuad);

	glProgramUniform4f(ids->quadVertex, ids->quadVertexMod, 0.5f, 0.5f, 1, 1);
	glProgramUniform4f(ids->quadVertex, ids->quadVertexUV, 0, 1,0,1);
	glProgramUniform4f(ids->quadVertex, ids->quadVertexColor, 1,1,1,1);
	glBindTexture(GL_TEXTURE_2D, ad->frameBufferTextures[0]);
	glDrawArraysInstancedBaseInstance(GL_TRIANGLE_STRIP, 0, 4, 1, 0);

	glBindFramebuffer (GL_DRAW_FRAMEBUFFER, ad->frameBuffers[1]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer (GL_FRAMEBUFFER, 0);




	ortho(&ad->pipelineIds, rectCenDim(cam->x,cam->y, cam->z, cam->z/ad->aspectRatio));
	glBindProgramPipeline(ad->pipelineIds.programQuad);
	// drawRect(&ad->pipelineIds, rectCenDim(0, 0, 0.01f, 100), rect(0,0,1,1), vec4(0.4f,1,0.4f,1), ad->textures[0]);
	// drawRect(&ad->pipelineIds, rectCenDim(0, 0, 100, 0.01f), rect(0,0,1,1), vec4(0.4f,0.4f,1,1), ad->textures[0]);

	// drawRect(&ad->pipelineIds, rectCenDim(0, 0, 5, 5), rect(0,0,1,1), vec4(1,1,1,1), ad->textures[2]);
	// drawRect(&ad->pipelineIds, rectCenDim(0, 0, 5, 5), rect(0,0,1,1), vec4(1,1,1,1), 3);






	ortho(&ad->pipelineIds, rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0));
	glBindProgramPipeline(ad->pipelineIds.programQuad);
	drawTextA(&ad->pipelineIds, vec2(0,-30),  vec4(1,1,1,1), &ad->fontArial, 0, 2, "Pos  : (%f,%f,%f)", ad->camPos.x, ad->camPos.y, ad->camPos.z);
	drawTextA(&ad->pipelineIds, vec2(0,-60),  vec4(1,1,1,1), &ad->fontArial, 0, 2, "Look : (%f,%f,%f)", cLook.x, cLook.y, cLook.z);
	drawTextA(&ad->pipelineIds, vec2(0,-90),  vec4(1,1,1,1), &ad->fontArial, 0, 2, "Up   : (%f,%f,%f)", cUp.x, cUp.y, cUp.z);
	drawTextA(&ad->pipelineIds, vec2(0,-120), vec4(1,1,1,1), &ad->fontArial, 0, 2, "Right: (%f,%f,%f)", cRight.x, cRight.y, cRight.z);
	drawTextA(&ad->pipelineIds, vec2(0,-150), vec4(1,1,1,1), &ad->fontArial, 0, 2, "Rot  : (%f,%f)", ad->camRot.x, ad->camRot.y);

	drawTextA(&ad->pipelineIds, vec2(0,-180), vec4(1,1,1,1), &ad->fontArial, 0, 2, "Poly Count  : (%f)", (float)triangleCount);





	swapBuffers(&ad->systemData);
	glFinish();

	clearTMemory();
}
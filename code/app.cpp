#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <gl\gl.h>
// #include <gl\glext.h>
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
#define STBVOX_CONFIG_MODE 0
#include "stb_voxel_render.h"



//-----------------------------------------
//				WHAT TO DO
//-----------------------------------------
/*
- Joysticks, Keyboard, Mouse, Xinput-DirectInput
- stb_voxel
- Sound
- Data Package - Streaming
- Expand Font functionality
- Gui
- Framebuffer Render to resolution

- create simpler windows.h
- remove c runtime library, implement sin, cos...
- savestates, hotrealoading
- pre and post functions to appMain
- memory expansion
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
		GLOP(void, BindTextureUnit, GLuint unit, GLuint texture)





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
	    vec3(-0.5f,-0.5f, 0.5f),
	    vec3(-0.5f, 0.5f, 0.5f),
	    vec3(0.5f, 0.5f,-0.5f),
	    vec3(-0.5f,-0.5f,-0.5f),
	    vec3(-0.5f, 0.5f,-0.5f), 
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

	out gl_PerVertex { vec4 gl_Position; };
	out vec4 Color;

	uniform mat4x4 model;
	uniform mat4x4 view;
	uniform mat4x4 proj;

	uniform mat4x4 projViewModel;

	void main() {
		float c = gl_VertexID;
		// Color = vec4(1/(c/36),1/(c/7),1/(c/2),1);
		Color = vec4(c/36,c/36,c/36,1);
	
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
	// uniform sampler2DArray s[2];
	// uniform sampler2D s;

	smooth in vec2 uv;
	in vec4 Color;

	layout(depth_less) out float gl_FragDepth;
	out vec4 color;

	void main() {
		color = texture(s, uv) * Color;


         // "   vec4 tex2 = texture(tex_array[1], vec3(texcoord_2, float(tex2_id)));\n"

		// color = texture(s[0], vec3(uv, 2)) * Color;
		// color = texture(s[0], uv) * Color;
		// color = texture(s, uv) * Color;
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
	uint cubeVertexModel;
	uint cubeVertexView;
	uint cubeVertexProj;

	uint programVoxel;
	uint voxelVertex;
	uint voxelFragment;
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
	// GLenum glError = glGetError(); printf("GLError: %i\n", glError);
	glUseProgramStages(shaderId, GL_FRAGMENT_SHADER_BIT, *fId);
	// glError = glGetError(); printf("GLError: %i\n", glError);

	return shaderId;
}

void ortho(PipelineIds* ids, Rect r) {
	r = rectGetCenDim(r);
	glProgramUniform4f(1, ids->quadVertexCamera, r.cen.x, r.cen.y, r.dim.w, r.dim.h);
}

void lookAt(PipelineIds* ids, Vec3 pos, Vec3 look, Vec3 up) {
	Mat4 view;
	viewMatrix(&view, pos, look, up);

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
		drawRect(*ids, rect(q.x0, q.y0, q.x1, q.y1), rect(q.s0,q.t0,q.s1,q.t1), color, 3);
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
		drawRect(*ids, rect(q.x0, q.y0, q.x1, q.y1), rect(q.s0,q.t0,q.s1,q.t1), color, 3);
	}
}

void drawCube(PipelineIds* ids, Vec3 trans, Vec3 scale, float degrees, Vec3 rot) {
	// Quat xRot = quat(dt, normVec3(vec3(1,0.7f,0.5f)));
	// Quat yRot = quat(0, vec3(0,1,0));
	// Quat zRot = quat(0, vec3(0,0,1));
	// quatRotationMatrix(&rm, zRot*yRot*xRot);

	Mat4 sm; scaleMatrix(&sm, scale);
	Mat4 rm; quatRotationMatrix(&rm, quat(degrees, rot));
	Mat4 tm; translationMatrix(&tm, trans);
	Mat4 model = tm*rm*sm;

	glProgramUniformMatrix4fv(ids->cubeVertex, ids->cubeVertexModel, 1, 1, model.e);
	glDrawArraysInstancedBaseInstance(GL_TRIANGLES, 0, 36, 1, 0);
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
	// Vec3 camUp;
	// Vec3 camRight;
	Vec2 camRot;
	float aspectRatio;

	Font fontArial;



	float transform[3][3];
	int quadCount;
	uint shader;
	uint voxelVertex;
	uint voxelFragment;
	uint bufferId;
	char* meshBuffer;
	GLuint voxelSamplers[2];	
	GLuint voxelTextures[2];
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

		// ids->cubeVertexScale = glGetUniformLocation(ids->cubeVertex, "scale");
		// ids->cubeVertexRot = glGetUniformLocation(ids->cubeVertex, "rot");
		// ids->cubeVertexTrans = glGetUniformLocation(ids->cubeVertex, "trans");
		ids->cubeVertexModel = glGetUniformLocation(ids->cubeVertex, "model");
		ids->cubeVertexView = glGetUniformLocation(ids->cubeVertex, "view");
		ids->cubeVertexProj = glGetUniformLocation(ids->cubeVertex, "proj");

		ad->programs[1] = ids->programCube;

		ad->camera = vec3(0,0,10);
		ad->camPos = vec3(0,4,10);
		ad->camLook = vec3(-1,0,0);
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






		stbvox_mesh_maker mm;
		stbvox_init_mesh_maker(&mm);
		stbvox_input_description* inputDesc = stbvox_get_input_description(&mm);
		*inputDesc = {};

		int bCount = stbvox_get_buffer_count(&mm);


		ad->meshBuffer = (char*)getPMemory(megaBytes(1));
		stbvox_set_buffer(&mm, 0, 0, ad->meshBuffer, megaBytes(1));
		int count = stbvox_get_buffer_count(&mm);
		int perQuad = stbvox_get_buffer_size_per_quad(&mm, 0);


		glCreateSamplers(2, ad->voxelSamplers);
		// glSamplerParameteri(ad->samplers[0], GL_TEXTURE_WRAP_S, GL_REPEAT);
		// glSamplerParameteri(ad->samplers[0], GL_TEXTURE_WRAP_T, GL_REPEAT);
		// glSamplerParameteri(ad->samplers[0], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// glSamplerParameteri(ad->samplers[0], GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		// glSamplerParameteri(ad->samplers[1], GL_TEXTURE_WRAP_S, GL_REPEAT);
		// glSamplerParameteri(ad->samplers[1], GL_TEXTURE_WRAP_T, GL_REPEAT);
		// glSamplerParameteri(ad->samplers[1], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// glSamplerParameteri(ad->samplers[1], GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);



		glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, ad->voxelTextures);

		uint format = GL_RGB;
		uint internalFormat = GL_RGB8;
		int texCount = 3;
		int x = 32;
		int y = 32;

		texId = ad->voxelTextures[0];
		glTextureStorage3D(texId, 1, internalFormat, x, y, texCount);
		for(int tc = 0; tc < texCount; tc++) {
			unsigned char* stbData;
			int x,y,n;
			if(tc == 0) stbData = stbi_load("..\\data\\minecraft textures\\stone_slab_top.png", &x, &y, &n, 0);
			if(tc == 1) stbData = stbi_load("..\\data\\minecraft textures\\anvil_base.png", &x, &y, &n, 0);
			if(tc == 2) stbData = stbi_load("..\\data\\minecraft textures\\brick_slab_side3.png", &x, &y, &n, 0);
			
			glTextureSubImage3D(texId, 0, 0, 0, tc, x, y, 1, format, GL_UNSIGNED_BYTE, stbData);

			stbi_image_free(stbData);
		}

		glTextureParameteri(texId, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
		glTextureParameteri(texId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTextureParameteri(texId, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTextureParameteri(texId, GL_TEXTURE_WRAP_T, GL_REPEAT);



		texId = ad->voxelTextures[1];
		glTextureStorage3D(texId, 1, internalFormat, x, y, texCount);
		for(int tc = 0; tc < texCount; tc++) {
			unsigned char* stbData;
			int x,y,n;
			if(tc == 0) stbData = stbi_load("..\\data\\minecraft textures\\beacon_top.png", &x, &y, &n, 0);
			if(tc == 1) stbData = stbi_load("..\\data\\minecraft textures\\bed_feet_bottom.png", &x, &y, &n, 0);
			if(tc == 2) stbData = stbi_load("..\\data\\minecraft textures\\cobblestone.png", &x, &y, &n, 0);

			glTextureSubImage3D(texId, 0, 0, 0, tc, x, y, 1, format, GL_UNSIGNED_BYTE, stbData);

			stbi_image_free(stbData);
		}

		glTextureParameteri(texId, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
		glTextureParameteri(texId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTextureParameteri(texId, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTextureParameteri(texId, GL_TEXTURE_WRAP_T, GL_REPEAT);





		// unsigned char tex1[] = {0,0,1,2};
		// inputDesc->block_tex1 = (unsigned char*)tex1;
		unsigned char tex2[] = {0,0,1,2};
		inputDesc->block_tex2 = (unsigned char*)tex2;



		const int bx = 32+2;
		const int by = 128;
		const int bz = 32+2;
		int size = bz*by*bx;

		unsigned char* voxelBlocks = (unsigned char*)getPMemory(size);
		zeroMemory(voxelBlocks, size);
		// unsigned char* voxelLights = (unsigned char*)getPMemory(size);
		// zeroMemory(voxelLights, size);

		inputDesc->blocktype = voxelBlocks + 1*by*bx + 1*bx + 1;
		// inputDesc->lighting = voxelLights + 1*by*bx + 1*bx + 1;

		stbvox_set_input_stride(&mm, bx*by,bx);
		stbvox_set_input_range(&mm, 1,1,1, bx-1,by-1,bz-1);

		for(int z = 2; z < bz-2; z++) {
			for(int y = 2; y < 3; y++) {
				for(int x = 2; x < bx-2; x++) {
					voxelBlocks[z*by*bx + y*bx + x] = 1;
				}
			}
		}

		for(int z = 2; z < 12; z++) {
			for(int x = 2; x < 12; x++) {
				voxelBlocks[z*by*bx + 3*bx + x] = 2;
			}
		}

		for(int z = 14; z < 17; z++) {
			for(int x = 14; x < 17; x++) {
				voxelBlocks[z*by*bx + 3*bx + x] = 3;
			}
		}

		stbvox_set_default_mesh(&mm, 0);

		int success = stbvox_make_mesh(&mm);

		stbvox_set_mesh_coordinates(&mm, 0,0,0);
		stbvox_get_transform(&mm, ad->transform);
		float bounds [2][3]; stbvox_get_bounds(&mm, bounds);
		ad->quadCount = stbvox_get_quad_count(&mm, 0);

		ad->shader = createShader(stbvox_get_vertex_shader(), stbvox_get_fragment_shader(), &ad->voxelVertex, &ad->voxelFragment);

		glCreateBuffers(1, &ad->bufferId);
		glNamedBufferData(ad->bufferId, ad->quadCount*4*sizeof(uint)*2, ad->meshBuffer, GL_STATIC_DRAW_ARB);
		glBindBuffer(GL_ARRAY_BUFFER, ad->bufferId);

		// interleaved buffer - 2 uints in a row -> 8 bytes stride
		int vaLoc = glGetAttribLocation(ad->voxelVertex, "attr_vertex");
		glVertexAttribIPointer(vaLoc, 1, GL_UNSIGNED_INT, 8, (void*)0);
		glEnableVertexAttribArray(vaLoc);
		int fLoc = glGetAttribLocation(ad->voxelVertex, "attr_face");
		glVertexAttribIPointer(fLoc, 4, GL_UNSIGNED_BYTE, 8, (void*)4);
		glEnableVertexAttribArray(fLoc);







		return; // window operations only work after first frame?
	}

	if(second) {
		setWindowProperties(windowHandle, wSettings->res.w, wSettings->res.h, -1920, 0);
		setWindowStyle(windowHandle, wSettings->style);
		setWindowMode(windowHandle, wSettings, WINDOW_MODE_FULLBORDERLESS);
	}

	if(reload) {
		loadFunctions();
	}



	updateInput(&ad->input, isRunning, windowHandle);
	getWindowProperties(windowHandle, &wSettings->currentRes.x, &wSettings->currentRes.y,0,0,0,0);
	ad->aspectRatio = wSettings->currentRes.x / (float)wSettings->currentRes.y;

	PipelineIds* ids = &ad->pipelineIds;

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

	if(input->mouseButtonDown[1]) {
		float dt = 0.005f;
		ad->camRot.y += dt * input->mouseDeltaY;
		ad->camRot.x += dt * input->mouseDeltaX;

		float margin = 0.00001f;
		clamp(&ad->camRot.y, -M_PI+margin, M_PI-margin);
	}

	Vec3 gUp = vec3(0,1,0);
	Vec3 cLook = ad->camLook;
	rotateVec3(&cLook, ad->camRot.x, gUp);
	rotateVec3(&cLook, ad->camRot.y, normVec3(cross(gUp, cLook)));
	Vec3 cUp = normVec3(cross(cLook, normVec3(cross(gUp, cLook))));
	Vec3 cRight = normVec3(cross(gUp, cLook));

	if( input->keysDown[VK_W] || input->keysDown[VK_A] || input->keysDown[VK_S] || 
		input->keysDown[VK_D] || input->keysDown[VK_SHIFT] || input->keysDown[VK_CONTROL]) {

		Vec3 gUp = vec3(0,1,0);
		float speed = 0.1f;
		if(input->mouseButtonDown[0]) speed = 0.5f;
		if(input->keysDown[VK_W]) 		ad->camPos += -normVec3(cLook)*speed;
		if(input->keysDown[VK_A]) 		ad->camPos += -normVec3(cRight)*speed;
		if(input->keysDown[VK_S]) 		ad->camPos += normVec3(cLook)*speed;
		if(input->keysDown[VK_D]) 		ad->camPos += normVec3(cRight)*speed;
		if(input->keysDown[VK_SHIFT]) 	ad->camPos += normVec3(gUp)*speed;
		if(input->keysDown[VK_CONTROL])	ad->camPos += -normVec3(gUp)*speed;
	}

	glViewport(0,0, wSettings->currentRes.x, wSettings->currentRes.y);
	glClearColor(0.3f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// glDepthRange(-1.0,1.0);
	glEnable(GL_DEPTH_TEST);


	if(second) {
		GLenum glError = glGetError(); printf("GLError: %i\n", glError);
	}







	// glEnable(GL_CULL_FACE);
	glDisable(GL_CULL_FACE);
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



	float camera_pos[4] = {ad->camPos.x, ad->camPos.y, ad->camPos.z, 1};
	int tex_unit[2];

	int i;
	for (i=0; i < STBVOX_UNIFORM_count; ++i) {
		stbvox_uniform_info sui;
		if (stbvox_get_uniform_info(&sui, i)) {
			if(i == STBVOX_UNIFORM_transform) continue;

			for(int shaderStage = 0; shaderStage < 2; shaderStage++) {
				GLint location;
				GLuint program;
				if(shaderStage == 0) {
					location = glGetUniformLocation(ad->voxelVertex, sui.name);
					program = ad->voxelVertex;
				} else {
					location = glGetUniformLocation(ad->voxelFragment, sui.name);
					program = ad->voxelFragment;
				}

				if (location != -1) {
					int arrayLength = sui.array_length;
					void* data = sui.default_value;

					switch (i) {
						case STBVOX_UNIFORM_camera_pos: { // only needed for fog
						   		data = camera_pos;
						   } break;

						case STBVOX_UNIFORM_tex_array: {
							tex_unit[0] = 0;
							tex_unit[1] = 0;

							data = tex_unit;
						} break;

						//       case STBVOX_UNIFORM_face_data: {
						//          // glUniform1i(loc, SAMPLER_YOU_WILL_BIND_PER_MESH_FACE_DATA_TO);
						   // GLint sampler_unit = 2;
						//         	   // glUniform1iv(loc, 1, &sampler_unit);
						//         	   if(loc2 != -1) glProgramUniform1iv(program, loc2, 1, &sampler_unit);
						//         	   if(loc3 != -1) glProgramUniform1iv(program, loc3, 1, &sampler_unit);
						//       } break;

						case STBVOX_UNIFORM_ambient: {	 // you definitely want to override this
							// float ambient[4][4] = { 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4};
							// glProgramUniform4fv(program, loc3, sui.array_length, sui.default_value);
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

	GLuint transformUniform1 = glGetUniformLocation(ad->voxelVertex, "transform");
	glProgramUniform3fv(ad->voxelVertex, transformUniform1, 3, ad->transform[0]);
	GLuint transformUniform2 = glGetUniformLocation(ad->voxelFragment, "transform");
	glProgramUniform3fv(ad->voxelFragment, transformUniform2, 3, ad->transform[0]);

	// Vec3 trans = vec3(0,0,0);
	// Vec3 scale = vec3(1, 1, 1);
	// // Vec3 scale = vec3(1, 1, 1);
	// float degrees = 0;
	// Vec3 rot = vec3(1,0,0);

	// Mat4 sm; scaleMatrix(&sm, scale);
	// Mat4 rm; quatRotationMatrix(&rm, quat(degrees, rot));
	// Mat4 tm; translationMatrix(&tm, trans);
	// Mat4 model = tm*rm*sm;

	Mat4 view;
	viewMatrix(&view, ad->camPos, cLook, cUp);
	Mat4 proj;
	projMatrix(&proj, degreeToRadian(60), ad->aspectRatio, 0.1f, 2000);

	// Mat4 finalMat = proj*view*model;
	Mat4 finalMat = proj*view;
	GLint modelViewUni = glGetUniformLocation(ad->voxelVertex, "model_view");
	glProgramUniformMatrix4fv(ad->voxelVertex, modelViewUni, 1, 1, finalMat.e);



	glBindTextures(0,2,ad->voxelTextures);
	glBindTextures(0,2,ad->voxelSamplers);

	glBindProgramPipeline(ad->shader);
	glDrawArrays(GL_QUADS, 0, ad->quadCount*4);



	// // glDisableVertexAttrribArray(0);
	// glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	// glActiveTextureARB(GL_TEXTURE0_ARB);

	// glUseProgram(0);

	// glDisable(GL_BLEND);
	// glDisable(GL_CULL_FACE);
	// glDisable(GL_DEPTH_TEST);

	// glDisable(GL_TEXTURE_2D);






	// lookAt(&ad->pipelineIds, ad->camPos, cLook, cUp);
	// perspective(&ad->pipelineIds, degreeToRadian(60), ad->aspectRatio, 0.1f, 2000);
	// glBindProgramPipeline(ad->pipelineIds.programCube);

	// static float dt = 0;
	// dt += 0.01f;
	// drawCube(&ad->pipelineIds, vec3(5,5,-5), vec3(6,2,1), dt, normVec3(vec3(0.9f,0.6f,0.2f)));

	// for(int i = -10; i < 10; i++) 
	// 	drawCube(&ad->pipelineIds, vec3(i*5,0,0), vec3(1,1,1), 0, normVec3(vec3(0.9f,0.6f,0.2f)));
	// for(int i = -10; i < 10; i++) 
	// 	drawCube(&ad->pipelineIds, vec3(0,i*5,0), vec3(1,1,1), 0, normVec3(vec3(0.9f,0.6f,0.2f)));
	// for(int i = -10; i < 10; i++) 
	// 	drawCube(&ad->pipelineIds, vec3(0,0,i*5), vec3(1,1,1), 0, normVec3(vec3(0.9f,0.6f,0.2f)));



	// ortho(&ad->pipelineIds, rectCenDim(cam->x,cam->y, cam->z, cam->z/ad->aspectRatio));
	// glBindProgramPipeline(ad->pipelineIds.programQuad);
	// drawRect(ad->pipelineIds, rectCenDim(0, 0, 0.01f, 100), rect(0,0,1,1), vec4(0.4f,1,0.4f,1), ad->textures[0]);
	// drawRect(ad->pipelineIds, rectCenDim(0, 0, 100, 0.01f), rect(0,0,1,1), vec4(0.4f,0.4f,1,1), ad->textures[0]);

	// drawRect(ad->pipelineIds, rectCenDim(0, 0, 5, 5), rect(0,0,1,1), vec4(1,1,1,1), ad->textures[2]);
	// drawRect(ad->pipelineIds, rectCenDim(0, 0, 5, 5), rect(0,0,1,1), vec4(1,1,1,1), 3);



	ortho(&ad->pipelineIds, rect(0, -wSettings->currentRes.h, wSettings->currentRes.w, 0));
	glBindProgramPipeline(ad->pipelineIds.programQuad);
	drawTextA(&ad->pipelineIds, vec2(0,-30),  vec4(1,1,1,1), &ad->fontArial, 0, 2, "Pos  : (%f,%f,%f)", ad->camPos.x, ad->camPos.y, ad->camPos.z);
	drawTextA(&ad->pipelineIds, vec2(0,-60),  vec4(1,1,1,1), &ad->fontArial, 0, 2, "Look : (%f,%f,%f)", cLook.x, cLook.y, cLook.z);
	drawTextA(&ad->pipelineIds, vec2(0,-90),  vec4(1,1,1,1), &ad->fontArial, 0, 2, "Up   : (%f,%f,%f)", cUp.x, cUp.y, cUp.z);
	drawTextA(&ad->pipelineIds, vec2(0,-120), vec4(1,1,1,1), &ad->fontArial, 0, 2, "Right: (%f,%f,%f)", cRight.x, cRight.y, cRight.z);
	drawTextA(&ad->pipelineIds, vec2(0,-150), vec4(1,1,1,1), &ad->fontArial, 0, 2, "Rot  : (%f,%f)", ad->camRot.x, ad->camRot.y);


	swapBuffers(&ad->systemData);
	glFinish();

	clearTMemory();
}
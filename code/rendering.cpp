
struct DrawCommandList;
struct GraphicsState;
extern DrawCommandList* globalCommandList;
extern GraphicsState* globalGraphicsState;

#define getPStruct(type) 		(type*)(getPMemory(sizeof(type)))
#define getPArray(type, count) 	(type*)(getPMemory(sizeof(type) * count))
#define getTStruct(type) 		(type*)(getTMemory(sizeof(type)))
#define getTArray(type, count) 	(type*)(getTMemory(sizeof(type) * count))
#define getTString(size) 		(char*)(getTMemory(size)) 
#define getDStruct(type) 		(type*)(getDMemory(sizeof(type)))
#define getDArray(type, count) 	(type*)(getDMemory(sizeof(type) * count))

// @Cleanup

struct MemoryBlock;
void* getPMemory(int size, MemoryBlock * memory = 0);
void* getTMemory(int size, MemoryBlock * memory = 0);


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
		} if(text[ti] == '%' && text[ti+1] == 's') {
			char* str = va_arg(vl, char*);
			int sLen = strLen(str);
			memCpy(buffer + bi, str, sLen);

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

	int wrapWidth;
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

void dcText(char* text, Font* font, Vec2 pos, Vec4 color, Vec2i align = vec2i(-1,1), int wrapWidth = 0, int shadow = 0, Vec4 shadowColor = vec4(0,0,0,1), DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Text, Text);

	command->text = text;
	command->font = font;
	command->pos = pos;
	command->color = color;
	command->vAlign = align.x;
	command->hAlign = align.y;
	command->shadow = shadow;
	command->shadowColor = shadowColor;
	command->wrapWidth = wrapWidth;
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
	"..\\data\\Textures\\Misc\\white.png",
	"..\\data\\Textures\\Misc\\rect.png",
	"..\\data\\Textures\\Misc\\circle.png",
	"..\\data\\Textures\\Misc\\test.png",
};

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
							"..\\data\\Textures\\Skyboxes\\xoGVD3X.jpg",
						  };

// enum TextureType {
// 	TEXTURE_TYPE_
// };

struct Texture {
	// char* name;
	uint id;
	Vec2i dim;
	int channels;
	int levels;
};

int getMaximumMipmapsFromSize(int size) {
	int mipLevels = 1;
	while(size >= 2) {
		size /= 2;
		mipLevels++;
	}

	return mipLevels;
}

void loadTexture(Texture* texture, unsigned char* buffer, int w, int h, int mipLevels, int internalFormat, int channelType, int channelFormat, bool reload = false) {

	if(!reload) {
		glCreateTextures(GL_TEXTURE_2D, 1, &texture->id);
		glTextureStorage2D(texture->id, mipLevels, internalFormat, w, h);

		texture->dim = vec2i(w,h);
		texture->channels = 4;
		texture->levels = mipLevels;
	}	

	glTextureSubImage2D(texture->id, 0, 0, 0, w, h, channelType, channelFormat, buffer);
	glGenerateTextureMipmap(texture->id);
}

void loadTextureFromFile(Texture* texture, char* path, int mipLevels, int internalFormat, int channelType, int channelFormat, bool reload = false) {
	int x,y,n;
	unsigned char* stbData = stbi_load(path, &x, &y, &n, 0);

	if(mipLevels == -1) mipLevels = getMaximumMipmapsFromSize(min(x,y));
	
	loadTexture(texture, stbData, x, y, mipLevels, internalFormat, channelType, channelFormat, reload);

	stbi_image_free(stbData);
}

void loadCubeMapFromFile(Texture* texture, char* filePath, int mipLevels, int internalFormat, int channelType, int channelFormat, bool reload = false) {
	int texWidth, texHeight, n;
	uint* stbData = (uint*)stbi_load(filePath, &texWidth, &texHeight, &n, 4);

	int skySize = texWidth/(float)4;

	if(!reload) {
		texture->dim = vec2i(skySize, skySize);
		texture->channels = 4;
		texture->levels = 6;

		glCreateTextures(GL_TEXTURE_CUBE_MAP_ARRAY, CUBEMAP_SIZE, &texture->id);
		glTextureStorage3D(texture->id, mipLevels, internalFormat, skySize, skySize, 6);
	}

	uint* skyTex = getTArray(uint, skySize*skySize);
	Vec2i texOffsets[] = {{2,1}, {0,1}, {1,0}, {1,2}, {1,1}, {3,1}};
	for(int i = 0; i < 6; i++) {
		Vec2i offset = texOffsets[i] * skySize;

		for(int x = 0; x < skySize; x++) {
			for(int y = 0; y < skySize; y++) {
				skyTex[y*skySize + x] = stbData[(offset.y+y)*texWidth + (offset.x+x)];
			}
		}

		glTextureSubImage3D(texture->id, 0, 0, 0, i, skySize, skySize, 1, channelType, channelFormat, skyTex);
	}
	// glGenerateTextureMipmap(ad->cubemapTextureId);

	stbi_image_free(stbData);
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
	"..\\data\\Fonts\\LiberationMono-Bold.ttf",
	"..\\data\\Fonts\\SourceSansPro-Regular.ttf",
	"..\\data\\Fonts\\consola.ttf",
	"..\\data\\Fonts\\arial.ttf",
	"..\\data\\Fonts\\calibri.ttf",
};

struct Font {
	char* fileBuffer;
	Texture tex;
	int glyphStart, glyphCount;
	stbtt_bakedchar* cData;
	int height;
	float baseOffset;
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

enum SamplerType {
	SAMPLER_NORMAL = 0,
	SAMPLER_VOXEL_1,
	SAMPLER_VOXEL_2,
	SAMPLER_VOXEL_3,
	SAMPLER_SIZE,
};

struct GraphicsState {
	Shader shaders[SHADER_SIZE];
	Texture textures[TEXTURE_SIZE];

	Texture cubeMaps[CUBEMAP_SIZE];

	Texture textures3d[2];
	GLuint samplers[SAMPLER_SIZE];

	Font fonts[FONT_SIZE][20];

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

Texture* getCubemap(int textureId) {
	Texture* t = globalGraphicsState->cubeMaps + textureId;
	return t;
}

Font* getFont(int fontId, int height) {

	// Search if Font in this size exists, if not create it.

	int fontSlots = arrayCount(globalGraphicsState->fonts[0]);
	Font* fontArray = globalGraphicsState->fonts[fontId];
	Font* fontSlot = 0;
	for(int i = 0; i < fontSlots; i++) {
		int fontHeight = fontArray[i].height;
		if(fontHeight == height || fontHeight == 0) {
			fontSlot = fontArray + i;
			break;
		}
	}

	// We are going to assume for now that a font size of 0 means it is uninitialized.

	if(fontSlot->height == 0) {
		Font font;
		char* path = fontPaths[fontId];

		// font.fileBuffer = (char*)getPMemory(fileSize(path) + 1);
		char* fileBuffer = getTArray(char, fileSize(path) + 1);

		readFileToBuffer(fileBuffer, path);
		Vec2i size = vec2i(512,512);
		unsigned char* fontBitmapBuffer = (unsigned char*)getTMemory(size.x*size.y);
		unsigned char* fontBitmap = (unsigned char*)getTMemory(size.x*size.y*4);
		
		font.height = height;
		font.baseOffset = 0.8f;
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
		loadTexture(&tex, fontBitmap, size.w, size.h, 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
		font.tex = tex;

		*fontSlot = font;
	}

	return fontSlot;
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




enum TextStatus {
	TEXTSTATUS_END = 0, 
	TEXTSTATUS_NEWLINE, 
	TEXTSTATUS_WRAPPED, 
	TEXTSTATUS_DEFAULT, 
	TEXTSTATUS_SIZE, 
};

void getTextQuad(char c, Font* font, Vec2 pos, Rect* r, Rect* uv) {
	stbtt_aligned_quad q;
	stbtt_GetBakedQuad(font->cData, font->tex.dim.w, font->tex.dim.h, c-font->glyphStart, pos.x, pos.y, &q);

	float baseLine = 0.8f;
	float off = baseLine * font->height;

	*r = rect(q.x0, q.y0 - off, q.x1, q.y1 - off);
	*uv = rect(q.s0, q.t0, q.s1, q.t1);
}

float getCharAdvance(char c, Font* font) {
	float result = stbtt_GetCharAdvance(font->cData, c-font->glyphStart);
	return result;
}



struct TextInfo {
	Vec2 pos;
	int index;
	Vec2 posAdvance;
	Rect r;
	Rect uv;

	bool lineBreak;
	Vec2 breakPos;
};

struct TextSimInfo {
	Vec2 pos;
	int index;
	int wrapIndex;

	bool lineBreak;
	Vec2 breakPos;
};

TextSimInfo initTextSimInfo(Vec2 startPos) {
	TextSimInfo tsi;
	tsi.pos = startPos;
	tsi.index = 0;
	tsi.wrapIndex = 0;
	tsi.lineBreak = false;
	tsi.breakPos = vec2(0,0);
	return tsi;
}

int textSim(char* text, Font* font, TextSimInfo* tsi, TextInfo* ti, Vec2 startPos = vec2(0,0), int wrapWidth = 0) {
	if(text[tsi->index] == '\0') {
		ti->pos = tsi->pos;
		ti->index = tsi->index;
		return 0;
	}

	ti->lineBreak = false;

	if(tsi->lineBreak) {
		ti->lineBreak = true;
		ti->breakPos = tsi->breakPos;
		tsi->lineBreak = false;
	}

	Vec2 oldPos = tsi->pos;

	int i = tsi->index;
	char t = text[i];

	bool wrapped = false;

	if(wrapWidth != 0 && i == tsi->wrapIndex) {
		char c = text[i];
		float wordWidth = 0;
		if(c == '\n') wordWidth = getCharAdvance(c, font);

		int it = i;
		while(c != '\n' && c != '\0' && c != ' ') {
			wordWidth += getCharAdvance(c, font);
			it++;
			c = text[it];
		}

		if(tsi->pos.x + wordWidth > startPos.x + wrapWidth) {
			wrapped = true;
		}

		if(it != i) tsi->wrapIndex = it;
		else tsi->wrapIndex++;
	}

	if(t == '\n' || wrapped) {
		tsi->lineBreak = true;
		if(t == '\n') tsi->breakPos = tsi->pos + vec2(getCharAdvance(t, font),0);
		if(wrapped) tsi->breakPos = tsi->pos;

		tsi->pos.x = startPos.x;
		tsi->pos.y -= font->height;

		if(wrapped) {
			return textSim(text, font, tsi, ti, startPos, wrapWidth);
		}
	} else {
		getTextQuad(t, font, tsi->pos, &ti->r, &ti->uv);
		tsi->pos.x += getCharAdvance(t, font);
	}

	if(ti) {
		ti->pos = oldPos;
		ti->index = tsi->index;
		ti->posAdvance = tsi->pos - oldPos;
	}

	tsi->index++;

	return 1;
}

Vec2 getTextDim(char* text, Font* font, Vec2 startPos = vec2(0,0), int wrapWidth = 0) {
	float maxX = startPos.x;

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		if(!textSim(text, font, &tsi, &ti, startPos, wrapWidth)) break;

		maxX = max(maxX, ti.pos.x + ti.posAdvance.x);
	}

	Vec2 dim = vec2(maxX - startPos.x, startPos.y - (tsi.pos.y - font->height));

	return dim;
}

Vec2 testgetTextStartPos(char* text, Font* font, Vec2 startPos, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	Vec2 dim = getTextDim(text, font, startPos, wrapWidth);
	startPos.x -= (align.x+1)*0.5f*dim.w;
	startPos.y -= (align.y-1)*0.5f*dim.h;

	return startPos;
}

void drawText(char* text, Font* font, Vec2 startPos, Vec4 color, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);
	startPos = vec2(roundInt((int)startPos.x), roundInt((int)startPos.y));

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		if(!textSim(text, font, &tsi, &ti, startPos, wrapWidth)) break;
		if(text[ti.index] == '\n') continue;

		drawRect(ti.r, ti.uv, color, font->tex.id);
	}
}

Rect getTextCursor(char* text, Font* font, Vec2 startPos, int index, float cursorWidth, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		int result = textSim(text, font, &tsi, &ti, startPos, wrapWidth);

		if(ti.index == index) {
			Rect cursorRect = rectCenDim(ti.pos - vec2(0, font->height/2), vec2(cursorWidth,font->height));
			return cursorRect;
		}

		if(!result) break;
	}

	return rect(0,0,0,0);
}

void drawTextSelection(char* text, Font* font, Vec2 startPos, int index1, int index2, Vec4 color, Vec2i align = vec2i(-1,1), int wrapWidth = 0, DrawCommandList* drawList = 0) {
	if(index1 == index2) return;
	if(index1 > index2) swap(&index1, &index2);

	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

	Vec2 lineStart;
	bool drawSelection = false;

	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		int result = textSim(text, font, &tsi, &ti, startPos, wrapWidth);

		bool endReached = ti.index == index2;

		if(drawSelection) {
			if(ti.lineBreak || endReached) {

				Vec2 lineEnd = ti.lineBreak ? ti.breakPos : ti.pos;
				if(!result) lineEnd = tsi.pos;

				Rect r = rect(lineStart - vec2(0,font->height), lineEnd);
				dcRect(r, color, drawList);

				lineStart = ti.pos;

				if(endReached) break;
			}
		}

		if(!drawSelection && (ti.index >= index1)) {
			drawSelection = true;
			lineStart = ti.pos;
		}

		if(!result) break;
	}
}

int textMouseToIndex(char* text, Font* font, Vec2 startPos, Vec2 mousePos, Vec2i align = vec2i(-1,1), int wrapWidth = 0) {
	startPos = testgetTextStartPos(text, font, startPos, align, wrapWidth);

	if(mousePos.y > startPos.y) return 0;
	
	bool foundLine = false;
	TextSimInfo tsi = initTextSimInfo(startPos);
	while(true) {
		TextInfo ti;
		if(!textSim(text, font, &tsi, &ti, startPos, wrapWidth)) break;
		
		bool fLine = valueBetween(mousePos.y, ti.pos.y - font->height, ti.pos.y);
		if(fLine) foundLine = true;
		else if(foundLine) return ti.index-1;

	    if(foundLine) {
	    	float charMid = ti.pos.x + ti.posAdvance.x*0.5f;
			if(mousePos.x < charMid) return ti.index;
		}
	}

	return tsi.index;
}

char* textSelectionToString(char* text, int index1, int index2) {
	assert(index1 >= 0 && index2 >= 0);

	int range = abs(index1 - index2);
	char* str = getTStringDebug(range + 1); // We assume text selection will only be used for debug things.
	strCpy(str, text + minInt(index1, index2), range);
	return str;
}




void drawCube(Vec3 trans, Vec3 scale, Vec4 color, float degrees, Vec3 rot) {
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

	// Disabling these arrays is very important.

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

	// Disabling these arrays is very important.

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	
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

void executeCommandList(DrawCommandList* list, bool print = false) {
	// TIMER_BLOCK();

	if(print) {
		printf("\nDrawCommands: %i \n", list->count);
	}

	char* drawListIndex = (char*)list->data;
	for(int i = 0; i < list->count; i++) {
		int command = *((int*)drawListIndex);
		drawListIndex += sizeof(int);

		if(print) {
			printf("%i ", command);
		}

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

				if(dc.shadow != 0) 
					drawText(dc.text, dc.font, dc.pos + vec2(dc.shadow,-dc.shadow), dc.shadowColor, vec2i(dc.vAlign, dc.hAlign), dc.wrapWidth);
				drawText(dc.text, dc.font, dc.pos, dc.color, vec2i(dc.vAlign, dc.hAlign), dc.wrapWidth);
			} break;

			case Draw_Command_Scissor_Type: {
				dcGetStructAndIncrement(Scissor);
				Rect r = dc.rect;
				Vec2 dim = rectGetDim(r);
				assert(dim.w >= 0 && dim.h >= 0);
				glScissor(r.min.x, r.min.y, dim.x, dim.y);
			} break;

			default: {} break;
		}
	}

	if(print) {
		printf("\n\n");
	}
}


extern DrawCommandList* theCommandList;

enum CommandState {
	STATE_SCISSOR,
	STATE_POLYGONMODE, 
	STATE_POLYGON_OFFSET, 
	STATE_LINEWIDTH,
	STATE_CULL,
	STATE_DEPTH_TEST,
	STATE_DEPTH_FUNC,
};

enum Polygon_Mode {
	POLYGON_MODE_FILL = 0,
	POLYGON_MODE_LINE,
	POLYGON_MODE_POINT,
};

#define Command_List(func) \
	func(State) \
	func(Enable) \
	func(Disable) \
	func(Cube) \
	func(Line) \
	func(Line2d) \
	func(Quad) \
	func(Quad2d) \
	func(Rect) \
	func(RoundedRect) \
	func(RoundedRectGradient) \
	func(RoundedRectOutline) \
	func(Text) \
	func(TextSelection) \
	func(Scissor) \
	func(Blend) \
	func(PolygonOffset) \
	func(Triangle) \

#define Make_Enum(name) Draw_Command_Type_##name,
enum DrawListCommand {
	Command_List(Make_Enum)
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

//

#pragma pack(push,1)

#define func(type, name) type name;
#define Make_Command_Struct(name) \
	struct Draw_Command_##name { \
		List_##name \
	};

#define LIST_FUNC(type, name) type name;

#define List_Enable
#define List_Disable

#define List_Cube \
	LIST_FUNC( Vec3, trans )\
	LIST_FUNC( Vec3, scale )\
	LIST_FUNC( Vec4, color )\
	LIST_FUNC( float, degrees )\
	LIST_FUNC( Vec3, rot)

#define List_Line \
	LIST_FUNC( Vec3, p0 )\
	LIST_FUNC( Vec3, p1 )\
	LIST_FUNC( Vec4, color )\

#define List_Line2d \
	LIST_FUNC( Vec2, p0 )\
	LIST_FUNC( Vec2, p1 )\
	LIST_FUNC( Vec4, color )\

#define List_Quad \
	LIST_FUNC( Vec3, p0  )\
	LIST_FUNC( Vec3, p1  )\
	LIST_FUNC( Vec3, p2  )\
	LIST_FUNC( Vec3, p3 )\
	LIST_FUNC( Vec4, color )\
	LIST_FUNC( int, textureId )\
	LIST_FUNC( Rect, uv )\
	LIST_FUNC( int, texZ )\

#define List_Quad2d \
	LIST_FUNC( Vec2, p0 )\
	LIST_FUNC( Vec2, p1 )\
	LIST_FUNC( Vec2, p2 )\
	LIST_FUNC( Vec2, p3 )\
	LIST_FUNC( Vec4, color )\
	LIST_FUNC( int, textureId )\
	LIST_FUNC( Rect, uv )\
	LIST_FUNC( int, texZ )\

#define List_Rect \
	LIST_FUNC( Rect, r )\
	LIST_FUNC( Rect, uv )\
	LIST_FUNC( Vec4, color )\
	LIST_FUNC( int, textureId )\
	LIST_FUNC( int, texZ )\

#define List_RoundedRect \
	LIST_FUNC( Rect, r )\
	LIST_FUNC( Vec4, color )\
	LIST_FUNC( float, steps )\
	LIST_FUNC( float, size )\

#define List_RoundedRectGradient \
	LIST_FUNC( Rect, r )\
	LIST_FUNC( Vec4, color )\
	LIST_FUNC( float, steps )\
	LIST_FUNC( float, size )\
	LIST_FUNC( Vec4, off )\

#define List_RoundedRectOutline \
	LIST_FUNC( Rect, r )\
	LIST_FUNC( Vec4, color )\
	LIST_FUNC( float, steps )\
	LIST_FUNC( float, size )\
	LIST_FUNC( float, offset )\

#define List_Text \
	LIST_FUNC( char*, text )\
	LIST_FUNC( Vec2, pos )\
	LIST_FUNC( Vec2i, align )\
	LIST_FUNC( int, wrapWidth )\
	LIST_FUNC( TextSettings, settings )\

#define List_TextSelection \
	LIST_FUNC( char*, text )\
	LIST_FUNC( Font*, font )\
	LIST_FUNC( Vec2, startPos )\
	LIST_FUNC( int, index1 )\
	LIST_FUNC( int, index2 )\
	LIST_FUNC( Vec4, color )\
	LIST_FUNC( Vec2i, align )\
	LIST_FUNC( int, wrapWidth )\

#define List_Scissor \
	LIST_FUNC( Rect, rect )\

#define List_Int \
	LIST_FUNC( int, state )\

#define List_State \
	LIST_FUNC( int, state )\
	LIST_FUNC( int, value )\

#define List_Blend \
	LIST_FUNC( int, sourceColor )\
	LIST_FUNC( int, destinationColor )\
	LIST_FUNC( int, sourceAlpha )\
	LIST_FUNC( int, destinationAlpha )\
	LIST_FUNC( int, functionColor )\
	LIST_FUNC( int, functionAlpha )\

#define List_PolygonOffset \
	LIST_FUNC( float, factor )\
	LIST_FUNC( float, units )\

#define List_Triangle \
	LIST_FUNC( Vec2, p )\
	LIST_FUNC( float, size )\
	LIST_FUNC( Vec2, dir )\
	LIST_FUNC( Vec4, color )\

Command_List(Make_Command_Struct)

struct Draw_Command_Int {
	int state;
};

#pragma pack(pop)

//

#define PUSH_DRAW_COMMANDS(commandType, structType) \
	if(!drawList) drawList = theCommandList; \
	char* list = (char*)drawList->data + drawList->bytes; \
	*((int*)list) = Draw_Command_Type_##commandType; \
	list += sizeof(int); \
	Draw_Command_##structType* command = (Draw_Command_##structType*)list; \
	drawList->count++; \
	drawList->bytes += sizeof(Draw_Command_##structType) + sizeof(int); \
	assert(sizeof(Draw_Command_##structType) + drawList->bytes < drawList->maxBytes);

#define PUSH_DRAW_COMMAND(commandStructType) \
	PUSH_DRAW_COMMANDS(commandStructType, commandStructType)

#define PUSH_DRAW_COMMANDS_AUTO(commandType, structType) \
	PUSH_DRAW_COMMANDS(commandType, structType) \
	List_##structType

#define PUSH_DRAW_COMMAND_AUTO(commandStructType) \
	PUSH_DRAW_COMMANDS(commandStructType, commandStructType) \
	List_##commandStructType

#define LIST_FUNC(type, name) command->name = name;

//

void dcCube(Vec3 trans, Vec3 scale, Vec4 color, float degrees = 0, Vec3 rot = vec3(0,0,0), DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Cube);
}

void dcLine(Vec3 p0, Vec3 p1, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Line);
}

void dcLine2d(Vec2 p0, Vec2 p1, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Line2d);
}

void dcQuad2d(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, Vec4 color, int textureId = -1, Rect uv = rect(0,0,1,1), int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Quad2d);

	command->textureId = textureId == -1 ? theGraphicsState->textureWhite->id : textureId;
}

void dcQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color, int textureId = -1, Rect uv = rect(0,0,1,1), int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Quad);

	command->textureId = textureId == -1 ? theGraphicsState->textureWhite->id : textureId;
}
void dcQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color, char* textureName, Rect uv = rect(0,0,1,1), int texZ = -1, DrawCommandList* drawList = 0) {
	int textureId = strLen(textureName) ? getTexture(textureName)->id : -1;
	dcQuad(p0, p1, p2, p3, color, textureId, uv, texZ, drawList);
}

void dcRect(Rect r, Rect uv, Vec4 color, int textureId = -1, int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Rect);
}
void dcRect(Rect r, Rect uv, Vec4 color, char* textureName, int texZ = -1, DrawCommandList* drawList = 0) {
	dcRect(r, uv, color, getTexture(textureName)->id, texZ, drawList);
}
void dcRect(Rect r, Vec4 color, DrawCommandList* drawList = 0) {
	dcRect(r, rect(0,0,1,1), color, theGraphicsState->textureWhite->id, -1, drawList);
}

void dcRoundedRect(Rect r, Vec4 color, float size, float steps = 0, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(RoundedRect);
}

void dcRoundedRectGradient(Rect r, Vec4 color, float size, Vec4 off, float steps = 0, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(RoundedRectGradient);
}

void dcRoundedRectOutline(Rect r, Vec4 color, float size, float offset = -1, float steps = 0, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(RoundedRectOutline);
}

void dcText(char* text, Vec2 pos, Vec2i align, int wrapWidth, TextSettings settings, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Text);
}
void dcText(char* text, Vec2 pos, Vec2i align, TextSettings settings, DrawCommandList* drawList = 0) {
	int wrapWidth = 0;
	PUSH_DRAW_COMMAND_AUTO(Text);
}
void dcText(char* text, Vec2 pos, TextSettings settings, DrawCommandList* drawList = 0) {
	Vec2i align = vec2i(-1,1);
	int wrapWidth = 0;
	PUSH_DRAW_COMMAND_AUTO(Text);
}

void dcText(char* text, Font* font, Vec2 pos, Vec4 color, Vec2i align = vec2i(-1,1), int wrapWidth = 0, int shadow = 0, Vec4 shadowColor = vec4(0,0,0,1), DrawCommandList* drawList = 0) {

	int cullWidth = -1;

	if(cullWidth == -1) {
		int shadowType = TEXTSHADOW_MODE_NOSHADOW;
		if(shadow != 0) shadowType = TEXTSHADOW_MODE_SHADOW;
		TextSettings ts = textSettings(font, color, shadowType, vec2(shadow,-shadow), len(vec2(shadow,-shadow)), shadowColor);

		dcText(text, pos, align, wrapWidth, ts);

	} else {
		TextSettings ts = textSettings(font, color, TEXTSHADOW_MODE_SHADOW, vec2(shadow,-shadow), len(vec2(shadow,-shadow)), shadowColor);
		ts.cull = true;

		dcText(text, pos, align, cullWidth, ts);
	}
}

void dcTextLine(char* text, Font* font, Vec2 pos, Vec4 color, Vec2i align = vec2i(-1,1), int cullWidth = -1, int shadow = 0, Vec4 shadowColor = vec4(0,0,0,1), DrawCommandList* drawList = 0) {
	dcText(text, font, pos, color, align, 0, shadow, shadowColor);
}

void dcTextSelection(char* text, Font* font, Vec2 startPos, int index1, int index2, Vec4 color, Vec2i align = vec2i(-1,1), int wrapWidth = 0, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(TextSelection);
}

void dcScissor(Rect rect, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Scissor);
}

void dcState(int state, int value, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(State);
}

void dcEnable(int state, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMANDS_AUTO(Enable, Int);
}

void dcDisable(int state, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMANDS_AUTO(Disable, Int);
}

void dcBlend(int sourceColor, int destinationColor, int sourceAlpha, int destinationAlpha, int functionColor, int functionAlpha, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Blend);
}

void dcBlend(int source, int destination, int function, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Blend);

	command->sourceColor = source;
	command->destinationColor = destination;
	command->sourceAlpha = source;
	command->destinationAlpha = destination;
	command->functionColor = function;
	command->functionAlpha = function;
}

void dcPolygonOffset(float factor, float units, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(PolygonOffset);
}

void dcTriangle(Vec2 p, float size, Vec2 dir, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND_AUTO(Triangle);
}

//

int stateSwitch(int state) {
	switch(state) {
		case STATE_CULL: return GL_CULL_FACE;
		case STATE_SCISSOR: return GL_SCISSOR_TEST;
		case STATE_DEPTH_TEST: return GL_DEPTH_TEST;
		case STATE_POLYGON_OFFSET: return GL_POLYGON_OFFSET_FILL;
	}
	return 0;
}

#define CONCAT(a, b) a##b

#define dcGetStructAndIncrement(structType) \
	CONCAT(Draw_Command_, structType) dc = *((CONCAT(Draw_Command_, structType)*)drawListIndex); \
	drawListIndex += sizeof(CONCAT(Draw_Command_, structType)); \

#define Make_Case(name) CONCAT(Draw_Command_Type_, name)

//

void executeCommandList(DrawCommandList* list, bool print = false, bool skipStrings = false) {
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

		// #define Make_Case Draw_Command_Type_#CURRENT_NAME

		switch(command) {

			#define CURRENT_NAME Cube
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				drawCube(dc.trans, dc.scale, dc.color, dc.degrees, dc.rot);
			} break;

			#define CURRENT_NAME Line
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				drawLine(dc.p0, dc.p1, dc.color);
			} break;

			#define CURRENT_NAME Line2d
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);

				drawLine(dc.p0, dc.p1, dc.color);
			} break;

			#define CURRENT_NAME Quad2d
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);

				drawQuad(dc.p0, dc.p1, dc.p2, dc.p3, dc.color, dc.textureId, dc.uv, dc.texZ);
			} break;

			#define CURRENT_NAME Quad
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);

				drawQuad(dc.p0, dc.p1, dc.p2, dc.p3, dc.color, dc.textureId, dc.uv, dc.texZ);
			} break;

			#define CURRENT_NAME State
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);

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

					case STATE_DEPTH_FUNC: glDepthFunc(dc.value); break;

					default: {} break;
				}
			} break;

			#define CURRENT_NAME Enable
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(Int);

				int m = stateSwitch(dc.state);
				glEnable(m);
			} break;			

			#define CURRENT_NAME Disable
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(Int);

				int m = stateSwitch(dc.state);
				glDisable(m);
			} break;	

			#define CURRENT_NAME Rect
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				drawRect(dc.r, dc.color, dc.uv, dc.textureId, dc.texZ);
			} break;

			#define CURRENT_NAME RoundedRect
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				drawRectRounded(dc.r, dc.color, dc.size, dc.steps);
			} break;

			#define CURRENT_NAME RoundedRectGradient
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				drawRectRoundedGradient(dc.r, dc.color, dc.size, dc.off, dc.steps);
			} break;

			#define CURRENT_NAME RoundedRectOutline
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				drawRectRoundedOutline(dc.r, dc.color, dc.size, dc.offset, dc.steps);
			} break;

			#define CURRENT_NAME Text
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);

				if(skipStrings) break;

				drawText(dc.text, dc.pos, dc.align, dc.wrapWidth, dc.settings);
			} break;

			#define CURRENT_NAME TextSelection
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);

				if(skipStrings) break;

				drawTextSelection(dc.text, dc.font, dc.startPos, dc.index1, dc.index2, dc.color, dc.align, dc.wrapWidth);
			} break;

			#define CURRENT_NAME Scissor
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				Rect r = dc.rect;
				Vec2 dim = r.dim();
				assert(dim.w >= 0 && dim.h >= 0);
				glScissor(r.min.x, r.min.y, dim.x, dim.y);
			} break;

			#define CURRENT_NAME Blend
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				glBlendFuncSeparate(dc.sourceColor, dc.destinationColor, 
				                    dc.sourceAlpha, dc.destinationAlpha);
				glBlendEquationSeparate(dc.functionColor, dc.functionAlpha);
			} break;

			#define CURRENT_NAME PolygonOffset
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				glPolygonOffset(dc.factor, dc.units);
			} break;

			#define CURRENT_NAME Triangle
			case Make_Case(CURRENT_NAME): {
				dcGetStructAndIncrement(CURRENT_NAME);
				drawTriangle(dc.p, dc.size, dc.dir, dc.color);
			} break;

			default: {} break;

			#undef CURRENT_NAME
		}
	}

	if(print) {
		printf("\n\n");
	}
}


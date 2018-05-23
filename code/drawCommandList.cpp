
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

enum DrawListCommand {
	Draw_Command_State_Type,
	Draw_Command_Enable_Type,
	Draw_Command_Disable_Type,
	Draw_Command_Cube_Type,
	Draw_Command_Line_Type,
	Draw_Command_Line2d_Type,
	Draw_Command_Quad_Type,
	Draw_Command_Quad2d_Type,
	Draw_Command_Rect_Type,
	Draw_Command_RoundedRect_Type,
	Draw_Command_Text_Type,
	Draw_Command_Scissor_Type,
	Draw_Command_Blend_Type,
	Draw_Command_PolygonOffset_Type,
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

#pragma pack(push,1)
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

struct Draw_Command_Line2d {
	Vec2 p0, p1;
	Vec4 color;
};

struct Draw_Command_Quad {
	Vec3 p0, p1, p2, p3;
	Vec4 color;
	int textureId;
	Rect uv;
	int texZ;
};

struct Draw_Command_Quad2d {
	Vec2 p0, p1, p2, p3;
	Vec4 color;
	int textureId;
	Rect uv;
	int texZ;
};

struct Draw_Command_Rect {
	Rect r, uv;
	Vec4 color;
	int textureId;
	int texZ;
};

struct Draw_Command_RoundedRect {
	Rect r;
	Vec4 color;

	float steps;
	float size;
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
	float cullWidth;
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

struct Draw_Command_Blend {
	int sourceColor;
	int destinationColor;
	int sourceAlpha;
	int destinationAlpha;
	int functionColor;
	int functionAlpha;
};

struct Draw_Command_PolygonOffset {
	float factor;
	float units;
};
#pragma pack(pop)


#define PUSH_DRAW_COMMAND(commandType, structType) \
	if(!drawList) drawList = theCommandList; \
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

void dcLine2d(Vec2 p0, Vec2 p1, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Line2d, Line2d);

	command->p0 = p0;
	command->p1 = p1;
	command->color = color;
}

void dcQuad2d(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, Vec4 color, int textureId = -1, Rect uv = rect(0,0,1,1), int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Quad2d, Quad2d);

	command->p0 = p0;
	command->p1 = p1;
	command->p2 = p2;
	command->p3 = p3;
	command->color = color;
	command->textureId = textureId == -1 ? theGraphicsState->textureWhite->id : textureId;
	command->uv = uv;
	command->texZ = texZ;
}

void dcQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color, int textureId = -1, Rect uv = rect(0,0,1,1), int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Quad, Quad);

	command->p0 = p0;
	command->p1 = p1;
	command->p2 = p2;
	command->p3 = p3;
	command->color = color;
	command->textureId = textureId == -1 ? theGraphicsState->textureWhite->id : textureId;
	command->uv = uv;
	command->texZ = texZ;
}
void dcQuad(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, Vec4 color, char* textureName, Rect uv = rect(0,0,1,1), int texZ = -1, DrawCommandList* drawList = 0) {
	int textureId = strLen(textureName) ? getTexture(textureName)->id : -1;
	dcQuad(p0, p1, p2, p3, color, textureId, uv, texZ, drawList);
}

void dcRect(Rect r, Rect uv, Vec4 color, int textureId = -1, int texZ = -1, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Rect, Rect);

	command->r = r;
	command->uv = uv;
	command->color = color;
	command->textureId = textureId;
	command->texZ = texZ;
}
void dcRect(Rect r, Rect uv, Vec4 color, char* textureName, int texZ = -1, DrawCommandList* drawList = 0) {
	int textureId = getTexture(textureName)->id;
	dcRect(r, uv, color, textureId, texZ, drawList);
}
void dcRect(Rect r, Vec4 color, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Rect, Rect);

	command->r = r;
	command->uv = rect(0,0,1,1);
	command->color = color;
	command->textureId = theGraphicsState->textureWhite->id;
	command->texZ = -1;
}

void dcRoundedRect(Rect r, Vec4 color, float size, float steps = 0, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(RoundedRect, RoundedRect);

	command->r = r;
	command->color = color;
	command->steps = steps;
	command->size = size;
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
	command->cullWidth = -1;
}

void dcTextLine(char* text, Font* font, Vec2 pos, Vec4 color, Vec2i align = vec2i(-1,1), int cullWidth = -1, int shadow = 0, Vec4 shadowColor = vec4(0,0,0,1), DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Text, Text);

	command->text = text;
	command->font = font;
	command->pos = pos;
	command->color = color;
	command->vAlign = align.x;
	command->hAlign = align.y;
	command->shadow = shadow;
	command->shadowColor = shadowColor;
	command->wrapWidth = 0;
	command->cullWidth = cullWidth;
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

void dcBlend(int sourceColor, int destinationColor, int sourceAlpha, int destinationAlpha, int functionColor, int functionAlpha, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Blend, Blend);

	command->sourceColor = sourceColor;
	command->destinationColor = destinationColor;
	command->sourceAlpha = sourceAlpha;
	command->destinationAlpha = destinationAlpha;

	command->functionColor = functionColor;
	command->functionAlpha = functionAlpha;
}

void dcBlend(int source, int destination, int function, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(Blend, Blend);

	command->sourceColor = source;
	command->destinationColor = destination;
	command->sourceAlpha = source;
	command->destinationAlpha = destination;

	command->functionColor = function;
	command->functionAlpha = function;
}

void dcPolygonOffset(float factor, float units, DrawCommandList* drawList = 0) {
	PUSH_DRAW_COMMAND(PolygonOffset, PolygonOffset);

	command->factor = factor;
	command->units = units;
}


//

void scissorTest(Rect r) {
	int left   = roundIntf(r.left);
	int bottom = roundIntf(r.bottom);
	int right  = roundIntf(r.right);
	int top    = roundIntf(r.top);

	// glScissor(left, bottom, right-left, top-bottom);
	dcScissor(rect(left, bottom, right, top));
}

Rect scissorRectScreenSpace(Rect r, float screenHeight) {
	Rect scissorRect = {r.min.x, r.min.y+screenHeight, r.max.x, r.max.y+screenHeight};
	return scissorRect;
}

void scissorTestScreen(Rect r) {
	Rect sr = scissorRectScreenSpace(r, theGraphicsState->screenRes.h);
	if(rectW(sr) < 0 || rectH(sr) < 0) sr = rect(0,0,0,0);

	scissorTest(sr);
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

#define dcGetStructAndIncrement(structType) \
	Draw_Command_##structType dc = *((Draw_Command_##structType*)drawListIndex); \
	drawListIndex += sizeof(Draw_Command_##structType); \

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

		switch(command) {
			case Draw_Command_Cube_Type: {
				dcGetStructAndIncrement(Cube);
				drawCube(dc.trans, dc.scale, dc.color, dc.degrees, dc.rot);
			} break;

			case Draw_Command_Line_Type: {
				dcGetStructAndIncrement(Line);
				drawLine(dc.p0, dc.p1, dc.color);
			} break;

			case Draw_Command_Line2d_Type: {
				dcGetStructAndIncrement(Line2d);

				// Vec2 verts[] = {dc.p0, dc.p1};
				Vec2 verts[] = {roundIntf(dc.p0.x)-0.5f, roundIntf(dc.p0.y)-0.5f, roundIntf(dc.p1.x)-0.5f, roundIntf(dc.p1.y)-0.5f};
				pushUniform(SHADER_QUAD, 0, "verts", verts, 2);
				pushUniform(SHADER_QUAD, 0, "setColor", linearToGamma(dc.color));
				pushUniform(SHADER_QUAD, 0, "primitiveMode", 1);

				glBindTextures(0,1,&theGraphicsState->textureWhite->id);

				glDrawArrays(GL_LINES, 0, 2);
			} break;

			case Draw_Command_Quad2d_Type: {
				dcGetStructAndIncrement(Quad2d);

				drawQuad(dc.p0, dc.p1, dc.p2, dc.p3, dc.color, dc.textureId, dc.uv, dc.texZ);
			} break;

			case Draw_Command_Quad_Type: {
				dcGetStructAndIncrement(Quad);

				drawQuad(dc.p0, dc.p1, dc.p2, dc.p3, dc.color, dc.textureId, dc.uv, dc.texZ);
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

					case STATE_DEPTH_FUNC: glDepthFunc(dc.value); break;

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
				drawRect(dc.r, dc.color, dc.uv, dc.textureId, dc.texZ);
			} break;

			case Draw_Command_RoundedRect_Type: {
				dcGetStructAndIncrement(RoundedRect);

				if(dc.steps == 0) dc.steps = 6;

				float s = dc.size;
				Rect r = dc.r;
				drawRect(rect(r.min.x+s, r.min.y, r.max.x-s, r.max.y), dc.color);
				drawRect(rect(r.min.x, r.min.y+s, r.max.x, r.max.y-s), dc.color);

				Vec2 verts[10];

				pushUniform(SHADER_QUAD, 0, "primitiveMode", 1);
				pushUniform(SHADER_QUAD, 0, "setColor", linearToGamma(dc.color));
				glBindTextures(0,1,&theGraphicsState->textureWhite->id);

				Rect rc = rectExpand(r, -vec2(s,s)*2);
				Vec2 corners[] = {rc.max, vec2(rc.max.x, rc.min.y), rc.min, vec2(rc.min.x, rc.max.y)};
				for(int cornerIndex = 0; cornerIndex < 4; cornerIndex++) {

					Vec2 corner = corners[cornerIndex];
					float round = s;
					float start = M_PI_2*cornerIndex;

					verts[0] = corner;

					for(int i = 0; i < dc.steps; i++) {
						float angle = start + i*(M_PI_2/(dc.steps-1));
						Vec2 v = vec2(sin(angle), cos(angle));

						verts[i+1] = corner + v*round;
					}

					pushUniform(SHADER_QUAD, 0, "verts", verts, dc.steps+1);
					glDrawArraysInstancedBaseInstance(GL_TRIANGLE_FAN, 0, dc.steps+1, 1, 0);
				}

			} break;

			case Draw_Command_Text_Type: {
				dcGetStructAndIncrement(Text);

				if(skipStrings) break;

				if(dc.cullWidth == -1) {
					int shadowType = TEXTSHADOW_MODE_NOSHADOW;
					if(dc.shadow != 0) shadowType = TEXTSHADOW_MODE_SHADOW;
					TextSettings ts = textSettings(dc.font, dc.color, shadowType, vec2(dc.shadow,-dc.shadow), len(vec2(dc.shadow,-dc.shadow)), dc.shadowColor);

					drawText(dc.text, dc.pos, vec2i(dc.vAlign, dc.hAlign), dc.wrapWidth, ts);

				} else {
					TextSettings ts = textSettings(dc.font, dc.color, TEXTSHADOW_MODE_SHADOW, vec2(dc.shadow,-dc.shadow), len(vec2(dc.shadow,-dc.shadow)), dc.shadowColor);
					ts.cull = true;

					drawText(dc.text, dc.pos, vec2i(dc.vAlign, dc.hAlign), dc.cullWidth, ts);
				}

			} break;

			case Draw_Command_Scissor_Type: {
				dcGetStructAndIncrement(Scissor);
				Rect r = dc.rect;
				Vec2 dim = rectDim(r);
				assert(dim.w >= 0 && dim.h >= 0);
				glScissor(r.min.x, r.min.y, dim.x, dim.y);
			} break;

			case Draw_Command_Blend_Type: {
				dcGetStructAndIncrement(Blend);
				glBlendFuncSeparate(dc.sourceColor, dc.destinationColor, 
				                    dc.sourceAlpha, dc.destinationAlpha);
				glBlendEquationSeparate(dc.functionColor, dc.functionAlpha);
			} break;

			case Draw_Command_PolygonOffset_Type: {
				dcGetStructAndIncrement(PolygonOffset);
				glPolygonOffset(dc.factor, dc.units);
			} break;

			default: {} break;
		}
	}

	if(print) {
		printf("\n\n");
	}
}


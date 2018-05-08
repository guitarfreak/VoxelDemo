
enum Entity_Type {
	ET_Player = 0,
	ET_Camera,
	ET_Rocket,
	ET_ParticleEmitter,
	ET_BlockResource,

	ET_Size,
};

struct Camera {
	Vec3 pos;
	Vec3 look;
	Vec3 up;
	Vec3 right;
};

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

	Quat rot;
	Vec3 velRot;
	Vec3 accRot;

	float time;
	float timeToLive;
};

struct ParticleEmitter {
	Particle* particleList;
	int particleListSize;
	int particleListCount;
	int particleListIndex;

	Vec3 pos;
	float spawnRate;
	int spawnCount;
	float spawnTime;

	Vec4 color;

	// Set either one.
	float time;
	float timeToLive;

	int liveTimeSpawnCount;
	int liveTimeSpawns;
};

struct Entity {
	int init;

	int type;
	int id;
	char name[16];

	Vec2i chunk;

	Vec3 pos;
	Vec3 dir;
	Vec3 rot;
	float rotAngle;
	Vec3 dim;

	Vec3 vel;
	Vec3 acc;

	int movementType;
	int spatial;

	bool onGround;

	bool deleted;
	bool isMoving;
	bool isColliding;

	union {
		// Player.
		struct {
			Vec3 camOff;
		};

		// ParticleEmitter.
		struct {
			ParticleEmitter emitter;
		};

		// Block.
		struct {
			uchar blockType;
		};

		// Rocket.
		struct {
			bool exploded;
		};
	};

};

struct EntityList {
	Entity* e;
	int size;
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

//

Vec3 getRotationToVector(Vec3 start, Vec3 dest, float* angle) {
	Vec3 side = normVec3(cross(start, normVec3(dest)));
	*angle = dot(start, normVec3(dest));
	*angle = acos(*angle)*2;

	return side;
}	

void initEntity(Entity* e, int type, Vec3 pos, Vec3 dim, Vec2i chunk) {
	*e = {};
	e->init = true;
	e->type = type;
	e->pos = pos;
	e->dim = dim;
	e->chunk = chunk;
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

//

void entityMouseLook(Entity* e, Input* input, float mouseSensitivity) {
	float turnRate = mouseSensitivity * 0.01f;
	e->rot.y -= turnRate * input->mouseDelta.y;
	e->rot.x -= turnRate * input->mouseDelta.x;

	float margin = 0.05f;
	clamp(&e->rot.y, -M_PI_2+margin, M_PI_2-margin);
	e->rot.x = modFloat(e->rot.x, (float)M_PI*2);
}

void entityKeyboardAcceleration(Entity* e, Input* input, float speed, float boost, bool freeForm) {

	Vec3 up = vec3(0,0,1);

	Camera cam = getCamData(e->pos, e->rot);

	bool rightLock = !freeForm || input->keysDown[KEYCODE_CTRL];
	if(rightLock) cam.look = cross(up, cam.right);

	Vec3 acceleration = vec3(0,0,0);
	if(input->keysDown[KEYCODE_SHIFT]) speed *= boost;
	if(input->keysDown[KEYCODE_W]) acceleration +=  normVec3(cam.look);
	if(input->keysDown[KEYCODE_S]) acceleration += -normVec3(cam.look);
	if(input->keysDown[KEYCODE_D]) acceleration +=  normVec3(cam.right);
	if(input->keysDown[KEYCODE_A]) acceleration += -normVec3(cam.right);

	if(freeForm) {
		if(input->keysDown[KEYCODE_E]) acceleration +=  normVec3(up);
		if(input->keysDown[KEYCODE_Q]) acceleration += -normVec3(up);
	}

	if(acceleration != vec3(0,0,0)) {
		e->acc += normVec3(acceleration)*speed;
	}
}

//

void particleEmitterUpdate(ParticleEmitter* e, float dt, float dtTime) {
	for(int i = 0; i < e->particleListCount; i++) {
		Particle* p = e->particleList + i;

		p->vel = p->vel + p->acc*dt;
		// p->vel = p->vel * pow(0.01f,dt);
		p->pos = p->pos - 0.5f*p->acc*dt*dt + p->vel*dt;

		p->velColor = p->velColor + p->accColor*dt;
		// // // p->velColor = p->velColor * pow(friction,dt);
		p->color = p->color - 0.5f*p->accColor*dt*dt + p->velColor*dt;

		p->velSize = p->velSize + p->accSize*dt;
		// // // p->velColor = p->velColor * pow(friction,dt);
		p->size = p->size - 0.5f*p->accSize*dt*dt + p->velSize*dt;

		// p->velRot = p->velRot + p->accRot*dt;
		// // p->velColor = p->velColor * pow(friction,dt);
		// p->rot = p->rot - 0.5f*p->accRot*dt*dt + p->velRot*dt;

		p->time += dtTime;
	}
}

void particleEmitterFinish(ParticleEmitter* e) {

	// Remove dead.
	for(int i = 0; i < e->particleListCount; i++) {
		Particle* p = e->particleList + i;

		if(p->time >= p->timeToLive) {
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



// for(int i = 0; i < emitter->particleListCount; i++) {
// 	Particle* p = emitter->particleList + i;

// 	Vec3 pos = p->pos;
// 	Vec3 size = p->size;

// 	float friction = 0.01f;

// 	Vec3 collisionNormal;
// 	bool result = collisionVoxelBoxDistance(&ad->voxelData, pos, size, e->chunk, &pos, &collisionNormal);

// 	bool groundRaycast = raycastGroundVoxelBox(&ad->voxelData, pos, size, e->chunk, 0);

// 	if(collisionNormal.xy != vec2(0,0) || groundRaycast) {
// 		p->vel.x *= pow(friction,dt);
// 		p->vel.y *= pow(friction,dt);
// 	} 

// 	if(collisionNormal.xy != vec2(0,0)) {
// 		p->vel = reflectVector(p->vel, collisionNormal);
// 		p->vel *= 0.5f;
// 	}

// 	if(collisionNormal.z != 0) {
// 		p->vel.z = 0;
// 	}

// 	p->pos = pos;
// }

// {
// 	float friction = 0.01f;

// 	Vec3 collisionNormal;
// 	Vec3 pos;
// 	bool result = collisionVoxelBoxDistance(&ad->voxelData, e->pos, e->dim, e->chunk, &pos, &collisionNormal);

// 	bool groundRaycast = raycastGroundVoxelBox(&ad->voxelData, pos, e->dim, e->chunk, 0);

// 	if(collisionNormal.xy != vec2(0,0) || groundRaycast) {
// 		e->vel.x *= pow(friction,dt);
// 		e->vel.y *= pow(friction,dt);
// 	} 

// 	if(collisionNormal.xy != vec2(0,0)) {
// 		e->vel = reflectVector(e->vel, collisionNormal);
// 		e->vel *= 0.5f;
// 	}

// 	if(collisionNormal.z != 0) {
// 		e->vel.z = 0;
// 	}

// 	e->pos = pos;
// }
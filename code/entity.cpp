
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

	// void* data;
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

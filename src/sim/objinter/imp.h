struct ObjInter;

struct Config;
struct Opt;
struct PFarrays;

void obj_inter_ini(const Config*, const Opt*, MPI_Comm, float dt, int maxp, /**/ ObjInter**);
void obj_inter_fin(ObjInter*);

void obj_inter_forces(ObjInter*, PFarrays *flu, int *flu_start, PFarrays *obj);

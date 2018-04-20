struct ObjInter;

struct Config;
struct Opt;
struct PFarrays;

// tag::mem[]
void obj_inter_ini(const Config*, const Opt*, MPI_Comm, float dt, int maxp, /**/ ObjInter**);
void obj_inter_fin(ObjInter*);
// end::mem[]

// tag::int[]
void obj_inter_forces(ObjInter*, PFarrays *flu, int *flu_start, PFarrays *obj);
// end::int[]

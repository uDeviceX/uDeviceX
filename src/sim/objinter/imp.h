struct ObjInter;

struct Config;
struct Opt;
struct PFarray;
struct PFarrays;

// tag::mem[]
void obj_inter_ini(const Config*, const Opt*, MPI_Comm, float dt, int maxp, /**/ ObjInter**);
void obj_inter_fin(ObjInter*);
// end::mem[]

// tag::int[]
void obj_inter_update_dpd_prms(float dt, float kBT, ObjInter*); // <1>
void obj_inter_forces(ObjInter*, const PairParams **fsi_prms, const PairParams **cnt_prms, PFarray *flu, const int *flu_start, PFarrays *obj); // <2>
// end::int[]

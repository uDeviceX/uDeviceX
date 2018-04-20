struct ObjInter;

struct Opt;
struct PFAarrays;

void obj_inter_ini(MPI_Comm, int maxp, const Opt*, /**/ ObjInter**);
void obj_inter_fin(ObjInter*);

void obj_inter_forces(ObjInter*, PFarrays *flu, int *flu_start, PFarrays *obj);

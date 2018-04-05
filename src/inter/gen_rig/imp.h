struct RigPinInfo;

// tag::helpers[]
struct RigGenInfo {
    float mass;
    int numdensity;
    const RigPinInfo *pi;
    int nt, nv;
    const int4 *tt;
    const float *vv;
    bool empty_pp;
};

struct FluInfo {
    Particle *pp;
    int *n;
};

struct RigInfo {
    int *ns, *nps, *n;
    float *rr0;
    Solid *ss;
    Particle *pp;
};
// end::helpers[]

// tag::int[]
void inter_gen_rig_from_solvent(const Coords*, MPI_Comm, RigGenInfo, /* io */ FluInfo, /* o */ RigInfo); // <1>
void inter_set_rig_ids(MPI_Comm, int n, /**/ Solid *ss); // <2>
// end::int[]

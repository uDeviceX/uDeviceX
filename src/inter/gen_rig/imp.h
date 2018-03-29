struct RigPinInfo;

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

namespace gen {
void gen_rig_from_solvent(const Coords *coords, MPI_Comm comm, RigGenInfo rgi,
                          /* io */ FluInfo fluinf, /* o */ RigInfo riginfo);

void set_rig_ids(MPI_Comm comm, int n, /**/ Solid * ss);
} // gen

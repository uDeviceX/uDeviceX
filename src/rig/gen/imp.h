struct Coords;
struct RigPinInfo;

struct RigGenInfo {
    float mass;
    int numdensity;
    const RigPinInfo *pi;
    int nt, nv;
    const int4 *tt;
    const Particle *pp;
    bool empty_pp;
};

struct FluInfo {
    Particle *pp;
    int *n;
};

struct RigInfo {
    int ns, *nps, *n;
    float *rr0;
    Solid *ss;
    Particle *pp;
};

void rig_gen_from_solvent(const Coords*, MPI_Comm, RigGenInfo, /*io*/ FluInfo, /**/ RigInfo);

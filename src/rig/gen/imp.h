struct Coords;
struct RigPinInfo;
struct MeshRead;

struct RigGenInfo {
    float mass;
    int numdensity;
    const RigPinInfo *pi;
    int nt, nv;
    const int4 *tt;     /* on device */
    const Particle *pp; /* on device */
    bool empty_pp;
    MeshRead *mesh;
};

struct FluInfo {
    Particle *pp;
    int *n;
};

/* everything on host */
struct RigInfo {
    int ns, *nps, *n;
    float *rr0;
    Solid *ss;
    Particle *pp;
};

void rig_gen_from_solvent(const Coords*, MPI_Comm, RigGenInfo, /*io*/ FluInfo, /**/ RigInfo);

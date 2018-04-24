struct Coords;
struct RigPinInfo;
struct Sdf;
struct MeshRead;

// tag::helpers[]
struct InterWalInfos {
    bool active;
    const Sdf *sdf;
};

struct InterFluInfos {
    FluQuants *q;
};

struct InterRbcInfos {
    bool active;
    RbcQuants *q;
};

struct InterRigInfos {
    bool active;
    bool empty_pp;
    RigQuants *q;
    const RigPinInfo *pi;
    const MeshRead *mesh;
    float mass;
    int numdensity;
};
// end::helpers[]

// tag::int[]
void inter_freeze(const Coords*, MPI_Comm, InterWalInfos, InterFluInfos, InterRbcInfos, InterRigInfos); // <1>
// end::int[]

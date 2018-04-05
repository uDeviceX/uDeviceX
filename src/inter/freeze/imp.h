struct Coords;
struct RigPinInfo;

// tag::helpers[]
struct InterWalInfos {
    bool active;
    Sdf *sdf;
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
    float mass;
    int numdensity;
};
// end::helpers[]

// tag::int[]
void inter_freeze(const Coords*, MPI_Comm, InterWalInfos, InterFluInfos, InterRbcInfos, InterRigInfos); // <1>
void inter_freeze_walls(MPI_Comm, int maxn, Sdf*, FluQuants*, WallQuants*); // <2>
// end::int[]

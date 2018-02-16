struct Coords;
struct RigPinInfo;

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
    RigQuants *q;
    const RigPinInfo *pi;
};

void inter_freeze(const Coords *coords, MPI_Comm cart, InterWalInfos, InterFluInfos, InterRbcInfos, InterRigInfos);
void inter_create_walls(MPI_Comm cart, int maxn, Sdf*, FluQuants*, WallQuants*);

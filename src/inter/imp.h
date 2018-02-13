struct Coords;
struct RigPinInfo;

void inter_freeze(const Coords *coords, const RigPinInfo *pi, bool rigids, MPI_Comm cart, Sdf*, FluQuants*, RigQuants*, RbcQuants*);
void inter_create_walls(MPI_Comm cart, int maxn, Sdf*, FluQuants*, WallQuants*);

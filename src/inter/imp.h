struct Coords;
void inter_freeze(const Coords *coords, bool rigids, MPI_Comm cart, Sdf*, FluQuants*, RigQuants*, RbcQuants*);
void inter_create_walls(MPI_Comm cart, int maxn, Sdf*, FluQuants*, WallQuants*);

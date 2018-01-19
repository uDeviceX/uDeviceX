namespace inter {
void freeze(Coords coords, MPI_Comm cart, Sdf*, FluQuants*, rig::RigQuants*, RbcQuants*);
void create_walls(MPI_Comm cart, int maxn, Sdf*, FluQuants*, wall::Quants*);
}

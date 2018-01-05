namespace inter {
void freeze(Coords coords, MPI_Comm cart, Sdf*, flu::Quants*, rig::Quants*, rbc::Quants*);
void create_walls(MPI_Comm cart, int maxn, Sdf*, flu::Quants*, wall::Quants*);
}

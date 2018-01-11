namespace inter {
void freeze(Coords coords, MPI_Comm cart, Sdf*, flu::FluQuants*, rig::Quants*, rbc::Quants*);
void create_walls(MPI_Comm cart, int maxn, Sdf*, flu::FluQuants*, wall::Quants*);
}

namespace inter {
void freeze(MPI_Comm cart, Sdf*, flu::Quants*, rig::Quants*, rbc::Quants*);
void create_walls(int maxn, Sdf*, flu::Quants*, wall::Quants*);
}

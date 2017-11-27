namespace inter {
void freeze(MPI_Comm cart, sdf::Quants, flu::Quants*, rig::Quants*, rbc::Quants*);
void create_walls(int maxn, sdf::Quants, flu::Quants*, wall::Quants*);
}

struct Coords;
namespace inter {
void freeze(const Coords *coords, MPI_Comm cart, Sdf*, FluQuants*, RigQuants*, RbcQuants*);
void create_walls(MPI_Comm cart, int maxn, Sdf*, FluQuants*, WallQuants*);
}

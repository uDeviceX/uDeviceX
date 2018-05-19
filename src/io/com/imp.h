struct Coords;
struct float3;
void io_com_dump(MPI_Comm, const Coords*, const char *name, long id, int n, const int *ii, const float3 *rr);

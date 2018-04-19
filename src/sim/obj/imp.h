struct Objects;
struct Config;

void objects_ini(const Config*, const Opt*, MPI_Comm, int maxp, int3 L, Objects**);
void objects_fin(const Opt*, Objects*);

void objects_update(Objects*);
void objects_distribute(Objects*);
void objects_dump(Objects*);


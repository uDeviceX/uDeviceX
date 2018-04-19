struct Objects;

struct Config;
struct Coords;
struct Opt;

void objects_ini(const Config*, const Opt*, MPI_Comm, const Coords*, int maxp, Objects**);
void objects_fin(Objects*);

void objects_clear_forces(Objects*);
void objects_update(float dt, Objects*);
void objects_distribute(Objects*);

void objects_mesh_dump(Objects*);

void objects_strt_templ(const char *base, Objects*);
void objects_strt_dump(const char *base, long id, Objects*);


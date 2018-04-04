struct Coords;
struct Particle;
struct Solid;

enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};

void restart_write_pp(MPI_Comm, const char *code, int id, long n, const Particle *pp);
void restart_write_ii(MPI_Comm, const char *code, int id, long n, const int *ii);
void restart_write_ss(MPI_Comm, const char *code, int id, long n, const Solid *ss);

void restart_read_pp(MPI_Comm, const char *code, int id, int *n, Particle *pp);
void restart_read_ii(MPI_Comm, const char *code, int id, int *n, int *ii);
void restart_read_ss(MPI_Comm, const char *code, int id, int *n, Solid *ss);

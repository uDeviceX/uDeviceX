struct Coords;
struct Particle;
struct Solid;

namespace restart {

enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};

void restart_write_pp(const Coords *coords, const char *code, const int id, const Particle *pp, const long n);
void restart_read_pp(const Coords *coords, const char *code, const int id, Particle *pp, int *n);

void restart_write_ii(const Coords *coords, const char *code, const char *subext, const int id, const int *ii, const long n);
void restart_read_ii(const Coords *coords, const char *code, const char *subext, const int id, int *ii, int *n);

void restart_write_ss(const Coords *coords, const char *code, const int id, const Solid *ss, const long n);
void restart_read_ss(const Coords *coords, const char *code, const int id, Solid *ss, int *n);
}

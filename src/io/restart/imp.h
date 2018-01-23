struct Coords;

namespace restart {

enum {TEMPL=-1,
      FINAL=-2};
enum {BEGIN=FINAL};

void write_pp(const Coords *coords, const char *code, const int id, const Particle *pp, const long n);
void read_pp(const Coords *coords, const char *code, const int id, Particle *pp, int *n);

void write_ii(const Coords *coords, const char *code, const char *subext, const int id, const int *ii, const long n);
void read_ii(const Coords *coords, const char *code, const char *subext, const int id, int *ii, int *n);

void write_ss(const Coords *coords, const char *code, const int id, const Solid *ss, const long n);
void read_ss(const Coords *coords, const char *code, const int id, Solid *ss, int *n);
}

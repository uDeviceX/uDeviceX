namespace restart {

enum {TEMPL=-1,
      FINAL=-2};
enum {BEGIN=FINAL};

void write_pp(const char *code, const int id, const Particle *pp, const long n);
void read_pp(const char *code, const int id, Particle *pp, int *n);

void write_ii(const char *code, const int id, const int *pp, const long n);
void read_ii(const char *code, const int id, int *pp, int *n);

void write_ss(const char *code, const int id, const Solid *ss, const long n);
void read_ss(const char *code, const int id, Solid *ss, int *n);
}

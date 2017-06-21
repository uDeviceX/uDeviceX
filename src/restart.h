namespace restart {
void write_pp(const char *code, const int id, const Particle *pp, const long n);
void read_pp(const char *code, const int id, Particle *pp, int *n);

void write_ss(const char *code, const int id, const Solid *ss, const long n);
void read_ss(const char *code, const int id, Solid *ss, int *n);
}

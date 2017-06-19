namespace restart {
void write(const char *code, const int id, const Particle *pp, const long n);
void read(const char *code, const int id, Particle *pp, int *n);

void write(const char *code, const int id, const Solid *ss, const long n);
void read(const char *code, const int id, Solid *ss, int *n);
}

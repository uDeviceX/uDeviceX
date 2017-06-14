namespace restart {

void write(const char *fname, const Particle *pp, const int  n);
void read (const char *fname,       Particle *pp,       int *n);

void write(const char *fname, const Solid *ss, const int  n);
void read (const char *fname,       Solid *ss,       int *n);

}

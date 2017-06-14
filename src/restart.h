namespace restart {

void write(const char *basename, const Particle *pp, const long n, const int step);
void read (const char *basename,       Particle *pp,       int *n);

void write(const char *basename, const Solid *ss, const int  n);
void read (const char *basename,       Solid *ss,       int *n);

}

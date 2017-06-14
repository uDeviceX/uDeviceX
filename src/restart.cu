#include "common.h"
#include "m.h"

#include "restart.h"

namespace restart {
enum {X, Y, Z};

// pattern : basename.rank-step
#define PATTERN "%s.%04d-%05d"
// buff size
#define BS (256)

namespace bopwrite {
void header(const char *name, const long n, const int step) {
    char fname[BS] = {0};
    sprintf(fname, "restart/" PATTERN ".bop", name, m::rank, step);
        
    FILE *f = fopen(fname, "w");

    if (f == NULL)
    ERR("could not open <%s>\n", fname);

    fprintf(f, "%ld\n", n);
    fprintf(f, "DATA_FILE: " PATTERN ".values\n", name, m::rank, step);
    fprintf(f, "DATA_FORMAT: float\n");
    fprintf(f, "VARIABLES: x y z vx vy vz\n");
    fclose(f);
}

void data(const char *name, const Particle *pp, const long n, const int step) {
    char fname[BS] = {0};
    sprintf(fname, "restart/" PATTERN ".values", name, m::rank, step);

    FILE *f = fopen(fname, "w");
    fwrite((float *) pp, sizeof(float), sizeof(Particle)/sizeof(float) * n, f);
    fclose(f);
}
} // namespace bopwrite

namespace bopread {
void header(const char *name, long *n) {
    FILE *f = fopen(name, "r");
    if (fscanf(f, "%ld\n", n) != 1) ERR("wrong format\n");
    fclose(f);
}

void data(const char *name, const long n, Particle *pp) {
    FILE *f = fopen(name, "r");
    fread(pp, sizeof(Particle), n, f);
    fclose(f);
}
} // namespace bopread

void write(const char *basename, const Particle *pp, const long n, const int step) {
    bopwrite::header(basename, n, step);
    bopwrite::data(basename, pp, n, step);
}

void read (const char *basename, Particle *pp, int *n) {
    long np = 0;
    char bop[BS] = {0}, val[BS] = {0};
    sprintf(bop, "%s.bop", basename);
    sprintf(val, "%s.values", basename);
    
    bopread::header(bop, &np);
    bopread::data(val, np, pp);
    *n = np;
}

void write(const char *basename, const Solid *ss, const long n, const int step) {
    char fname[BS] = {0};
    sprintf(fname, "restart/" PATTERN ".solid", basename, m::rank, step);

    FILE *f = fopen(fname, "r");
    fprintf(f, "%ld\n", n);
    fwrite(ss, sizeof(Solid), n, f);
    fclose(f);
}

void read (const char *basename, Solid *ss, int *n, const int step) {
    long ns = 0;
    char fname[BS] = {0};
    sprintf(fname, "restart/" PATTERN ".solid", basename, m::rank, step);
    
    FILE *f = fopen(fname, "r");
    fscanf(f, "%ld\n", &ns);
    fread(ss, sizeof(Solid), ns, f);
    fclose(f);

    *n = ns;
}

} // namespace restart

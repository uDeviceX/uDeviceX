#include "common.h"
#include "m.h"

#include "restart.h"

namespace restart {
enum X, Y, Z};

// pattern : basename.rank-step
#define PATTERN "%s.%04d-%05d"

namespace write {
static void header(const long n, const char *name, const int step) {
    char fname[256] = {0};
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
} // namespace write

void write(const char *basename, const Particle *pp, const int n) {
    
}

void read (const char *basename, Particle *pp, int *n) {
    
}

void write(const char *fname, const Solid *ss, const int  n) {

}

void read (const char *fname, Solid *ss, int *n) {

}

} // namespace restart

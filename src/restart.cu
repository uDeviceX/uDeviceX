#include <mpi.h>
#include "common.h"
#include <conf.h>
#include "conf.common.h"
#include "m.h"

#include "restart.h"

#define DBG(...) MSG(__VA_ARGS__)

namespace restart {
enum {X, Y, Z};

/* pattern : 
   sing processor  : base/strt/code/id.ext
   mult processors : base/strt/code/XXX.YYY.ZZZ/id.ext
   base depends on read/write
 */
#define PF    "%s.%s"
#define DIR_S "%s/strt/%s/"                 PF
#define DIR_M "%s/strt/%s/%03d.%03d.%03d/"  PF

#define READ (true)
#define DUMP (false)

// buff size
#define BS (256)

// check fprintf (BS-1 for \0 character)
#define CSPR(a) do {                                        \
        int check = a;                                      \
        if (check < 0 || check >= BS-1)                     \
        ERR("Buffer too small to handle this format\n");    \
    } while (0)

#define CF(f, fname) do {if (f==NULL) ERR("could not open the file <%s>\n", fname);} while(0)

void id2str(const int id, char *str) {
    switch (id) {
    case TEMPL:
        CSPR(sprintf(str, "templ"));
        break;
    case FINAL:
        CSPR(sprintf(str, "final"));
        break;
    default:
        CSPR(sprintf(str, "%05d", id));
        break;
    }
}

void gen_name(const bool read, const char *code, const int id, const char *ext, /**/ char *name) {
    char idcode[BS] = {0};
    id2str(id, /**/ idcode);
    
    if (m::size == 1)
    CSPR(sprintf(name, DIR_S, read ? BASE_STRT_READ : BASE_STRT_DUMP, code, idcode, ext));
    else
    CSPR(sprintf(name, DIR_M, read ? BASE_STRT_READ : BASE_STRT_DUMP, code, m::coords[X], m::coords[Y], m::coords[Z], idcode, ext));
}

namespace bopwrite {
void header(const char *bop, const char *rel, const long n) {
    FILE *f = fopen(bop, "w"); CF(f, bop);
    
    fprintf(f, "%ld\n", n);
    fprintf(f, "DATA_FILE: %s\n", rel);
    fprintf(f, "DATA_FORMAT: float\n");
    fprintf(f, "VARIABLES: x y z vx vy vz\n");
    fclose(f);
}

void data(const char *val, const Particle *pp, const long n) {
    FILE *f = fopen(val, "w"); CF(f, val);
    fwrite((float *) pp, sizeof(float), sizeof(Particle)/sizeof(float) * n, f);
    fclose(f);
}
} // namespace bopwrite

namespace bopread {
void header(const char *name, long *n) {
    FILE *f = fopen(name, "r"); CF(f, name);
    if (fscanf(f, "%ld\n", n) != 1) ERR("wrong format\n");
    fclose(f);
}

void data(const char *name, const long n, Particle *pp) {
    FILE *f = fopen(name, "r"); CF(f, name);
    fread(pp, sizeof(Particle), n, f);
    fclose(f);
}
} // namespace bopread

void write_pp(const char *code, const int id, const Particle *pp, const long n) {
    char bop[BS] = {0}, rel[BS] = {0}, val[BS] = {0}, idcode[BS] = {0};
    gen_name(DUMP, code, id, "bop"   , /**/ bop);
    gen_name(DUMP, code, id, "values", /**/ val);

    id2str(id, /**/ idcode);
    CSPR(sprintf(rel, PF, idcode, "values"));    

    bopwrite::header(bop, rel, n);
    bopwrite::data(val, pp, n);
}

void read_pp(const char *code, const int id, Particle *pp, int *n) {
    long np = 0;
    char bop[BS] = {0}, val[BS] = {0};
    gen_name(READ, code, id, "bop"   , /**/ bop);
    gen_name(READ, code, id, "values", /**/ val);
    DBG("reading <%s> and <%s>", bop, val);
    bopread::header(bop, &np);
    bopread::data(val, np, pp);
    *n = np;
    DBG("I have read %ld pp", np);
}

void write_ss(const char *code, const int id, const Solid *ss, const long n) {
    char fname[BS] = {0};
    gen_name(DUMP, code, id, "solid", /**/ fname);
        
    FILE *f = fopen(fname, "w"); CF(f, fname);
    fprintf(f, "%ld\n", n);
    fwrite(ss, sizeof(Solid), n, f);
    fclose(f);
}

void read_ss(const char *code, const int id, Solid *ss, int *n) {
    long ns = 0;
    char fname[BS] = {0};
    gen_name(READ, code, id, "solid", /**/ fname);
    fprintf(stderr, "reading %s\n", fname);
    FILE *f = fopen(fname, "r"); CF(f, fname);
    fscanf(f, "%ld\n", &ns);
    fread(ss, sizeof(Solid), ns, f);
    fclose(f);
    *n = ns;
    DBG("I have read %ld ss.", ns);
}

#undef READ
#undef DUMP

} // namespace restart

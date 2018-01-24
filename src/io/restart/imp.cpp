#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/type.h"
#include "inc/def.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "utils/error.h"
#include "utils/imp.h"
#include "coords/imp.h"

#include "imp.h"

//#define DBG(...) msg_print(__VA_ARGS__)
#define DBG(...) 

enum {X, Y, Z};

/* pattern : 
   sing processor  : base/code/id.ext
   mult processors : base/code/XXX.YYY.ZZZ/id.ext
   base depends on read/write
 */
#define PF    "%s.%s"
#define DIR_S "%s/%s/"     PF
#define DIR_M "%s/%s/%s/"  PF

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

static void id2str(const int id, char *str) {
    switch (id) {
    case RESTART_TEMPL:
        CSPR(sprintf(str, "templ"));
        break;
    case RESTART_FINAL:
        CSPR(sprintf(str, "final"));
        break;
    default:
        CSPR(sprintf(str, "%05d", id));
        break;
    }
}

static void gen_name(const Coords *coords, const bool read, const char *code, const int id, const char *ext, /**/ char *name) {
    char idcode[BS] = {0};
    id2str(id, /**/ idcode);
    int size;
    size = coords_size(coords);
    if (size == 1) {
        CSPR(sprintf(name, DIR_S, read ? BASE_STRT_READ : BASE_STRT_DUMP, code, idcode, ext));
    }
    else {
        char stamp[FILENAME_MAX];
        coord_stamp(coords, stamp);
        CSPR(sprintf(name, DIR_M, read ? BASE_STRT_READ : BASE_STRT_DUMP, code, stamp, idcode, ext));
    }
}

static void write_header_pp(const char *bop, const char *rel, const long n) {
    FILE *f;
    UC(efopen(bop, "w", /**/ &f));
    
    fprintf(f, "%ld\n", n);
    fprintf(f, "DATA_FILE: %s\n", rel);
    fprintf(f, "DATA_FORMAT: float\n");
    fprintf(f, "VARIABLES: x y z vx vy vz\n");

    UC(efclose(f));
}

static void write_header_ii(const char *bop, const char *rel, const long n) {
    FILE *f;
    UC(efopen(bop, "w", /**/ &f));
    
    fprintf(f, "%ld\n", n);
    fprintf(f, "DATA_FILE: %s\n", rel);
    fprintf(f, "DATA_FORMAT: int\n");
    fprintf(f, "VARIABLES: id\n");
    UC(efclose(f));
}

template <typename T>
static void write_data(const char *val, const T *dat, const long n) {
    FILE *f;
    UC(efopen(val, "w", /**/ &f));
    UC(efwrite(dat, sizeof(T), n, f));
    UC(efclose(f));
}

static void read_n(const char *name, long *n) {
    FILE *f;
    UC(efopen(name, "r", /**/ &f));
    if (fscanf(f, "%ld\n", n) != 1) ERR("wrong format\n");
    UC(efclose(f));
}

template <typename T>
static void read_data(const char *name, const long n, T *dat) {
    FILE *f;
    UC(efopen(name, "r", /**/ &f));
    UC(efread(dat, sizeof(T), n, f));
    UC(efclose(f));
}

void restart_write_pp(const Coords *coords, const char *code, const int id, const Particle *pp, const long n) {
    char bop[BS] = {0}, rel[BS] = {0}, val[BS] = {0}, idcode[BS] = {0};
    gen_name(coords, DUMP, code, id, "bop"   , /**/ bop);
    gen_name(coords, DUMP, code, id, "values", /**/ val);

    id2str(id, /**/ idcode);
    CSPR(sprintf(rel, PF, idcode, "values"));    

    write_header_pp(bop, rel, n);
    write_data(val, pp, n);
}

void restart_read_pp(const Coords *coords, const char *code, const int id, Particle *pp, int *n) {
    long np = 0;
    char bop[BS] = {0}, val[BS] = {0};
    gen_name(coords, READ, code, id, "bop"   , /**/ bop);
    gen_name(coords, READ, code, id, "values", /**/ val);
    msg_print("reading <%s> and <%s>", bop, val);
    read_n(bop, &np);
    read_data(val, np, pp);
    *n = np;
    DBG("I have read %ld pp", np);
}

void restart_write_ii(const Coords *coords, const char *code, const char *subext, const int id, const int *ii, const long n) {
    char bop[BS] = {0}, rel[BS] = {0}, val[BS] = {0}, idcode[BS] = {0},
        extbop[BS] = {0}, extval[BS] = {0};
    CSPR(sprintf(extbop, "%s.bop",    subext));
    CSPR(sprintf(extval, "%s.values", subext));
    
    gen_name(coords, DUMP, code, id, extbop, /**/ bop);
    gen_name(coords, DUMP, code, id, extval, /**/ val);

    id2str(id, /**/ idcode);
    CSPR(sprintf(rel, PF, idcode, "id.values"));    

    write_header_ii(bop, rel, n);
    write_data(val, ii, n);
}

void restart_read_ii(const Coords *coords, const char *code, const char *subext, const int id, int *ii, int *n) {
    long np = 0;
    char bop[BS] = {0}, val[BS] = {0}, extbop[BS] = {0}, extval[BS] = {0};
    CSPR(sprintf(extbop, "%s.bop",    subext));
    CSPR(sprintf(extval, "%s.values", subext));
    
    gen_name(coords, READ, code, id, extbop, /**/ bop);
    gen_name(coords, READ, code, id, extval, /**/ val);
    DBG("reading <%s> and <%s>", bop, val);
    read_n(bop, &np);
    read_data(val, np, ii);
    *n = np;
    DBG("I have read %ld pp", np);
}

void restart_write_ss(const Coords *coords, const char *code, const int id, const Solid *ss, const long n) {
    char fname[BS] = {0};
    FILE *f;
    gen_name(coords, DUMP, code, id, "solid", /**/ fname);
        
    UC(efopen(fname, "w", /**/ &f));
    fprintf(f, "%ld\n", n);
    UC(efwrite(ss, sizeof(Solid), n, f));
    UC(efclose(f));
}

void restart_read_ss(const Coords *coords, const char *code, const int id, Solid *ss, int *n) {
    long ns = 0;
    char fname[BS] = {0};
    FILE *f;
    gen_name(coords, READ, code, id, "solid", /**/ fname);
    fprintf(stderr, "reading %s\n", fname);

    UC(efopen(fname, "r", /**/ &f));
    fscanf(f, "%ld\n", &ns);
    UC(efread(ss, sizeof(Solid), ns, f));
    UC(efclose(f));
    *n = ns;
    DBG("I have read %ld ss.", ns);
}

#undef PF
#undef DIR_S
#undef DIR_M

#undef READ
#undef DUMP

#undef DBG

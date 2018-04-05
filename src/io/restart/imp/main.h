/* 
   pattern : 
   base/code/id.ext
   base depends on read/write
*/

#define PATTERN "%s/%s/%s"

// buff size
enum { BS = 1024 };

// check fprintf (BS-1 for \0 character)
#define CSPR(a) do {                                        \
        int check = a;                                      \
        if (check < 0 || check >= BS-1)                     \
        ERR("Buffer too small to handle this format\n");    \
    } while (0)

static void id2code(const int id, char *code) {
    switch (id) {
    case RESTART_TEMPL:
        CSPR(sprintf(code, "templ"));
        break;
    case RESTART_FINAL:
        CSPR(sprintf(code, "final"));
        break;
    default:
        CSPR(sprintf(code, "%05d", id));
        break;
    }
}

static void gen_base_name_dump(const char *base, const char *code, const int id, /**/ char *name) {
    char idcode[BS] = {0};
    id2code(id, /**/ idcode);
    CSPR(sprintf(name, PATTERN, base, code, idcode));
}

static void gen_base_name_read(const char *base, const char *code, const int id, /**/ char *name) {
    char idcode[BS] = {0};
    id2code(id, /**/ idcode);
    CSPR(sprintf(name, PATTERN ".bop", base, code, idcode));
}

template <typename T>
static void write(MPI_Comm comm, const char *base, const char *code, int id, long n, const T *tt, BopType type, int nvars, const char *vars) {
    char name[BS];
    BopData *bop;
    T *tt_bop;
    gen_base_name_dump(base, code, id, /**/ name);

    BPC(bop_ini(&bop));
    BPC(bop_set_n(n, bop));
    BPC(bop_set_type(type, bop));
    BPC(bop_set_vars(nvars, vars, bop));
    BPC(bop_alloc(bop));

    tt_bop = (T*) bop_get_data(bop);
    memcpy(tt_bop, tt, n * sizeof(T));

    BPC(bop_write_header(comm, name, bop));
    BPC(bop_write_values(comm, name, bop));

    BPC(bop_fin(bop));
}

template <typename T>
static void read(MPI_Comm comm, const char *base, const char *code, const int id, int *n, T *tt) {
    long np = 0;
    char hname[BS], dname[BS];
    BopData *bop;
    const T *tt_bop;
    gen_base_name_read(base, code, id, /**/ hname);
    msg_print("reading '%s'", hname);

    BPC(bop_ini(&bop));
    BPC(bop_read_header(comm, hname, bop, dname));
    BPC(bop_alloc(bop));
    BPC(bop_read_values(comm, dname, bop));

    BPC(bop_get_n(bop, &np));
    tt_bop = (const T*) bop_get_data(bop);
    memcpy(tt, tt_bop, np * sizeof(T));
    *n = np;    

    BPC(bop_fin(bop));    
}

void restart_write_pp(MPI_Comm comm, const char *base, const char *code, int id, long n, const Particle *pp) {
    write(comm, base, code, id, n, pp, BopFLOAT, 6, "x y z vx vy vz");
}

void restart_read_pp(MPI_Comm comm, const char *base, const char *code, int id, int *n, Particle *pp) {
    read(comm, base, code, id, n, pp);
}

void restart_write_ii(MPI_Comm comm, const char *base, const char *code, int id, long n, const int *ii) {
    write(comm, base, code, id, n, ii, BopINT, 1, "i");
}

void restart_read_ii(MPI_Comm comm, const char *base, const char *code, int id, int *n, int *ii) {
    read(comm, base, code, id, n, ii);    
}

void restart_write_ss(MPI_Comm comm, const char *code, int id, long n, const Solid *ss) {
    static const char *vars =
        "Ixx Ixy Ixz Iyy Iyz Izz "
        "mass x y z vx vy vz omx omy omz "
        "e0x e0y e0z e1x e1y e1z e2x e2y e2z "
        "fx fy fz tx ty tz i";
    enum {NVARS=32};
    write(comm, BASE_STRT_DUMP, code, id, n, ss, BopFLOAT, NVARS, vars);
}

void restart_read_ss(MPI_Comm comm, const char *code, int id, int *n, Solid *ss) {
    read(comm, BASE_STRT_READ, code, id, n, ss);
}

#undef PATTERN

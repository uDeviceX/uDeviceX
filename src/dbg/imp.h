struct Dbg;

struct Config;
struct Force;
struct Particle;
struct int3;
struct Coords;

// tag::kind[]
enum {
    DBG_POS,      // <1>
    DBG_POS_SOFT, // <2>
    DBG_VEL,      // <3>
    DBG_FORCES,   // <4>
    DBG_COLORS,   // <5>
    DBG_CLIST,    // <6>
    DBG_NKIND_
};
// end::kind[]

// tag::mem[]
void dbg_ini(Dbg**);
void dbg_fin(Dbg*);
// end::mem[]

// tag::ini[]
void dbg_enable(int kind, Dbg *dbg);
void dbg_disable(int kind, Dbg *dbg);
void dbg_set_verbose(bool, Dbg *dbg);
void dbg_set_dump(bool, Dbg *dbg);
// end::ini[]

// tag::cnf[]
void dbg_set_conf(const Config*, Dbg*);
// end::cnf[]

// tag::int[]
void dbg_check_pos(Coords c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_pos_soft(Coords c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_vel(Coords c, const char *base, const Dbg *dbg, int n, const Particle *pp);
void dbg_check_forces(Coords c, const Dbg *dbg, int n, const Force *ff);
void dbg_check_colors(Coords c, const Dbg *dbg, int n, const int *cc);
void dbg_check_clist(Coords c, const Dbg *dbg, int3 L, const int *starts, const int *counts, int n, const Particle *pp);
// end::int[]

struct Dbg;

struct Config;
struct Force;
struct Particle;
struct int3;

enum {
    DBG_POS,
    DBG_POS_SOFT,
    DBG_VEL,
    DBG_FORCES,
    DBG_COLORS,
    DBG_CLIST
};

void dbg_ini(Dbg**);
void dbg_fin(Dbg*);
void dbg_set(int kind, Dbg*);
void dbg_set_conf(const Config*, Dbg*);

void dbg_check_pos(int n, const Particle *pp);
void dbg_check_pos_soft(int n, const Particle *pp);
void dbg_check_vel(int n, const Particle *pp);
void dbg_check_forces(int n, const Force *ff);
void dbg_check_colors(int n, const int *ff);
void dbg_check_clist(int3 L, const int *starts, const int *counts, int n, const Particle *pp);

namespace dbg {
/* check if particles are inside domain (size L) */
void check_pos(const Particle *pp, int n, const char *file, int line, const char *M);

/* check positions of particles after a [p]osition [u]pdate (e.g. bounce-back, update) */
void check_pos_pu(const Particle *pp, int n, const char *file, int line, const char *M);

/* check if forces are within bounds */
void check_ff(const Force *ff, int n, const char *file, int line, const char *M);

/* check if velocities are sane */
void check_vv(const Particle *pp, int n, const char *file, int line, const char *M);

/* check if colors exist */
void check_cc(const int *cc, int n, const char *file, int line, const char *M);

/* check if particles are in correct cells */
void check_cells(int nx, int ny, int nz, const int *ss, const int *cc, const Particle *pp,
                 const char *file, int line, const char *M);
} // dbg

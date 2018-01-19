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
    DBG_CLIST,
    DBG_NKIND_
};

void dbg_ini(Dbg**);
void dbg_fin(Dbg*);
void dbg_set(int kind, Dbg*);
void dbg_set_conf(const Config*, Dbg*);

void dbg_check_pos(const Dbg *dbg, int n, const Particle *pp);
void dbg_check_pos_soft(const Dbg *dbg, int n, const Particle *pp);
void dbg_check_vel(const Dbg *dbg, int n, const Particle *pp);
void dbg_check_forces(const Dbg *dbg, int n, const Force *ff);
void dbg_check_colors(const Dbg *dbg, int n, const int *cc);
void dbg_check_clist(const Dbg *dbg, int3 L, const int *starts, const int *counts, int n, const Particle *pp);

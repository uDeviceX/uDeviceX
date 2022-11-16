#define T HeRead

struct int4;

typedef struct T T;

int he_read_ini(const char *path, T**);

/* tri = [t0, t1, t2] [t0, t2, t2], ... */
int he_read_tri_ini(int nv, int nt, int *tri, T**);

/* tri = [tri[0].x tri[0].y tri[0].z], ... [tri[0].x tri[0].y tri[0].z] */
int he_read_int4_ini(int nv, int nt, const int4 *tri, T**);

int he_read_fin(T*);

int he_read_nv(T*);
int he_read_nt(T*);
int he_read_ne(T*);
int he_read_nh(T*);

int he_read_nxt(T*, int**);
int he_read_flp(T*, int**);
int he_read_ver(T*, int**);
int he_read_tri(T*, int**);
int he_read_edg(T*, int**);

int he_read_hdg_ver(T*, int**);
int he_read_hdg_edg(T*, int**);
int he_read_hdg_tri(T*, int**);

int he_read_info(T*, FILE*);

#undef T

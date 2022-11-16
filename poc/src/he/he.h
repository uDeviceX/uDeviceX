#define T He

typedef struct T T;
typedef struct HeRead HeRead;

int he_ini(HeRead*, T**);
int he_file_ini(const char *path, T**);
int he_fin(T*);

int he_nv(T*); /* number of vert, edg, edg, hdg */
int he_nt(T*);
int he_ne(T*);
int he_nh(T*);

int he_nxt(T*, int h); /* return half-edg */
int he_flp(T*, int h);

int he_ver(T*, int h); /* return ver, tri, or edg */
int he_tri(T*, int h);
int he_edg(T*, int h);

int he_hdg_ver(T*, int v); /* return half-edg */
int he_hdg_edg(T*, int e);
int he_hdg_tri(T*, int t);

int he_bnd(T*, int h); /* 1 for boundary */

#undef T

#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"

void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
void emalloc(size_t, void **);
void efree(void *);

void edg_ini(int md, int nv, int *hx);
void edg_set(int md, int i, int j, int val, int *hx, int *hy);
int edg_get(int md, int i, int j, const int *hx, const int *hy);
int edg_valid(int md, int i, int j, const int *hx);

struct Edg {
    int md, nv;
    int *hx, *hy;
};

void e_ini(int md, int nv, Edg **pq) {
    Edg *q;
    EMALLOC(1, &q);
    EMALLOC(md * nv, &q->hx);
    EMALLOC(md * nv, &q->hy);
    edg_ini(md, nv, q->hx);
    q->md = md;
    q->nv = nv;
    *pq = q;
}

void e_fin(Edg *q) {
    EFREE(q->hx);
    EFREE(q->hy);
    EFREE(q);
}

void e_set(Edg *q, int i, int j, int val) {
    UC(edg_set(q->md, i, j, val, q->hx, q->hy));
}

int e_get(Edg *q, int i, int j) {
    int ans;
    UC(ans = edg_get(q->md, i, j, q->hx, q->hy));
    return ans;
}

int e_valid(Edg *q, int i, int j) {
    return edg_valid(q->md, i, j, q->hx);
}

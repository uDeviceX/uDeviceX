#include <stdlib.h>
#include <stdio.h>

#include "utils/error.h"
#include "imp.h"

void edg_ini(int md, int nv, /**/ int *hx) {
    int n, i;
    n = md * nv;
    for (i = 0; i < n; i ++) hx[i] = -1;
}

void edg_set(int md, int f, int x, int y,  /**/ int *hx, int *hy) {
    int d, j;
    j = f*md; d = 0;
    while (hx[j] != -1) {
        j++; d++;
        if (d >= md) ERR("invalid edg_set call: %d/%d/%d/%d/%d", md, f, x, y, j);
    }
    hx[j] = x; hy[j] = y;
}

int edg_get(int md, int i, int x, const int *hx, const int *hy) { /* next */
    int d;
    i *= md; d = 0;
    while (hx[i] != x) {
        i++; d++;
        if (d >= md)
            ERR("invalid edg_get call: %d/%d/%d", md, i, x);
    }
    return hy[i];
}

int edg_valid(int md, int i, int x, const int *hx) { /* valid? */
    int d;
    i *= md; d = 0;
    while (hx[i] != x) {
        i++; d++;
        if (d >= md) return 0;
    }
    return 1;
}

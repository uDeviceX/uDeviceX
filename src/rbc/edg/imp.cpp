#include <assert.h>
#include <stdio.h>
#include "msg.h"

#include "imp.h"

namespace edg {
void ini(int md, int nv, /**/ int *hx) {
    int n, i;
    n = md * nv;
    for (i = 0; i < n; i ++) hx[i] = -1;
}

void set(int md, int f, int x, int y,  /**/ int *hx, int *hy) {
    int d, j;
    j = f*md; d = 0;
    while (hx[j] != -1) {
        j++; d++;
        if (d >= md) ERR("invalid edg::set call: %d/%d/%d/%d/%d", md, f, x, y, j);
    }
    hx[j] = x; hy[j] = y;
}

int get(int md, int i, int x, int *hx, int *hy) { /* next */
    int d;
    i *= md; d = 0;
    while (hx[i] != x) {
        i++; d++;
        if (d >= md) ERR("invalid edg::get call: %d/%d/%d", md, i, x);
    }
    return hy[i];
}
}

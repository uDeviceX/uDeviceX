#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/dev.h"

#include "d/api.h"
#include "utils/kl.h"
#include "utils/cc.h"
#include "utils/error.h"
#include "utils/imp.h"

#include "type.h"
#include "imp.h"
#include "dev/main.h"

static void check_frags(int nfrags) {
    if (nfrags > MAX_FRAGS)
        ERR("Too many fragments: %d/%d", nfrags, MAX_FRAGS);
}
static int get_stride(int nfrags)       {return nfrags + 1;}
static int get_size(int nw, int nfrags) {return (nw + 1) * get_stride(nfrags);}

void emap_ini(int nw, int nfrags, int cap[], /**/ EMap *map) {
    int i, c, sz;
    UC(check_frags(nfrags));
    sz = get_size(nw, nfrags);
    Dalloc(&map->counts,  sz);
    Dalloc(&map->starts,  sz);
    Dalloc(&map->offsets, sz);
    Dalloc(&map->cap, nfrags);
    cH2D(map->cap, cap, nfrags);

    for (i = 0; i < nfrags; ++i) {
        c = cap[i];
        Dalloc(&map->ids[i], c);
    }
}

void emap_fin(int nfrags, EMap *map) {
    UC(check_frags(nfrags));
    Dfree(map->counts);
    Dfree(map->starts);
    Dfree(map->offsets);
    Dfree(map->cap);

    for (int i = 0; i < nfrags; ++i)
        Dfree(map->ids[i]);
}

void emap_reini(int nw, int nfrags, /**/ EMap map) {
    size_t sz;
    UC(check_frags(nfrags));
    sz = get_size(nw, nfrags) * sizeof(int);
    if (sz == 0) return;
    CC(d::MemsetAsync(map.counts,  0, sz));
    CC(d::MemsetAsync(map.starts,  0, sz));
    CC(d::MemsetAsync(map.offsets, 0, sz));
}

void emap_scan(int nw, int nfrags, /**/ EMap map) {
    int i, *cc, *ss, *oo, *oon, stride;
    UC(check_frags(nfrags));
    stride = get_stride(nfrags);
    for (i = 0; i < nw; ++i) {
        cc  = map.counts  + i * stride;
        ss  = map.starts  + i * stride;
        oo  = map.offsets + i * stride;
        oon = oo + stride; /* oo next */
        KL(emap_dev::scan2d, (1, 32), (nfrags, cc, oo, /**/ oon, ss));
    }
}

void emap_download_tot_counts(int nw, int nfrags, EMap map, /**/ int counts[]) {
    int *src, stride;
    UC(check_frags(nfrags));
    stride = get_stride(nfrags);
    src = map.offsets + nw * stride;
    cD2H(counts, src, nfrags);
}

static void transpose(int nw, int stride, int nfrags, const int *buf, /**/ int *counts[]) {
    int i, j;
    for (i = 0; i < nfrags; ++i) {
        for (j = 0; j < nw; ++j)
            counts[i][j] = buf[i*stride + j];
    }
}

void emap_download_all_counts(int nw, int nfrags, EMap map, /**/ int *counts[]) {
    int *buf, stride, sz;
    UC(check_frags(nfrags));
    stride = get_stride(nfrags);
    sz = stride * nw;
    EMALLOC(stride * nw, &buf);
    cD2H(buf, map.counts, sz);
    transpose(nw, stride, nfrags, buf, /**/ counts);
    EFREE(buf);
}

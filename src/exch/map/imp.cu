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
    if (nfrags >= MAX_FRAGS)
        ERR("Too many fragments: %d/%d", nfrags, MAX_FRAGS);
}
static int get_stride(int nfrags)       {return nfrags + 1;}
static int get_size(int nw, int nfrags) {return (nw + 1) * get_stride(nfrags);}

void emap_ini(int nw, int nfrags, int cap[], /**/ EMap *map) {
    int i, c;
    size_t sz;
    UC(check_frags(nfrags));
    sz = get_size(nw, nfrags) * sizeof(int);
    CC(d::Malloc((void**) &map->counts,  sz));
    CC(d::Malloc((void**) &map->starts,  sz));
    CC(d::Malloc((void**) &map->offsets, sz));
    CC(d::Malloc((void**) &map->cap, nfrags * sizeof(int)));
    cH2D(map->cap, cap, nfrags);

    for (i = 0; i < nfrags; ++i) {
        c = cap[i];
        sz = c * sizeof(int);
        CC(d::Malloc((void**) &map->ids[i], sz));
    }
}

void emap_fin(int nfrags, EMap *map) {
    UC(check_frags(nfrags));
    CC(d::Free(map->counts));
    CC(d::Free(map->starts));
    CC(d::Free(map->offsets));
    CC(d::Free(map->cap));

    for (int i = 0; i < nfrags; ++i)
        CC(d::Free(map->ids[i]));
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

void emap_download_counts(int nw, int nfrags, EMap map, /**/ int counts[]) {
    int *src, stride;
    size_t sz = nfrags * sizeof(int);
    UC(check_frags(nfrags));
    stride = get_stride(nfrags);
    src = map.offsets + nw * stride;
    CC(d::Memcpy(counts, src, sz, D2H));
}

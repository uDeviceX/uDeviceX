#include <mpi.h>
#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "utils/cc.h"

#include "mpi/glb.h"
#include "inc/dev.h"
#include "inc/type.h"
#include "parser/imp.h"
#include "partlist/type.h"
#include "clist/imp.h"

#include "cloud/imp.h"
#include "flu/type.h"
#include "fluforces/imp.h"

#include "io/txt/imp.h"

static Particle *pp, *pp0;
static Force *ff;
static int n;
static Clist clist;
static ClistMap *cmap;
static FluForcesBulk *bulkforces;

static void read_pp(const char *fname) {
    TxtRead *tr;
    size_t szp, szf;
    UC(txt_read_pp(fname, &tr));
    n = txt_read_get_n(tr);
    msg_print("have read %d particles", n);

    szp = (n + 32) * sizeof(Particle);
    szf = (n + 32) * sizeof(Force);
    
    CC(d::Malloc((void**)&pp, szp));
    CC(d::Malloc((void**)&pp0, szp));
    CC(d::Malloc((void**)&ff, szf));

    CC(d::Memcpy(pp, txt_read_get_pp(tr), szp, H2D));
    CC(d::Memset(ff, 0, szf));
    
    UC(txt_read_fin(tr));
}

static void dealloc() {
    CC(d::Free(pp));
    CC(d::Free(pp0));
    CC(d::Free(ff));
    n = 0;
}

static void build_clist() {
    UC(clist_build(n, n, pp, /**/ pp0, &clist, cmap));
    Particle *tmp = pp;
    pp = pp0;
    pp0 = tmp;
}

int main(int argc, char **argv) {
    Config *cfg;
    const char *fname;
    Cloud cloud;
    int maxp;
    
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    
    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));

    UC(conf_lookup_string(cfg, "fname", &fname));
    UC(read_pp(fname));

    maxp = n + 32;

    UC(clist_ini(XS, YS, ZS, &clist));
    UC(clist_ini_map(maxp, 1, &clist, &cmap));
    UC(build_clist());

    UC(fluforces_bulk_ini(maxp, &bulkforces));

    ini_cloud(pp, &cloud);
    
    UC(fluforces_bulk_prepare(n, &cloud, /**/ bulkforces));
    UC(fluforces_bulk_apply(n, bulkforces, clist.starts, clist.counts, /**/ ff));
    
    UC(fluforces_bulk_fin(bulkforces));

    UC(clist_fin(&clist));
    UC(clist_fin_map(cmap));
    UC(dealloc());
    
    UC(conf_fin(cfg));
    m::fin();
}

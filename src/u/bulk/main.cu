#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "utils/imp.h"
#include "utils/cc.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "inc/dev.h"
#include "inc/type.h"
#include "parser/imp.h"
#include "partlist/type.h"
#include "clist/imp.h"
#include "pair/imp.h"

#include "coords/ini.h"
#include "coords/imp.h"

#include "cloud/imp.h"
#include "flu/type.h"
#include "fluforces/imp.h"

#include "io/txt/imp.h"

static Particle *pp, *pp0, *pp_hst;
static Force *ff, *ff_hst;
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

    UC(emalloc(szp, (void**)&pp_hst));
    UC(emalloc(szf, (void**)&ff_hst));

    CC(d::Malloc((void**)&pp, szp));
    CC(d::Malloc((void**)&pp0, szp));
    CC(d::Malloc((void**)&ff, szf));

    memcpy(pp_hst, txt_read_get_pp(tr), szp);
    CC(d::Memcpy(pp, pp_hst, szp, H2D));
    CC(d::Memset(ff, 0, szf));

    UC(txt_read_fin(tr));
}

static void dealloc() {
    CC(d::Free(pp));
    CC(d::Free(pp0));
    CC(d::Free(ff));
    UC(efree(pp_hst));
    UC(efree(ff_hst));
    n = 0;
}

static void build_clist() {
    UC(clist_build(n, n, pp, /**/ pp0, &clist, cmap));
    Particle *tmp = pp;
    pp = pp0;
    pp0 = tmp;
}

static void set_params(float dt, PairParams *p) {
    enum {ncolors = 2};
    float a[] = {adpd_b, adpd_br, adpd_r};
    float g[] = {gdpd_b, gdpd_br, gdpd_r};
    UC(pair_set_dpd(ncolors, a, g, p));
    UC(pair_compute_dpd_sigma(kBT, dt, /**/ p));
    UC(pair_set_lj(ljsigma, ljepsilon, p));
}

int main(int argc, char **argv) {
    Config *cfg;
    const char *fin, *fout;
    Cloud cloud;
    Coords *coords;
    int maxp;
    int3 L;
    PairParams *params;
    float dt;

    m::ini(&argc, &argv);
    msg_ini(m::rank);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));

    UC(coords_ini_conf(m::cart, cfg, &coords));
    L = subdomain(coords);

    UC(pair_ini(&params));
    UC(conf_lookup_float(cfg, "time.dt", &dt));
    UC(set_params(dt, params));

    UC(conf_lookup_string(cfg, "in", &fin));
    UC(conf_lookup_string(cfg, "out", &fout));
    UC(read_pp(fin));

    maxp = n + 32;

    UC(clist_ini(L.x, L.y, L.z, &clist));
    UC(clist_ini_map(maxp, 1, &clist, &cmap));
    UC(build_clist());

    UC(fluforces_bulk_ini(L, maxp, &bulkforces));

    UC(ini_cloud(pp, &cloud));

    UC(fluforces_bulk_prepare(n, &cloud, /**/ bulkforces));
    UC(fluforces_bulk_apply(params, n, bulkforces, clist.starts, clist.counts, /**/ ff));

    // particles are reordered because of clists
    CC(d::Memcpy(pp_hst, pp, n*sizeof(Particle), D2H));
    CC(d::Memcpy(ff_hst, ff, n*sizeof(Force)   , D2H));
    UC(txt_write_pp_ff(n, pp_hst, ff_hst, fout));

    UC(fluforces_bulk_fin(bulkforces));
    UC(clist_fin(&clist));
    UC(clist_fin_map(cmap));
    UC(dealloc());

    UC(pair_fin(params));
    UC(coords_fin(coords));
    UC(conf_fin(cfg));
    m::fin();
}

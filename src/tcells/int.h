namespace tcells {

struct Quants {
    int *ss_hst, *cc_hst, *ii_hst; /* starts, counts, ids on hst */ 
    int *ss_dev, *cc_dev, *ii_dev; /* starts, counts, ids on dev */ 
};

void build_hst(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids);
void build_dev(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids, /*w*/ scan::Work *w);
}

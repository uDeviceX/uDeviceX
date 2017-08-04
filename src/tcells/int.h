namespace tcells {

struct Quants {
    int *ss_hst, *cc_hst, *ii_hst; /* starts, counts, ids on hst */ 
    int *ss_dev, *cc_dev, *ii_dev; /* starts, counts, ids on dev */ 
};

void alloc_quants(int max_num_mesh, /**/ Quants *q);
void free_quants(/**/ Quants *q);

void build_hst(const Mesh m, const Particle *i_pp, const int ns, /**/ Quants *q);
void build_dev(const Mesh m, const Particle *i_pp, const int ns, /**/ Quants *q, /*w*/ scan::Work *w);
}

struct Contact;
struct int3;
struct PairParams;

// tag::mem[]
void cnt_ini(int maxp, int rank, int3 L, int nobj, /**/ Contact **c);
void cnt_fin(Contact *c);
// end::mem[]

// tag::int[]
void cnt_build_cells(int nw, const PaWrap *pw, /**/ Contact *c); // <1>
void cnt_bulk(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw); // <2>
void cnt_halo(const Contact *cnt, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw, Pap26 all_pp, Fop26 all_ff, const int *all_counts); // <3>
// end::int[]

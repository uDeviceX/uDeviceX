struct Contact;
struct int3;
struct PairParams;

// tag::mem[]
void cnt_ini(int maxp, int rank, int3 L, /**/ Contact **c);
void cnt_fin(Contact *c);
// end::mem[]

// tag::int[]
void cnt_build_cells(int nw, const PaWrap *pw, /**/ Contact *c); // <1>
void cnt_bulk(const PairParams*, const Contact *c, int nw, PaWrap *pw, FoWrap *fw); // <2>
void cnt_halo(const PairParams*, const Contact *c, int nw, PaWrap *pw, FoWrap *fw, Pap26 PP, Fop26 FF, int counts[26]); // <3>
// end::int[]

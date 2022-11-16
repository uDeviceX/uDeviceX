struct RNDunif;
struct Fsi;
struct int3;

// tag::mem[]
void fsi_ini(int rank, int3 L, /**/ Fsi **fsi);
void fsi_fin(Fsi *fsi);
// end::mem[]

// tag::int[]
void fsi_bind_solvent(PaArray pa, Force *ff, int n, const int *starts, /**/ Fsi *fsi);  // <1>
void fsi_bulk(Fsi *fsi, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw); // <2>
void fsi_halo(Fsi *fsi, int nw, const PairParams **prms, Pap26 all_pp, Fop26 all_ff, const int *all_counts); // <3>
// end::int[]

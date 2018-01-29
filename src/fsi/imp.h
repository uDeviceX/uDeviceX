struct RNDunif;
struct Fsi;
struct int3;

// tag::mem[]
void fsi_ini(int rank, int3 L, /**/ Fsi **fsi);
void fsi_fin(Fsi *fsi);
// end::mem[]

// tag::int[]
void fsi_bind_solvent(Cloud c, Force *ff, int n, int *starts, /**/ Fsi *fsi); // <1>
void fsi_bulk(Fsi *fsi, int nw, PaWrap *pw, FoWrap *fw);                      // <2>
void fsi_halo(Fsi *fsi, Pap26 PP, Fop26 FF, int counts[26]);                  // <3>
// end::int[]

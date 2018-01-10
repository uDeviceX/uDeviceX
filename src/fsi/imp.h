struct RNDunif;

void fsi_ini(int rank, /**/ Fsi *fsi);
void fsi_fin(Fsi *fsi);

void fsi_bind(SolventWrap wrap, /**/ Fsi *fsi);
void fsi_bulk(Fsi *fsi, int nw, PaWrap *pw, FoWrap *fw);
void fsi_halo(Fsi *fsi, Pap26 PP, Fop26 FF, int counts[26]);

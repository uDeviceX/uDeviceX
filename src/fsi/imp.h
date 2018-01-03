struct RNDunif;
namespace fsi {
void ini(int rank, /**/ Fsi *fsi);
void fin(Fsi *fsi);
void bind(SolventWrap wrap, /**/ Fsi *fsi);
void bulk(Fsi *fsi, int nw, PaWrap *pw, FoWrap *fw);
void halo(Fsi *fsi, Pap26 PP, Fop26 FF, int counts[26]);
}

namespace cnt {

struct Contact {
    clist::Clist cells;
    clist::Map cmap;
    RNDunif *rgen;
};

void ini(int rank, /**/ Contact *c);
void fin(Contact *c);

void build_cells(int nw, const PaWrap *pw, /**/ Contact *c);
void bulk(const Contact *c, int nw, PaWrap *pw, FoWrap *fw);
void halo(const Contact *c, int nw, PaWrap *pw, FoWrap *fw, Pap26 PP, Fop26 FF, int counts[26]);
}

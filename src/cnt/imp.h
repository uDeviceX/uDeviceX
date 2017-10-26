namespace cnt {

struct Contact {
    clist::Clist cells;
    clist::Map cmap;
    rnd::KISS *rgen;
};

void ini();
void fin();

void ini(Contact *c);
void fin(Contact *c);

void bind(int nw, PaWrap *pw, FoWrap *fw);
void bulk(int nw, PaWrap *pw, FoWrap *fw);
void halo(int nw, PaWrap *pw, FoWrap *fw, Pap26 PP, Fop26 FF, int counts[26]);
}

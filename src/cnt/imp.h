struct Contact;

void cnt_ini(int rank, /**/ Contact **c);
void cnt_fin(Contact *c);

void cnt_build_cells(int nw, const PaWrap *pw, /**/ Contact *c);
void cnt_bulk(const Contact *c, int nw, PaWrap *pw, FoWrap *fw);
void cnt_halo(const Contact *c, int nw, PaWrap *pw, FoWrap *fw, Pap26 PP, Fop26 FF, int counts[26]);


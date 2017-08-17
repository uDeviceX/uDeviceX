namespace dpdr {
namespace sub {
void recv(const int *np, const int *nc, /**/ Rbufs *b) {
    for (int i = 0; i < 26; ++i)
        if (np[i] > 0) aD2D0(b->pp.d[i], b->ppdev.d[i], np[i]);

    for (int i = 0; i < 26; ++i)
        if (nc[i] > 0) aD2D0(b->cum.d[i], b->cumdev.d[i],  nc[i]);
}

void recv_ii(const int *np, /**/ RIbuf *b) {
    for (int i = 0; i < 26; ++i)
        if (np[i] > 0) aD2D0(b->ii.d[i], b->iidev.d[i], np[i]);
}

}
}

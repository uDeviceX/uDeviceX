namespace dpdr {
namespace sub {
void recv(const int *np, const int *nc, /**/ Rbufs *b) {
    for (int i = 0; i < 26; ++i)
        if (np[i] > 0) CC(cudaMemcpyAsync(b->pp.d[i], b->ppdev.d[i], sizeof(Particle) * np[i], D2D));

    for (int i = 0; i < 26; ++i)
        if (nc[i] > 0) CC(cudaMemcpyAsync(b->cum.d[i], b->cumdev.d[i],  sizeof(int) * nc[i], D2D));
}

void recv_ii(const int *np, /**/ RIbuf *b) {
    for (int i = 0; i < 26; ++i)
        if (np[i] > 0) CC(cudaMemcpyAsync(b->ii.d[i], b->iidev.d[i], sizeof(int) * np[i], D2D));
}

}
}

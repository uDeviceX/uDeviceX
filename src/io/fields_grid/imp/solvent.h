void fields_grid(const Coords *coords, MPI_Comm cart, QQ qq, NN nn, /*w*/ Particle *hst) {
    Particle *o;
    int n;
    o = qq.o;
    n = nn.o;
    cD2H(hst, o, n);
    UC(io::field::dump(coords, cart, hst, n));
}

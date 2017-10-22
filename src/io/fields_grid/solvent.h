void fields_grid(QQ qq, NN nn, /*w*/ Particle *hst) {
    Particle *o;
    int n;
    o = qq.o;
    n = nn.o;
    cD2H(hst, o, n);
    io::field::dump(hst, n);
}

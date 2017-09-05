void fields_grid(QQ qq, NN nn, /*w*/ Particle *hst) {

    Particle *o, *s, *r;
    o = qq.o;
    s = qq.s;
    r = qq.r;

    const int n = nn.o + nn.s + nn.r;

    int start = 0;
    cD2H(hst + start, o, nn.o); start += nn.o;
    if (solids0) {
        cD2H(hst + start, s, nn.s); start += nn.s;
    }
    if (rbcs) {
        cD2H(hst + start, r, nn.r); start += nn.r;
    }

    h5::dump(hst, n);
}

void flu_punto_dump(Coords *coords, const FluQuants *q) {
    Particle *dev, *hst;
    int n;
    n = q->n; dev = q->pp; hst = q->pp_hst;
    if (n) cD2H(hst, dev, n);
    UC(punto_dump(n, hst, "preved.punto"));
}

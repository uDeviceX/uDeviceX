int separator(int argc, char **argv) {
    for (int i = 1; i < argc; ++i)
    if (strcmp("--", argv[i]) == 0) return i;
    return -1;
}

void read_data(const char *fpp, BopData *dpp, const char *fii, BopData *dii) {
    char fdname[FILENAME_MAX];

    BPC( bop_read_header(fpp, dpp, fdname) );
    BPC( bop_alloc(dpp) );
    BPC( bop_read_values(fdname, dpp) );
    
    BPC( bop_read_header(fii, dii, fdname) );
    BPC( bop_alloc(dii) );
    BPC( bop_read_values(fdname, dii) );

    BopType tp, ti;
    BPC( bop_get_type(dpp, &tp) );
    BPC( bop_get_type(dii, &ti) );
    if (tp != BopFLOAT && tp != BopFASCII) ERR("expected float data form <%s>\n", fpp);
    if (ti != BopINT   && ti != BopIASCII) ERR("expected int data form <%s>\n", fii);
}

int max_index(const int *ii, const int n) {
    int m = -1;
    for (int i = 0; i < n; ++i) m = m < ii[i] ? ii[i] : m;
    return m;
}

void pp2rr_sorted(const int *ii, const float *fdata, const int n, const int stride, /**/ float *rr) {
    for (int j = 0; j < n; ++j) {
        const int i = ii[j];
        const float *r = fdata + j * stride;
        for (int c = 0; c < 3; ++c)
        rr[3*i + c] = r[c];
    }
}

static int separator(int argc, char **argv) {
    for (int i = 1; i < argc; ++i)
    if (strcmp("--", argv[i]) == 0) return i;
    return -1;
}

static void read_data(const char *fpp, BopData *dpp, const char *fii, BopData *dii) {
    char fdname[CBUFSIZE];

    bop_read_header(fpp, dpp, fdname);
    bop_alloc(dpp);
    bop_read_values(fdname, dpp);
    
    bop_read_header(fii, dii, fdname);
    bop_alloc(dii);
    bop_read_values(fdname, dii);

    if (dpp->type != FLOAT && dpp->type != FASCII) ERR("expected float data form <%s>\n", fpp);
    if (dii->type != INT   && dii->type != IASCII) ERR("expected int data form <%s>\n", fii);
}

static int max_index(const int *ii, const int n) {
    int m = -1;
    for (int i = 0; i < n; ++i) m = m < ii[i] ? ii[i] : m;
    return m;
}

static void pp2rr_sorted(const int *ii, const float *fdata, const int n, const int stride, /**/ float *rr) {
    for (int j = 0; j < n; ++j) {
        const int i = ii[j];
        const float *r = fdata + j * stride;
        for (int c = 0; c < 3; ++c)
        rr[3*i + c] = r[c];
    }
}

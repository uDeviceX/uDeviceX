static bool prop(const char *pname, const char *str) {
    const int l1 = strlen(pname);
    const int l2 = strlen(str);

    if (l1 > l2) return false;
    for (int i = 0; i < l1; ++i) if (pname[i] != str[i]) return false;
    return true;
}

static void read_ply(FILE *f, const char *fname, MeshRead *q) {
    int nv, nt;
    int4 *tt;
    float *rr;
    int l = 0;
#define BUFSIZE 256 // max number of chars per line
#define MAXLINES 64 // max number of line for header

    // https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html#Stringizing
#define xstr(s) str(s)
#define str(s) #s

    nv = nt = -1;
    while (l++ < MAXLINES) {
        char cbuf[BUFSIZE + 1] = {0}; // + 1 for \0

        const int checker = fscanf(f, " %[^\n]" xstr(BUFSIZE) "c", cbuf);
        if (checker != 1)
            ERR("Fail to read '%s'", fname);

        int ibuf;
        if    (sscanf(cbuf, "element vertex %d", &ibuf) == 1) nv = ibuf;
        else if (sscanf(cbuf, "element face %d", &ibuf) == 1) nt = ibuf;
        else if (prop("end_header", cbuf)) break;
    }
    if (l >= MAXLINES || nt == -1 || nv == -1)
        ERR("Fail to read '%s': did not catch end_header", fname);
    EMALLOC(  nt, &tt);
    EMALLOC(3*nv, &rr);
    for (int i = 0; i < nv; ++i)
        fscanf(f, "%f %f %f\n",
               rr + 3*i + 0, rr + 3*i + 1, rr + 3*i + 2);
    int4 t; t.z = 0;
    for (int i = 0; i < nt; ++i) {
        fscanf(f, "%*d %d %d %d\n", &t.x, &t.y, &t.z);
        tt[i] = t;
    }
    q->nv = nv; q->nt = nt;
    q->rr = rr; q->tt = tt;
}

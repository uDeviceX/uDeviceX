static bool prop(const char *pname, const char *str) {
    const int l1 = strlen(pname);
    const int l2 = strlen(str);

    if (l1 > l2) return false;
    for (int i = 0; i < l1; ++i) if (pname[i] != str[i]) return false;
    return true;
}

static void read_ply(const char *fname, int *nt, int *nv, int4 **tt, float **vv) {
    FILE *f;
    UC(efopen(fname, "r", /**/ &f));

    int l = 0;
    *nt = *nv = -1;

#define BUFSIZE 256 // max number of chars per line
#define MAXLINES 64 // max number of line for header

    // https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html#Stringizing
#define xstr(s) str(s)
#define str(s) #s

    while (l++ < MAXLINES) {
        char cbuf[BUFSIZE + 1] = {0}; // + 1 for \0

        const int checker = fscanf(f, " %[^\n]" xstr(BUFSIZE) "c", cbuf);

        if (checker != 1) {
            fprintf(stderr, "Something went wrong reading <%s>\n", fname);
            exit(1);
        }

        int ibuf;
        if    (sscanf(cbuf, "element vertex %d", &ibuf) == 1) *nv = ibuf;
        else if (sscanf(cbuf, "element face %d", &ibuf) == 1) *nt = ibuf;
        else if (prop("end_header", cbuf)) break;
    }

    if (l >= MAXLINES || *nt == -1 || *nv == -1) {
        printf("Something went wrong, did not catch end_header\n");
        exit(1);
    }

    *tt = new int4[*nt];
    *vv = new float[3 * *nv];

    for (int i = 0; i < *nv; ++i)
        fscanf(f, "%f %f %f\n",
               *vv + 3*i + 0,
               *vv + 3*i + 1,
               *vv + 3*i + 2);

    int4 t; t.z = 0;
    for (int i = 0; i < *nt; ++i) {
        fscanf(f, "%*d %d %d %d\n", &t.x, &t.y, &t.z);
        (*tt)[i] = t;
    }

    UC(efclose(f));
}

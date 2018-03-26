static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }

static void mkdir(const char *p, const char *s) {
    char path[FILENAME_MAX];
    cpy(path, p);
    cat(path, "/");
    cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}

static int max3(int a, int b, int c) {
    if (a > b) b = a;
    if (b > c) c = b;
    return c;
}
static int get_nbuf(int n, int nv, int nt, int ne) {
    int nm;
    nm = n / nv;
    return max3(3*n, 4*nm*nt, 3*nm*ne);
}

static int little_p0() {
    int a;
    unsigned char *b;
    a = 1;
    b = (unsigned char*)&a;
    return (*b) != 0;
}

static int little_p() {
    static int done = 0;
    static int little = 0;
    if (done) return little;
    little = little_p0();
    done = 1;
    return little;
}

static void big_endian_dbl0(double *px) {
    const int n = sizeof(double);
    int i;
    unsigned char *c, b[n];
    if (!little_p()) return;
    c = (unsigned char*)px;
    for (i = 0; i < n; i++) b[i] = c[i];
    for (i = 0; i < n; i++) c[i] = b[n - i - 1];
}

static void big_endian_dbl(int n, double *d) {
    int i;
    for (i = 0; i < n; i++)
        big_endian_dbl0(&d[i]);
}

static void big_endian_int0(int *px) {
    const int n = sizeof(int);
    int i;
    unsigned char *c, b[n];
    if (!little_p()) return;
    c = (unsigned char*)px;
    for (i = 0; i < n; i++) b[i] = c[i];
    for (i = 0; i < n; i++) c[i] = b[n - i - 1];
}

static void big_endian_int(int n, int *d) {
    int i;
    for (i = 0; i < n; i++)
        big_endian_int0(&d[i]);
}

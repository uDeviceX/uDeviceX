static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }

static int nword(const char *s) {
    enum {OUT, IN};
    const char *s0;
    char c;
    int state, n;
    s0 = s;
    state = OUT;
    n = 0;
    while ((c = *s++) != '\0') {
        if (!(isalnum(c) || isspace(c)))
            ERR("illegal character '%c' in '%s'", c, s0);
        else if (isspace(c))
            state = OUT;
        else if (state == OUT) {
            state = IN;
            n++;
        }
    }
    if (n == 0)
        ERR("wrong keys for bop: '%s'", s0);
    return n;
}

static void wrong_key(IOPoint *q, const char *key) {
    int i, nkey;
    nkey = q->nkey;
    msg_print("unkown key: '%s'", key);
    msg_print("possible values:");
    for (i = 0; i < nkey; i++)
        msg_print("'%s'", q->keys[i]);
    ERR("");
}

static void mkdir(const char *p, const char *s) {
    char path[FILENAME_MAX];
    cpy(path, p);
    cat(path, "/");
    cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}

static void reset(IOPoint *q) {
    int i, nkey;
    nkey = q->nkey;
    for (i = 0; i < nkey; i++)
        q->seen[i] = 0;
    q->n = UNSET;
}

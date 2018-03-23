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

static void reset(IOPoint *q) {
    int i, nkey;
    nkey = q->nkey;
    for (i = 0; i < nkey; i++)
        q->seen[i] = 0;
    q->n = UNSET;
}

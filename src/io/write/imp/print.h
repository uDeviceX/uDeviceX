enum { BUF = 9999 };

static void print(MPI_Comm comm, WriteFile* f, const char *fmt, ...) {
    char s[BUF];
    int sz;
    va_list args;
    if (!m::is_master(comm)) return;
    va_start (args, fmt);
    sz = vsnprintf(s, BUF - 1, fmt, args);
    va_end (args);
    msg_print(stderr, "sz: %d", sz);
}

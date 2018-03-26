enum { BUF = 9999 };

static void print(Out *out, const char *fmt, ...) {
    char s[BUF];
    int size;
    va_list args;
    size = 0;    
    if (m::is_master(out->comm)) {
        va_start(args, fmt);
        size = vsnprintf(s, BUF - 1, fmt, args);
        va_end(args);
    }
    UC(write_master(out->comm, s, size, out->file));
}

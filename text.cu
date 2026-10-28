#include "macros.cuh"
#include <ctype.h>

void error_signal(const char *file, int line, const char *fmt, ...);
void error_report();
void error_abort();


int string_nword(const char *s) {
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
    return n;
}

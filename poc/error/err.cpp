#include <stdio.h>
#include <stdarg.h>
#include "h.h"

static int s;
static int n;
static const char *f;
static char msg[BUFSIZ];

void line(int n0)   { n = n0; }
void set  (int s0)  { s = s0; }
void file(const char *f0) { f = f0; }

int status() { return s; };

void extra(const char* fmt, ... ) {
    va_list ap;
    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);    
};

void format() {
    printf("%s\n", msg);
}

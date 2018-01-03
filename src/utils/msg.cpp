#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "msg.h"

static int rank;
static const char *fmt = ".%03d";

// true if master rank
static void is_master() {return rank == 0;}

void msg_ini(int rnk) {
    rank = rnk;
}

void msg_print(const char *fmt, ...) {

}


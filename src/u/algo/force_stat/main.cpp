#include <stdio.h>
#include <stdlib.h>

#include "utils/msg.h"
#include "mpi/glb.h"

#include "parser/imp.h"
#include "utils/error.h"
#include "utils/imp.h"
#include "io/txt/imp.h"

static const char *prog_name = "./udx";
void usg0() {
    fprintf(stderr, "%s i=\\\"force.txt\\\"\n", prog_name);
    fprintf(stderr, "force.txt: f0x f0y f0z\n");
    fprintf(stderr, "           ...\n");
    fprintf(stderr, "           fnx fny fnz\n");
    fprintf(stderr, "dumps forces statistics\n");
    exit(0);
}

void usg(int c, char **v) {
    if (c > 1 && same_str(v[1], "-h"))
        usg0();
}

void main0(int n, const Force *ff) {
    msg_print("n: %d", n);
}

int main(int argc, char **argv) {
    const char *i; /* input file */
    const Force *ff;
    int n;
    Config *cfg;
    TxtRead *txt;
    usg(argc, argv);
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, /**/ cfg));

    UC(conf_lookup_string(cfg, "i", &i));
    UC(txt_read_ff(i, &txt));

    ff = txt_read_get_ff(txt);
    n  = txt_read_get_n(txt);
    UC(main0(n, ff));
    
    UC(txt_read_fin(txt));
    UC(conf_fin(cfg));
    m::fin();
}

#include <stdio.h>
#include <vector_types.h>
#include "inc/type.h"

#include <conf.h>
#include "inc/conf.h"
#include "utils/error.h"
#include "utils/imp.h"
#include "coords/type.h"
#include "coords/imp.h"

enum {X, Y, Z};

static void write_v(FILE *f, const float v[3]) {
    fprintf(f, "%+.6e %+.6e %+.6e ", v[X], v[Y], v[Z]);
}

void rig_dump(const int it, const Solid *ss, const Solid *ssbb, int ns, const Coords *c) {
    enum {X, Y, Z};
    static bool first = true;
    char fname[256];
    float com[3];
    FILE *fp;
    int j;
    const Solid *s, *sbb;

    for (j = 0; j < ns; ++j) {
        s   = ss   + j;
        sbb = ssbb + j;
            
        sprintf(fname, DUMP_BASE "/solid_diag_%04d.txt", (int) s->id);
        if (first) UC(efopen(fname, "w", /**/ &fp));
        else       UC(efopen(fname, "a", /**/ &fp));

        fprintf(fp, "%+.6e ", dt*it);

        // shift to global coordinates
        com[X] = xl2xg(c, s->com[X]);
        com[Y] = yl2yg(c, s->com[Y]);
        com[Z] = zl2zg(c, s->com[Z]);
            
        write_v(fp, com);
        write_v(fp, s->v );
        write_v(fp, s->om);
        write_v(fp, s->fo);
        write_v(fp, s->to);
        write_v(fp, s->e0);
        write_v(fp, s->e1);
        write_v(fp, s->e2);
        write_v(fp, sbb->fo);
        write_v(fp, sbb->to);
        fprintf(fp, "\n");
        
        UC(efclose(fp));
    }

    first = false;
}

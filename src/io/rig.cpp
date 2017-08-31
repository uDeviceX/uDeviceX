#include <stdio.h>
#include "inc/type.h"

#include <conf.h>
#include "inc/conf.h"

enum {X, Y, Z};

static void write_v(FILE *f, const float v[3]) {
    fprintf(f, "%+.6e %+.6e %+.6e ", v[X], v[Y], v[Z]);
}

void rig_dump(const int it, const Solid *ss, const Solid *ssbb, int ns, const int *mcoords) {
    static bool first = true;
    char fname[256];

    for (int j = 0; j < ns; ++j) {
        const Solid *s   = ss   + j;
        const Solid *sbb = ssbb + j;
            
        sprintf(fname, DUMP_BASE "/solid_diag_%04d.txt", (int) s->id);
        FILE *fp;
        if (first) fp = fopen(fname, "w");
        else       fp = fopen(fname, "a");

        fprintf(fp, "%+.6e ", dt*it);

        // make global coordinates
        float com[3];
        {
            const int L[3] = {XS, YS, ZS};
            int mi[3];
            for (int c = 0; c < 3; ++c) mi[c] = (mcoords[c] + 0.5) * L[c];
            for (int c = 0; c < 3; ++c) com[c] = s->com[c] + mi[c];
        }
            
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
        
        fclose(fp);
    }

    first = false;
}

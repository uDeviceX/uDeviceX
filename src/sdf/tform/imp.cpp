#include <stdio.h>

#include "utils/error.h"

#include "math/tform/imp.h"
#include "imp.h"

static void ini_tex2sdf0(Tform *tex2g, Tform *sdf2g, Tform *g2sdf, /**/ Tform *tex2sdf) {
    /* [G, T, L, M, N, lo] */

}

void ini_tex2sdf(/**/ Tform *tex2sdf) {
    /*
       [tex2g, g2tex]: genfi(-1/2, lo - M,   T-1/2, lo + L + M) $
       [sdf2g, g2sdf]: genfi(-1/2,      0,   N-1/2,          G) $
       tex2sdf: chain(tex2g, g2sdf)
    */
    Tform *tex2g, *sdf2g, *g2sdf;
    UC(tform_ini(&tex2g));
    UC(tform_ini(&sdf2g));
    UC(tform_ini(&g2sdf));

    UC(ini_tex2sdf0(tex2g, sdf2g, g2sdf, /**/ tex2sdf));

    tform_fin(tex2g);
    tform_fin(sdf2g);
    tform_fin(g2sdf);
}

void ini_sub2tex(/**/ Tform*) {

}

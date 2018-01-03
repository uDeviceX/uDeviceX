#include <stdio.h>

#include "math/tform/imp.h"
#include "imp.h"

void ini_tex2sdf(/**/ Tform *tex2sdf) {
    /*
       [tex2g, g2tex]: genfi(-1/2, lo - M,   T-1/2, lo + L + M) $
       [sdf2g, g2sdf]: genfi(-1/2,      0,   N-1/2,          G) $
       tex2sdf: chain(tex2g, g2sdf)
    */
    Tform *tex2g, *sdf2g, *g2sdf;
    tform_ini(&tex2g);
    tform_ini(&sdf2g);
    tform_ini(&g2sdf);


    tform_fin(tex2g);
    tform_fin(sdf2g);
    tform_fin(g2sdf);
}

void ini_sub2tex(/**/ Tform*) {

}

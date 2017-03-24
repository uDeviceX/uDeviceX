#include "sbounce.h"
#include <cstdio>

int main(int argc, char **argv)
{
    float rg[3] = {4.5f, -3.8f, 1.8f};
    float vg[3] = {0.4f, -1.8f, 0.2f};
    float rl[3], r[3], vl[3], v[3];

    float com[3] = {0.4f, 5.1f, -3.4f};

    float e0[3] = {1, 0, 0};
    float e1[3] = {0, 1, 0};
    float e2[3] = {0, 0, 1};
    
    solidbounce::r2local (e0, e1, e2, com, rg, /**/ rl);
    solidbounce::r2global(e0, e1, e2, com, rl, /**/ r );

    printf("rg = %+.10e %+.10e %+.10e\n", rg[0], rg[1], rg[2]);
    printf("rl = %+.10e %+.10e %+.10e\n", rl[0], rl[1], rl[2]);
    printf("r  = %+.10e %+.10e %+.10e\n",  r[0],  r[1],  r[2]);

    printf("\n");
    
    solidbounce::v2local (e0, e1, e2, vg, /**/ vl);
    solidbounce::v2global(e0, e1, e2, vl, /**/ v );

    printf("vg = %+.10e %+.10e %+.10e\n", vg[0], vg[1], vg[2]);
    printf("vl = %+.10e %+.10e %+.10e\n", vl[0], vl[1], vl[2]);
    printf("v  = %+.10e %+.10e %+.10e\n",  v[0],  v[1],  v[2]);

};

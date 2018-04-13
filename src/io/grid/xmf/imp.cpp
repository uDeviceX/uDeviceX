#include <string.h>
#include <stdio.h>
#include <vector_types.h>

#include "utils/error.h"
#include "utils/imp.h"

#include "imp.h"

static void header(FILE *f) {
    fprintf(f, "<?xml version=\"1.0\" ?>\n");
    fprintf(f, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(f, "<Xdmf Version=\"2.0\">\n");
    fprintf(f, " <Domain>\n");
}

static void epilogue(FILE *f) {
    fprintf(f, " </Domain>\n");
    fprintf(f, "</Xdmf>\n");
}

static float3 get_spacing(int3 D, int3 G) {
    float3 d;
    d.x = (float) D.x / (float) G.x;
    d.y = (float) D.y / (float) G.y;
    d.z = (float) D.z / (float) G.z;
    return d;
}

static void grid(int3 D, int3 G, FILE *f, const char *path, int n, const char **names) {
    int i;
    float3 o, d; /* origin, spacing */
    o.x = o.y = o.z = 0.0;
    d = get_spacing(D, G);

    fprintf(f, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
    fprintf(f, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", 1 + G.z, 1 + G.y, 1 + G.x);
    fprintf(f, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(f, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", o.z, o.y, o.x);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", d.z, d.y, d.x);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "     </Geometry>\n");
    for(i = 0; i < n; ++i) {
        fprintf(f, "     <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", names[i]);
        fprintf(f, "       <DataItem Dimensions=\"%d %d %d 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", G.z, G.y, G.x);
        fprintf(f, "        %s:/%s\n", path, names[i]);
        fprintf(f, "       </DataItem>\n");
        fprintf(f, "     </Attribute>\n");
    }
    fprintf(f, "   </Grid>\n");
}

static int eq(const char *a, const char *b) { return !strcmp(a, b); }
static void xsuffix(const char *i, /**/ char *o) {
    /* replace suffix .h5 by .xmf */
    char s[] = ".h5";
    int n;
    n = strlen(i) - strlen(s);
    if (n >= 0 && eq(i + n, s)) {strncpy(o, i, n); o[n] = '\0';}
    else                        strcpy (o, i);
    strcat(o, ".xmf");
}
static void basename(const char *i, /**/ char *o) {
    const char *p;
    p = i;
    while (*i != '\0') if (*i++ == '/') p = i;
    strcpy(o, p);
}

void xmf_write(int3 domainSize, int3 gridSize, const char *path, const char **names, int ncomp) {
    char w[BUFSIZ];
    FILE *f;

    xsuffix(path, /**/ w);
    UC(efopen(w, "w", /**/ &f));
    header(f);

    basename(path, /**/ w);
    grid(domainSize, gridSize, f, w, ncomp, names);
    epilogue(f);
    UC(efclose(f));
}

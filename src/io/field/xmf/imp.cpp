#include <string>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "imp.h"

namespace xmf {

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

static void grid(FILE * f, const char * const path, const char * const *names, int n) {
    enum {X, Y, Z};
    int i;
    int *d, G[3]; /* domain size */
    d = m::dims;
    G[X] = XS*d[X]; G[Y] = YS*d[Y]; G[Z] = ZS*d[Z];

    fprintf(f, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
    fprintf(f, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", 1 + G[Z], 1 + G[Y], 1 + G[X]);
    fprintf(f, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(f, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", 0.0, 0.0, 0.0);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(f, "        %e %e %e\n", 1.0, 1.0, 1.0);
    fprintf(f, "       </DataItem>\n");
    fprintf(f, "     </Geometry>\n");
    for(i = 0; i < n; ++i) {
        fprintf(f, "     <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", names[i]);
        fprintf(f, "       <DataItem Dimensions=\"%d %d %d 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", G[Z], G[Y], G[X]);
        fprintf(f, "        %s:/%s\n", path, names[i]);
        fprintf(f, "       </DataItem>\n");
        fprintf(f, "     </Attribute>\n");
    }
    fprintf(f, "   </Grid>\n");
}

static void xsuffix(char *i, /**/ char *o) {
    /* replace suffix by `xmf'? */
    sprintf(o, "%s.xmf", std::string(i).substr(0, std::string(i).find_last_of(".h5") - 2).data());
}

void write(const char* path, const char * const * const names, int n) {
    char w[BUFSIZ];
    FILE *f;

    xsuffix(path, /**/ w);
    f = fopen(w, "w");
    header(f);
    grid(f, std::string(path).substr(std::string(path).find_last_of("/") + 1).c_str(), names, n);
    epilogue(f);
    fclose(f);
}

} /* namespace */

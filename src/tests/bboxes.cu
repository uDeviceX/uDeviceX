#include <cstdio>
#include <cstdlib>

#include "../common.h"
#include "../k/mesh.h"
#include "../mesh.impl.h"

enum {X, Y, Z};
void gen(Particle *pp, int n)
{
    printf("%d\n", n);
    for (int i = 0; i < n; ++i)
    {
        pp[i].r[X] = drand48();
        pp[i].r[Y] = drand48();
        pp[i].r[Z] = drand48();
    }
}

int main(int argc, char **argv)
{
    const int ns = 12;
    const int nps = 274;
    
    const int n = ns * nps;
    Particle *pp_hst = new Particle[n], *pp_dev;
    float *bboxes_hst = new float[6*n], *bboxes_dev;
    float *bboxes = new float[6*n];

    printf("1\n");
    
    CC(cudaMalloc(&pp_dev, n * sizeof(Particle)));
    CC(cudaMalloc(&bboxes_dev, 6 * n * sizeof(float)));

    printf("1\n");
    gen(pp_hst, n);
    printf("1\n");;
    CC(cudaMemcpy(pp_dev, pp_hst, n*sizeof(Particle), H2D));

    //mesh::bboxes_hst(pp_hst, nps, ns, /**/ bboxes_hst);
    //mesh::bboxes_dev(pp_dev, nps, ns, /**/ bboxes_dev);

    delete[] pp_hst;
    delete[] bboxes;
    delete[] bboxes_hst;

    CC(cudaFree(pp_dev));
    CC(cudaFree(bboxes_dev));

    return 0;
}

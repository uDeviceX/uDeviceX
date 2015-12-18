#pragma once

#include <vector>
#include <list>
#include <string>

struct Extent
{
    float xmin, ymin, zmin;
    float xmax, ymax, zmax;
};

Extent compute_extent(const char * const path);

struct TransformedExtent
{
    float transform[4][4];

    float xmin[3], xmax[3], local_xmin[3], local_xmax[3];

    TransformedExtent(Extent extent, const int domain_extent[3]);

    void build_transform(const Extent extent, const int domain_extent[3]);

    void apply(float x[3], float y[3]);

    bool collides(const TransformedExtent a, const float tol);
};

class Checker
{
    const float safetymargin;
    float h[3];
    int n[3], ntot;

    std::vector<std::list<TransformedExtent> > data;

public:

Checker(float hh, int dext[3], const float safetymargin):
    safetymargin(safetymargin)
    {
        h[0] = h[1] = h[2] = hh;

        for (int d = 0; d < 3; ++d)
            n[d] = (int)ceil((double)dext[d] / h[d]) + 2;

        ntot = n[0] * n[1] * n[2];

        data.resize(ntot);
    }

    bool check(TransformedExtent& ex);

    void add(TransformedExtent& ex);
};

static void verify(std::string path2ic)
{
    printf("VERIFYING <%s>\n", path2ic.c_str());

    FILE * f = fopen(path2ic.c_str(), "r");

    bool isgood = true;

    while(isgood)
    {
        float tmp[19];
        for(int c = 0; c < 19; ++c)
        {
            int retval = fscanf(f, "%f", tmp + c);

            isgood &= retval == 1;
        }

        if (isgood)
        {
            printf("reading: ");

            for(int c = 0; c < 19; ++c)
                printf("%f ", tmp[c]);

            printf("\n");
        }
    }

    fclose(f);

    printf("========================================\n\n\n\n");
}


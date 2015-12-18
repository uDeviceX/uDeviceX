/*
 *  main.cpp
 *  Part of CTC/cell-placement/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-18.
 *  Further edited by Dmitry Alexeev on 2014-03-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>

#include "checker.h"

using namespace std;

int main(int argc, const char ** argv)
{
    if (argc != 7)
    {
        printf("usage: ./cell-placement <xdomain-extent> <ydomain-extent> <zdomain-extent> <singlectc-xpos> <singlectc-ypos> <singlectc-zpos>\n");
        exit(-1);
    }

    int domainextent[3];
    for(int i = 0; i < 3; ++i)
        domainextent[i] = atoi(argv[1 + i]);

    float ctcposition[3];
    for(int i = 0; i < 3; ++i)
        ctcposition[i] = atoi(argv[1 + 3 + i]);

    printf("domain extent: %d %d %d, ctc position: %f %f %f\n",
	   domainextent[0], domainextent[1], domainextent[2],
	   ctcposition[0], ctcposition[1], ctcposition[2]);

    Extent extents[2] = {
            compute_extent("../cuda-rbc/rbc2.atom_parsed"),
            compute_extent("../cuda-ctc/sphere.dat")
    };

    bool failed = false;

    vector<TransformedExtent> results[2];

    const float tol = 0.02;

    Checker checker(8 + tol, domainextent, tol);
    int tot = 0;

    TransformedExtent onectc(extents[1], domainextent);

    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            onectc.transform[i][j] = i == j;

    onectc.transform[0][3] = ctcposition[0];
    onectc.transform[1][3] = ctcposition[1];
    onectc.transform[2][3] = ctcposition[2];

    onectc.xmin[0] = extents[1].xmin + onectc.transform[0][3];
    onectc.xmin[1] = extents[1].ymin + onectc.transform[1][3];
    onectc.xmin[2] = extents[1].zmin + onectc.transform[2][3];
    onectc.xmax[0] = extents[1].xmax + onectc.transform[0][3];
    onectc.xmax[1] = extents[1].ymax + onectc.transform[1][3];
    onectc.xmax[2] = extents[1].zmax + onectc.transform[2][3];

    checker.add(onectc);
    results[1].push_back(onectc);
    ++tot;

    while(!failed)
    {
        const int maxattempts = 10000000;

        int attempt = 0;
        for(; attempt < maxattempts; ++attempt)
        {
            TransformedExtent t(extents[0], domainextent);

            bool noncolliding = true;

#if 0
            //original code
            for(int i = 0; i < 2; ++i)
                for(int j = 0; j < results[i].size() && noncolliding; ++j)
                    noncolliding &= !t.collides(results[i][j], tol);
#else
            noncolliding = checker.check(t);
#endif

            if (noncolliding)
            {
                checker.add(t);
                results[0].push_back(t);
                ++tot;
                break;
            }
        }

        if (tot % 1000 == 0)
	{
            printf("Done with %d cells...\n", tot);
	    const double hematocrit = (tot - 1) * 94. / (domainextent[0] * domainextent[1] * domainextent[2]);
	    printf("Hematocrit %.1f%%\n", hematocrit*100);
	}

        failed |= attempt == maxattempts;
    }

    string output_names[2] = { "rbcs-ic.txt", "ctcs-ic.txt" };

    for(int idtype = 0; idtype < 2; ++idtype)
    {
        FILE * f = fopen(output_names[idtype].c_str(), "w");

        for(vector<TransformedExtent>::iterator it = results[idtype].begin(); it != results[idtype].end(); ++it)
        {
            for(int c = 0; c < 3; ++c)
                fprintf(f, "%f ", 0.5 * (it->xmin[c] + it->xmax[c]));

            for(int i = 0; i < 4; ++i)
                for(int j = 0; j < 4; ++j)
                    fprintf(f, "%f ", it->transform[i][j]);

            fprintf(f, "\n");
        }

        fclose(f);
    }

    printf("Generated %d RBCs, %d CTCs\n", (int)results[0].size(), (int)results[1].size());
    const double hematocrit = (tot - 1) * 94. / (domainextent[0] * domainextent[1] * domainextent[2]);
    printf("Hematocrit %.1f%%\n", hematocrit*100);

    return 0;
}

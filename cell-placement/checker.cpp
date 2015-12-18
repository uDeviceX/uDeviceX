#include <cmath>
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "checker.h"

using namespace std;

Extent compute_extent(const char * const path)
{
    ifstream in(path);
    string line;

    if (in.good())
        cout << "Reading file " << path << endl;
    else
    {
        cout << path << ": no such file" << endl;
        exit(1);
    }

    int nparticles, nbonds, ntriang, ndihedrals;

    in >> nparticles >> nbonds >> ntriang >> ndihedrals;

    if (in.good())
        cout << "File contains " << nparticles << " atoms, " << nbonds << " bonds, " << ntriang
	     << " triangles and " << ndihedrals << " dihedrals" << endl;
    else
    {
        cout << "Couldn't parse the file" << endl;
        exit(1);
    }

    vector<float> xs(nparticles), ys(nparticles), zs(nparticles);

    for(int i = 0; i < nparticles; ++i)
    {
        int dummy;
        in >> dummy >> dummy >> dummy >> xs[i] >> ys[i] >> zs[i];
    }

    Extent retval1 = {
	*min_element(xs.begin(), xs.end()),
	*min_element(ys.begin(), ys.end()),
	*min_element(zs.begin(), zs.end()),
	*max_element(xs.begin(), xs.end()),
	*max_element(ys.begin(), ys.end()),
	*max_element(zs.begin(), zs.end())
    };

    for (int i=0; i<nparticles; i++)
    {
        xs[i] += -0.5*(retval1.xmin + retval1.xmax);
        ys[i] += -0.5*(retval1.ymin + retval1.ymax);
        zs[i] += -0.5*(retval1.zmin + retval1.zmax);
    }

    Extent retval = {
	*min_element(xs.begin(), xs.end()),
	*min_element(ys.begin(), ys.end()),
	*min_element(zs.begin(), zs.end()),
	*max_element(xs.begin(), xs.end()),
	*max_element(ys.begin(), ys.end()),
	*max_element(zs.begin(), zs.end())
    };

    {
        printf("extent: \n");
        for(int i = 0; i < 6; ++i)
            printf("%f ", *(i + (float *)(&retval.xmin)));
        printf("\n");
    }

    return retval;
}

TransformedExtent::TransformedExtent(Extent extent, const int domain_extent[3])
{
    local_xmin[0] = extent.xmin;
    local_xmin[1] = extent.ymin;
    local_xmin[2] = extent.zmin;

    local_xmax[0] = extent.xmax;
    local_xmax[1] = extent.ymax;
    local_xmax[2] = extent.zmax;

    build_transform(extent, domain_extent);

    for(int i = 0; i < 8; ++i)
    {
	const int idx[3] = { i % 2, (i/2) % 2, (i/4) % 2 };

	float local[3];
	for(int c = 0; c < 3; ++c)
	    local[c] = idx[c] ? local_xmax[c] : local_xmin[c];

	float world[3];

	apply(local, world);

	if (i == 0)
	    for(int c = 0; c < 3; ++c)
		xmin[c] = xmax[c] = world[c];
	else
	    for(int c = 0; c < 3; ++c)
	    {
		xmin[c] = min(xmin[c], world[c]);
		xmax[c] = max(xmax[c], world[c]);
	    }
    }
}

void TransformedExtent::build_transform(const Extent extent, const int domain_extent[3])
{
    for(int i = 0; i < 4; ++i)
	for(int j = 0; j < 4; ++j)
	    transform[i][j] = i == j;

    for(int i = 0; i < 3; ++i)
	transform[i][3] = - 0.5 * (local_xmin[i] + local_xmax[i]);

    const float angles[3] = {
	(float)(M_PI/2 * 0 + 0 * 0.01 * (drand48() - 0.5) * 2 * M_PI),
	(float)(0 + 0.01 * (drand48() * 2 - 1) * M_PI),
	(float)(M_PI/2 + 0.01 * (drand48() - 0.5) * 2 * M_PI)
    };

    for(int d = 0; d < 3; ++d)
    {
	const float c = cos(angles[d]);
	const float s = sin(angles[d]);

	float tmp[4][4];

	for(int i = 0; i < 4; ++i)
	    for(int j = 0; j < 4; ++j)
		tmp[i][j] = i == j;

	if (d == 0)
	{
	    tmp[0][0] = tmp[1][1] = c;
	    tmp[0][1] = -(tmp[1][0] = s);
	}
	else
	    if (d == 1)
	    {
		tmp[0][0] = tmp[2][2] = c;
		tmp[0][2] = -(tmp[2][0] = s);
	    }
	    else
	    {
		tmp[1][1] = tmp[2][2] = c;
		tmp[1][2] = -(tmp[2][1] = s);
	    }

	float res[4][4];
	for(int i = 0; i < 4; ++i)
	    for(int j = 0; j < 4; ++j)
	    {
		float s = 0;

		for(int k = 0; k < 4; ++k)
		    s += transform[i][k] * tmp[k][j];

		res[i][j] = s;
	    }

	for(int i = 0; i < 4; ++i)
	    for(int j = 0; j < 4; ++j)
		transform[i][j] = res[i][j];
    }

    float maxlocalextent = 0;
    for(int i = 0; i < 3; ++i)
	maxlocalextent = max(maxlocalextent, local_xmax[i] - local_xmin[i]);

    for(int i = 0; i < 3; ++i)
	transform[i][3] += 0.5 * maxlocalextent + drand48() * (domain_extent[i] - maxlocalextent);
}

void TransformedExtent::apply(float x[3], float y[3])
{
    for(int i = 0; i < 3; ++i)
	y[i] = transform[i][0] * x[0] + transform[i][1] * x[1] + transform[i][2] * x[2] + transform[i][3];
}

bool TransformedExtent::collides(const TransformedExtent a, const float tol)
{
    float s[3], e[3];

    for(int c = 0; c < 3; ++c)
    {
	s[c] = max(xmin[c], a.xmin[c]);
	e[c] = min(xmax[c], a.xmax[c]);

	if (s[c] - e[c] >= tol)
	    return false;
    }

    return true;
}

bool Checker::check(TransformedExtent& ex)
{
    int imin[3], imax[3];

    for (int d=0; d<3; d++)
    {
	imin[d] = floor(ex.xmin[d] / h[d]) + 1;
	imax[d] = floor(ex.xmax[d] / h[d]) + 1;
    }

    for (int i=imin[0]; i<=imax[0]; i++)
	for (int j=imin[1]; j<=imax[1]; j++)
	    for (int k=imin[2]; k<=imax[2]; k++)
	    {
		const int icell = i * n[1] * n[2] + j * n[2] + k;

		bool good = true;

		for (auto rival : data[icell])
		    good &= !ex.collides(rival, safetymargin);

		if (!good)
		    return false;
	    }

    return true;
}

void Checker::add(TransformedExtent& ex)
{
    int imin[3], imax[3];

    for (int d=0; d<3; ++d)
    {
	imin[d] = floor(ex.xmin[d] / h[d]) + 1;
	imax[d] = floor(ex.xmax[d] / h[d]) + 1;
    }

    bool good = true;

    for (int i=imin[0]; i<=imax[0]; i++)
	for (int j=imin[1]; j<=imax[1]; j++)
	    for (int k=imin[2]; k<=imax[2]; k++)
	    {
		const int icell = i * n[1]*n[2] + j * n[2] + k;
		data[icell].push_back(ex);
	    }
}


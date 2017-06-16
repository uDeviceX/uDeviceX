#include <mpi.h>
#include <cassert>
#include "common.h"

#include "ply.h"

namespace ply
{
void write(const char *fname, const Mesh m)
{
#ifndef WRITE_BINARY
    FILE * f = fopen(fname, "w");

    assert(f != NULL);
    
    fprintf(f, "ply\n");
    fprintf(f, "format ascii 1.0\n");
    fprintf(f, "element vertex %d\n", m.nv);
    fprintf(f, "property float x\n");
    fprintf(f, "property float y\n");
    fprintf(f, "property float z\n");
    fprintf(f, "element face %d\n", m.nt);
    fprintf(f, "property list int int vertex_index\n");
    fprintf(f, "end_header\n");
    
    for (int i = 0; i < m.nv; ++i)
    fprintf(f, "%f %f %f\n", m.vv[3*i + 0], m.vv[3*i + 1], m.vv[3*i + 2]);
    
    for (int i = 0; i < m.nt; ++i)
    fprintf(f, "3 %d %d %d\n", m.tt[3*i + 0], m.tt[3*i + 1], m.tt[3*i + 2]);
    
    fclose(f);
#else
    FILE * f = fopen(fname, "wb");

    assert(f != NULL);

    {
        char header[1024];
        sprintf(header,
                "ply\n"
                "format binary_little_endian 1.0\n"
                "element vertex %d\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "element face %d\n"
                "property list int int vertex_index\n"
                "end_header\n",
                m.nv, m.nt);

        fwrite(header, sizeof(char), strlen(header), f);
    }

    fwrite(m.vv, sizeof(float), 3 * m.nv, f);

    int *ibuf = new int[4 * m.nt];
        
    for (int i = 0; i < m.nt; ++i)
    {
        ibuf[4*i + 0] = 3;
        ibuf[4*i + 1] = m.tt[3*i + 0];
        ibuf[4*i + 2] = m.tt[3*i + 1];
        ibuf[4*i + 3] = m.tt[3*i + 2];
    }
        
    fwrite(ibuf, sizeof(int), 4 * m.nt, f);

    delete[] ibuf;
    
    fclose(f);
#endif
}

static bool prop(const char *pname, const char *str)
{
    const int l1 = strlen(pname);
    const int l2 = strlen(str);

    if (l1 > l2) return false;
    for (int i = 0; i < l1; ++i) if (pname[i] != str[i]) return false;
    return true;
}

void read(const char *fname, Mesh *m)
{
    FILE *f = fopen(fname, "r");

    if (f == NULL)
    {
        fprintf(stderr, "error. could not read <%s>\nexiting...\n", fname);
        exit(1);
    }

    int l = 0;
    m->nt = m->nv = -1;

#define BUFSIZE 256 // max number of chars per line
#define MAXLINES 64 // max number of line for header

    // https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html#Stringizing
#define xstr(s) str(s)
#define str(s) #s
        
    while (l++ < MAXLINES)
    {
        char cbuf[BUFSIZE + 1] = {0}; // + 1 for \0

        const int checker = fscanf(f, " %[^\n]" xstr(BUFSIZE) "c", cbuf);

        if (checker != 1)
        {
            fprintf(stderr, "Something went wrong reading <%s>\n", fname);
            exit(1);
        }

        int ibuf;
        if (sscanf(cbuf, "element vertex %d", &ibuf) == 1) m->nv = ibuf;
        else if (sscanf(cbuf, "element face %d", &ibuf) == 1) m->nt = ibuf;
        else if (prop("end_header", cbuf)) break;
    }

    if (l >= MAXLINES || m->nt == -1 || m->nv == -1)
    {
        printf("Something went wrong, did not catch end_header\n");
        exit(1);
    }
        
    m->tt = new int[3 * m->nt];
    m->vv = new float[3 * m->nv];

    for (int i = 0; i < m->nv; ++i)
    fscanf(f, "%f %f %f\n",
           m->vv + 3*i + 0,
           m->vv + 3*i + 1,
           m->vv + 3*i + 2);

    for (int i = 0; i < m->nt; ++i)
    fscanf(f, "%*d %d %d %d\n",
           m->tt + 3*i + 0,
           m->tt + 3*i + 1,
           m->tt + 3*i + 2);
        
    fclose(f);
}
}


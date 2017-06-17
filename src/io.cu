#include <sys/stat.h>

#ifndef NO_H5
#include <hdf5.h>
#endif

#include <sstream>
#include <vector>
#include <conf.h>
#include "conf.common.h"
#include "m.h"     /* MPI */
#include "l/m.h"
#include "common.h"
#include "io.h"

bool H5FieldDump::directory_exists = false;

void _write_bytes(const void * const ptr, const int nbytes32, MPI_File f) {
    MPI_Offset base;
    MC(MPI_File_get_position(f, &base));
    MPI_Offset offset = 0, nbytes = nbytes32;
    MC(MPI_Exscan(&nbytes, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart));
    MPI_Status status;
    MC(MPI_File_write_at_all(f, base + offset, ptr, nbytes, MPI_CHAR, &status));
    MPI_Offset ntotal = 0;
    MC(MPI_Allreduce(&nbytes, &ntotal, 1, MPI_OFFSET, MPI_SUM, m::cart) );
    MC(MPI_File_seek(f, ntotal, MPI_SEEK_CUR));
}

void ply_dump(const char * filename,
              int *mesh_indices, const int ninstances, const int ntriangles_per_instance,
              Particle * _particles, int nvertices_per_instance) {
    std::vector<Particle> particles(_particles, _particles + ninstances * nvertices_per_instance);
    int NPOINTS = 0;
    const int n = particles.size();
    l::m::Reduce(&n, &NPOINTS, 1, MPI_INT, MPI_SUM, 0, m::cart) ;
    const int ntriangles = ntriangles_per_instance * ninstances;
    int NTRIANGLES = 0;
    l::m::Reduce(&ntriangles, &NTRIANGLES, 1, MPI_INT, MPI_SUM, 0, m::cart) ;
    MPI_File f;
    MPI_File_open(m::cart, filename,
                  MPI_MODE_WRONLY |  MPI_MODE_CREATE,
                  MPI_INFO_NULL, &f);
    MPI_File_set_size (f, 0);

    std::stringstream ss;
    if (m::rank == 0) {
        ss <<  "ply\n";
        ss <<  "format binary_little_endian 1.0\n";
        ss <<  "element vertex " << NPOINTS << "\n";
        ss <<  "property float x\nproperty float y\nproperty float z\n";
        ss <<  "property float u\nproperty float v\nproperty float w\n";
        ss <<  "element face " << NTRIANGLES << "\n";
        ss <<  "property list int int vertex_index\n";
        ss <<  "end_header\n";
    }
    std::string content = ss.str();
    _write_bytes(content.c_str(), content.size(), f);
    int L[3] = { XS, YS, ZS };
    for(int i = 0; i < n; ++i)
    for(int c = 0; c < 3; ++c)
    particles[i].r[c] += L[c] / 2 + m::coords[c] * L[c];

    _write_bytes(&particles.front(), sizeof(Particle) * n, f);
    int poffset = 0;
    MPI_Exscan(&n, &poffset, 1, MPI_INTEGER, MPI_SUM, m::cart);
    std::vector<int> buf;
    for(int j = 0; j < ninstances; ++j)
    for(int i = 0; i < ntriangles_per_instance; ++i) {
        int primitive[4] = { 3,
                             poffset + nvertices_per_instance * j + mesh_indices[3 * i + 0],
                             poffset + nvertices_per_instance * j + mesh_indices[3 * i + 1],
                             poffset + nvertices_per_instance * j + mesh_indices[3 * i + 2] };
        buf.insert(buf.end(), primitive, primitive + 4);
    }
    _write_bytes(&buf.front(), sizeof(int) * buf.size(), f);
    MPI_File_close(&f);
}

void H5FieldDump::_xdmf_header(FILE * xmf) {
    fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
    fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
    fprintf(xmf, " <Domain>\n");
}

void H5FieldDump::_xdmf_grid(FILE * xmf,
                             const char * const h5path, const char * const *channelnames, int nchannels) {
    fprintf(xmf, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
    fprintf(xmf, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n",
            1 + (int)globalsize[2], 1 + (int)globalsize[1], 1 + (int)globalsize[0]);

    fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(xmf, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(xmf, "        %e %e %e\n", 0.0, 0.0, 0.0);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");

    const float h = 1;
    fprintf(xmf, "        %e %e %e\n", h, h, h);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Geometry>\n");

    for(int ichannel = 0; ichannel < nchannels; ++ichannel) {
        fprintf(xmf, "     <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", channelnames[ichannel]);
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n",
                (int)globalsize[2], (int)globalsize[1], (int)globalsize[0]);
        fprintf(xmf, "        %s:/%s\n", h5path, channelnames[ichannel]);
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");
    }
    fprintf(xmf, "   </Grid>\n");
}

void H5FieldDump::_xdmf_epilogue(FILE * xmf) {
    fprintf(xmf, " </Domain>\n");
    fprintf(xmf, "</Xdmf>\n");
}

void H5FieldDump::_write_fields(const char * const path2h5,
                                const float * const channeldata[],
                                const char * const * const channelnames, const int nchannels) {
#ifndef NO_H5
    hid_t plist_id_access = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id_access, m::cart, MPI_INFO_NULL);

    hid_t file_id = H5Fcreate(path2h5, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_access);
    H5Pclose(plist_id_access);

    const int L[3] = { XS, YS, ZS };
    hsize_t globalsize[4] = {(hsize_t) m::dims[2] * L[2], (hsize_t) m::dims[1] * L[1], (hsize_t) m::dims[0] * L[0], 1};
    hid_t filespace_simple = H5Screate_simple(4, globalsize, NULL);

    for(int ichannel = 0; ichannel < nchannels; ++ichannel)
    {
        hid_t dset_id = H5Dcreate(file_id, channelnames[ichannel], H5T_NATIVE_FLOAT, filespace_simple,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);

        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        hsize_t start[4]  = { (hsize_t) m::coords[2] * L[2], (hsize_t) m::coords[1] * L[1], (hsize_t) m::coords[0] * L[0], 0};
        hsize_t extent[4] = { (hsize_t) L[2], (hsize_t) L[1], (hsize_t) L[0],  1};
        hid_t filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, extent, NULL);

        hid_t memspace = H5Screate_simple(4, extent, NULL);
        herr_t status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, channeldata[ichannel]);

        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Pclose(plist_id);
        H5Dclose(dset_id);
    }

    H5Sclose(filespace_simple);
    H5Fclose(file_id);

    if (!m::rank)
    {
        char wrapper[256];
        sprintf(wrapper, "%s.xmf", std::string(path2h5).substr(0, std::string(path2h5).find_last_of(".h5") - 2).data());

        FILE * xmf = fopen(wrapper, "w");

        _xdmf_header(xmf);
        _xdmf_grid(xmf, std::string(path2h5).substr(std::string(path2h5).find_last_of("/") + 1).c_str(),
                   channelnames, nchannels);
        _xdmf_epilogue(xmf);

        fclose(xmf);
    }
#endif // NO_H5
}

H5FieldDump::H5FieldDump() : last_idtimestep(0) {
    const int L[3] = { XS, YS, ZS };

    for(int c = 0; c < 3; ++c)
    globalsize[c] = L[c] * m::dims[c];
}

void H5FieldDump::dump_scalarfield(float *data,
                                   const char *channelname) {
    char path2h5[512];
    sprintf(path2h5, "h5/%s.h5", channelname);
    _write_fields(path2h5, &data, &channelname, 1);
}

void H5FieldDump::dump(Particle *p, int n) {
#ifndef NO_H5
    static int id = 0; /* dump id */
     
    const int ncells = XS * YS * ZS;
    std::vector<float> rho(ncells), u[3];
    for(int c = 0; c < 3; ++c)
    u[c].resize(ncells);
    for(int i = 0; i < n; ++i) {
        const int cellindex[3] = {
            max(0, min(XS - 1, (int)(floor(p[i].r[0])) + XS / 2)),
            max(0, min(YS - 1, (int)(floor(p[i].r[1])) + YS / 2)),
            max(0, min(ZS - 1, (int)(floor(p[i].r[2])) + ZS / 2))
        };
        const int entry = cellindex[0] + XS * (cellindex[1] + YS * cellindex[2]);
        rho[entry] += 1;
        for(int c = 0; c < 3; ++c) u[c][entry] += p[i].v[c];
    }

    for(int c = 0; c < 3; ++c)
    for(int i = 0; i < ncells; ++i)
    u[c][i] = rho[i] ? u[c][i] / rho[i] : 0;

    const char * names[] = { "density", "u", "v", "w" };
    if (!directory_exists) {
        if (m::rank == 0)
        mkdir("h5", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        directory_exists = true;
        MC(l::m::Barrier(m::cart));
    }

    char filepath[512];
    sprintf(filepath, "h5/flowfields-%04d.h5", id++);
    float * data[] = { rho.data(), u[0].data(), u[1].data(), u[2].data() };
    _write_fields(filepath, data, names, 4);
#endif // NO_H5
}

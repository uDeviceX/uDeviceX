/*
 * [t]riangle cells
 * Cells are organised as follows:
 * ids - array of triangle ids
 *       id = mid * nt + tid
 * starts - array s.t. start[cid] has first index of ids in the cell cid
 * counts - array containing the number of triangles overlapping that cell
 */

void build_tcells_hst(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids);
void build_tcells_dev(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids);

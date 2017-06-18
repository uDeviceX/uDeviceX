namespace phalo {
__constant__ int cellpackstarts[27];
struct CellPackSOA {
    int *start, *count, *scan, size;
    bool enabled;
};
__constant__ CellPackSOA cellpacks[26];
struct SendBagInfo {
    int *start_src, *count_src, *start_dst;
    int bagsize, *scattered_entries;
    Particle *dbag, *hbag;
};

__constant__ SendBagInfo baginfos[26];
__constant__ int *srccells[26 * 2], *dstcells[26 * 2];
}

typedef Sarray<int,  26> int26;
typedef Sarray<int,  27> int27;
typedef Sarray<int*, 26> intp26;
typedef Sarray<Particle*, 26> Particlep26;

void gather_cells(const int *start, const int *count, const int27 fragstarts, const int26 fragnc,
                  const int ncells, /**/ intp26 fragstr, intp26 fragcnt, intp26 fragcum) {
    if (ncells) dev::count<<<k_cnf(ncells)>>>(fragstarts, start, count, fragstr, fragcnt);
    dev::scan<32><<<26, 32 * 32>>>(fragnc, fragcnt, /**/ fragcum);
}

void copycells(const int27 fragstarts, const int ncells, const intp26 srccells, /**/ intp26 dstcells) {
    if (ncells) dev::copycells<<<k_cnf(ncells)>>>(fragstarts, srccells, /**/ dstcells);
}
  
// void pack(const int27 fragstarts, const Particle *pp, int n) {
//     if (ncells)
//     k_halo::fill_all<<<(ncells + 1) / 2, 32>>>(fragstarts, pp, frag::np,
//                                                frag::str, frag::cnt, frag::cum,
//                                                frag::capacity, frag::ii, frag::pp);
// }

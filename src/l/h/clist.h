namespace l { namespace clist {
namespace t = thrust;
template<typename T> T * ptr(t::device_vector<T>& v) { return raw_pointer_cast(v.data()); }

void h(float * const pp, int np,
       const int xcells, const int ycells, const int zcells,
       const float xstart, const float ystart, const float zstart,
       int* start, int* cnt) {
  t::device_vector<int> codes(np), pids(np);
  d::pid2code<<<k_cnf(np)>>>
    (ptr(codes), ptr(pids), np, pp, make_int3(xcells, ycells, zcells), make_float3(xstart, ystart, zstart));
  sort_by_key(codes.begin(), codes.end(), pids.begin());
  {
    int sz = 6*np;
    t::device_vector<float> tmp(sz);
    t::copy(t::device_ptr<float>(pp), t::device_ptr<float>(pp + sz), tmp.begin());

    d::gather<<<k_cnf(sz)>>>(ptr(tmp), ptr(pids), pp, sz);
    cudaPeekAtLastError();
  }

  int ncells = xcells * ycells * zcells;
  t::device_vector<int> cids(ncells), cidsp1(ncells);

  d::cids<<<k_cnf(ncells)>>>(ptr(cids), ncells, 0,  make_int3(xcells, ycells, zcells));
  d::cids<<<k_cnf(ncells)>>>(ptr(cidsp1), ncells, 1, make_int3(xcells, ycells, zcells) );

  t::lower_bound(codes.begin(), codes.end(), cids.begin(), cids.end(), t::device_ptr<int>(start));
  t::lower_bound(codes.begin(), codes.end(), cidsp1.begin(), cidsp1.end(), t::device_ptr<int>(cnt));

  d::count<<<k_cnf(ncells)>>>(start, cnt, ncells);

  /*  if (order != NULL)
      copy(pids.begin(), pids.end(), order); */
}
}}

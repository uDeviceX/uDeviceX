namespace clist {
template<typename T> T * ptr(device_vector<T>& v) { return raw_pointer_cast(v.data()); }

void build(float * const pp, int np,
	   const int xcells, const int ycells, const int zcells,
	   const float xstart, const float ystart, const float zstart,
	   int* start, int* cnt) {
  device_vector<int> codes(np), pids(np);
  k_clist::pid2code<<<k_cnf(np)>>>
    (ptr(codes), ptr(pids), np, pp, make_int3(xcells, ycells, zcells), make_float3(xstart, ystart, zstart));
  sort_by_key(codes.begin(), codes.end(), pids.begin());
  {
    int sz = 6*np;
    device_vector<float> tmp(sz);
    copy(device_ptr<float>(pp), device_ptr<float>(pp + sz), tmp.begin());

    k_clist::gather<<<k_cnf(sz)>>>(ptr(tmp), ptr(pids), pp, sz);
    cudaPeekAtLastError();
  }

  int ncells = xcells * ycells * zcells;
  device_vector<int> cids(ncells), cidsp1(ncells);

  k_clist::cids<<<k_cnf(ncells)>>>(ptr(cids), ncells, 0,  make_int3(xcells, ycells, zcells));
  k_clist::cids<<<k_cnf(ncells)>>>(ptr(cidsp1), ncells, 1, make_int3(xcells, ycells, zcells) );

  lower_bound(codes.begin(), codes.end(), cids.begin(), cids.end(), device_ptr<int>(start));
  lower_bound(codes.begin(), codes.end(), cidsp1.begin(), cidsp1.end(), device_ptr<int>(cnt));

  k_clist::count<<<k_cnf(ncells)>>>(start, cnt, ncells);

  /*  if (order != NULL)
      copy(pids.begin(), pids.end(), order); */
}
} /* namespace clist */

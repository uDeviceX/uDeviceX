namespace cnt {
int nsolutes;
DeviceBuffer<uchar4> *subindices;
scan::Work ws;
DeviceBuffer<int> *cellsentries, *cellsstart, *cellscount;
rnd::KISS* local_trunk;
}

const int sz = XS*YS*ZS + 16;

int no; /* number of objects */
DeviceBuffer<uchar4> *indexes;
DeviceBuffer<int> *entries;

namespace g {
scan::Work ws;
int *starts, *counts;
}
rnd::KISS* rgen;

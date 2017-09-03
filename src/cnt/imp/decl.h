const int sz = XS*YS*ZS + 16;

int no; /* number of objects */
DeviceBuffer<uchar4> *indexes;
DeviceBuffer<int> *entries;

scan::Work ws;
namespace g {
int *starts, *counts;
}
rnd::KISS* rgen;

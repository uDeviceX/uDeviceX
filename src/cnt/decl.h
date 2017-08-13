namespace cnt {
const int sz = XS*YS*ZS + 16;

int no; /* number of objects */
DeviceBuffer<uchar4> *indexes;
scan::Work ws;
int *starts, *counts;
DeviceBuffer<int> *entries;
rnd::KISS* rgen;
}

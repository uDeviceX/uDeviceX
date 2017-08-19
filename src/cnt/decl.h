namespace cnt {
const int sz = XS*YS*ZS + 16;

int no; /* number of objects */
uchar4 *indexes;
DeviceBuffer<int> *entries;

scan::Work ws;
int *starts, *counts;
rnd::KISS* rgen;
}

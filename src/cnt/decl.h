namespace cnt {
int no; /* number of objects */
DeviceBuffer<uchar4> *indexes;
scan::Work ws;
DeviceBuffer<int> *entries, *starts, *counts;
rnd::KISS* rgen;
}

namespace cnt {
int no; /* number of objects */
DeviceBuffer<uchar4> *indexes;
scan::Work ws;
int *starts;
DeviceBuffer<int> *entries, *counts;
rnd::KISS* rgen;
}

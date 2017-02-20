namespace Contact {
  int nsolutes;
  SimpleDeviceBuffer<uchar4> *subindices;
  SimpleDeviceBuffer<unsigned char> *compressed_cellscount;
  SimpleDeviceBuffer<int> *cellsentries, *cellsstart, *cellscount;
  Logistic::KISS* local_trunk;
}

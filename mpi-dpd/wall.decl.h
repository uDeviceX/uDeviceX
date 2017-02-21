namespace Wall {
  Logistic::KISS* trunk;

  int solid_size;
  float4 *solid4;
  cudaArray *arrSDF;
  CellLists *wall_cells;
}

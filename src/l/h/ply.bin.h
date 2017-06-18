namespace l { namespace ply {
void write(const char *fname, const Mesh m)
{
  FILE * f = fopen(fname, "wb");

  assert(f != NULL);

  {
    char header[1024];
    sprintf(header,
	    "ply\n"
	    "format binary_little_endian 1.0\n"
	    "element vertex %d\n"
	    "property float x\n"
	    "property float y\n"
	    "property float z\n"
	    "element face %d\n"
	    "property list int int vertex_index\n"
	    "end_header\n",
	    m.nv, m.nt);

    fwrite(header, sizeof(char), strlen(header), f);
  }

  fwrite(m.vv, sizeof(float), 3 * m.nv, f);

  int *ibuf = new int[4 * m.nt];

  for (int i = 0; i < m.nt; ++i)
    {
      ibuf[4*i + 0] = 3;
      ibuf[4*i + 1] = m.tt[3*i + 0];
      ibuf[4*i + 2] = m.tt[3*i + 1];
      ibuf[4*i + 3] = m.tt[3*i + 2];
    }

  fwrite(ibuf, sizeof(int), 4 * m.nt, f);

  delete[] ibuf;

  fclose(f);
}
}}

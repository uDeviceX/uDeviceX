struct BulkData {
    float4  *zip0;  /* xyzouvwo xyzouvwo xyzouvwo ...       */
    ushort4 *zip1;  /* xyzo xyzo xyzo...  in half precision */
    rnd::KISS *rnd; /* random generator per timestep        */
};


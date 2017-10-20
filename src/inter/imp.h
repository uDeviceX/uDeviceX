namespace inter {
void freeze(sdf::Quants qsdf, flu::Quants *qflu, rig::Quants *qrig, rbc::Quants *qrbc);
void create_walls(int maxn, sdf::Quants qsdf, flu::Quants* qflu, wall::Quants *qwall);
}

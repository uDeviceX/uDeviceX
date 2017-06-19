# restart

##  file format

Restart stores simulation variables of solvent, wall, rbcs and rigid bodies under the following file naming:

```
strt/[code]/[XXX].[YYY].[ZZZ]/[ttt].[ext]
```
where:
- `[code]` is `flu` (solvent), `wall`, `rbc` or `rig` (rigid bodies)
- `[XXX].[YYY].[ZZZ]` are the coordinates of the processor (no dir if single processor)
- `[ttt]` is the id of the restart
- `[ext]` is the extension of the file


# restart

##  file format

Restart stores simulation variables of solvent, wall, , rbcs and rigid bodies under the following file format:

```
strt/[code]/[XXX].[YYY].[ZZZ]/[ttt].[ext]
```
wher:
- `[code]` is `flu` (solvent), `wall`, `rbc` or `rig` (rigid bodies)
- `[XXX].[YYY].[ZZZ]` are the coordinates of the processor (no dir if single processor)
- `[ttt]` is the id of the restart
- `[ext]` is the extension of the file

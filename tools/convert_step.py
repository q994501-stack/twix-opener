"""STEP -> model-data.js для index.html.

Запуск:  pip install cadquery-ocp numpy
         python tools/convert_step.py Сборка_правая_часть.stp model-data.js

Каждое тело (solid) из STEP становится отдельной деталью. Вершины и индексы квантуются в uint16
и кодируются в base64 внутри обычного <script>, поэтому страница работает и без сервера.
Цвета деталей лежат в словаре COL (RGB из STEP, по порядку тел); для новых тел по умолчанию серый.
Названия и описания деталей задаются в index.html, объект INFO.
"""
import sys, json, base64
import numpy as np
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_REVERSED
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

COL = {0: (.282, .282, .282), 1: (1, 0, 0), 2: (1, 1, 0), 3: (0, .502, 1), 4: (.4, .8, 0), 5: (.6, .8, 1),
       6: (0, .502, 1), 7: (1, 0, 1), 8: (.4, 0, .8)}
GREY = (.5647, .5647, .5647)

src, dst = sys.argv[1], sys.argv[2]
rd = STEPControl_Reader(); rd.ReadFile(src); rd.TransferRoots()
shape = rd.OneShape()
BRepMesh_IncrementalMesh(shape, 0.08, False, 0.25, True)   # допуск 0.08 мм

meta, blob = [], bytearray()
pad = lambda: blob.extend(b'\0' * (-len(blob) % 4))
ex, i = TopExp_Explorer(shape, TopAbs_SOLID), 0
while ex.More():
    V, F = [], []
    fe = TopExp_Explorer(ex.Current(), TopAbs_FACE)
    while fe.More():
        f = TopoDS.Face(fe.Current()); loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is not None:
            tr, off = loc.Transformation(), len(V)
            V += [(p.X(), p.Y(), p.Z()) for p in (tri.Node(k).Transformed(tr) for k in range(1, tri.NbNodes() + 1))]
            rev = f.Orientation() == TopAbs_REVERSED
            for k in range(1, tri.NbTriangles() + 1):
                a, b, c = (n + off - 1 for n in tri.Triangle(k).Get())
                F.append((a, c, b) if rev else (a, b, c))
        fe.Next()
    V, F = np.array(V), np.array(F)
    t = V[F]; vol = abs(np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)
    lo, hi = V.min(0), V.max(0); step = float((hi - lo).max() / 65535)
    pad(); po = len(blob); blob += np.round((V - lo) / step).astype('<u2').tobytes()
    pad(); io = len(blob); blob += F.astype('<u2').ravel().tobytes()
    meta.append(dict(i=i, col=COL.get(i, GREY), lo=[round(x, 4) for x in lo], step=step, nv=len(V), ni=int(F.size),
                     po=po, io=io, vol=round(vol / 1000, 2), center=[round(x, 2) for x in (lo + hi) / 2],
                     bb=[[round(x, 2) for x in lo], [round(x, 2) for x in hi]]))
    i += 1; ex.Next()
pad()
open(dst, 'w', encoding='utf-8').write('window.TWIX_MODEL=' + json.dumps(
    dict(parts=meta, bin=base64.b64encode(bytes(blob)).decode()), ensure_ascii=False) + ';\n')
print(f'{i} деталей -> {dst}')

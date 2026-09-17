"""
把 body+hair 的 OBJ 序列预处理成二进制缓存，供 viewer.html 直接 fetch（零文本解析）。

前提：每帧顶点数 / 拓扑恒定（已验证）。所有帧顶点连续拼成单个大文件，
浏览器一次拉取后按 帧号*顶点数 偏移切片，省掉上千次小请求和 JS 文本解析。

输出（到 --out 目录，默认 tools/web/cache）：
  hair_pos.f32   594 帧 × Hv×3 float32   发丝顶点
  body_pos.f32   594 帧 × Bv×3 float32   身体顶点
  hair_idx.u32   线段索引（逐帧不变，发丝展开成 pair）
  body_idx.u32   三角面索引（逐帧不变）
  hair_col.f32   发根→发梢渐变色（逐帧不变，Hv×3）
并写 tools/web/frames.json（二进制清单）。

用法：
  python tools/preprocess_strands_bin.py \
      --hair_dir .../gt --body_dir .../target
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "tools" / "web"

ROOT_COLOR = np.array([0.55, 0.27, 0.07], dtype=np.float32)
TIP_COLOR = np.array([0.95, 0.80, 0.35], dtype=np.float32)


def natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', Path(p).stem)]


def read_verts(path):
    """只读 v 行 → (N,3) float32。"""
    if Path(path).suffix == ".npz":
        data = np.load(path)
        rods = np.asarray(data["rods"], dtype=np.float32)
        lengths = np.asarray(data["lengths"], dtype=np.int32)
        # Solver NPZ uses metres; hair/body OBJ and the viewer use centimetres.
        return np.concatenate([rods[i, :int(length)] for i, length in enumerate(lengths)]) * 100.0
    out = []
    for ln in open(path):
        if ln.startswith('v '):
            out.append(ln.split()[1:4])
    return np.asarray(out, dtype=np.float32)


def read_hair_topology(path):
    """读 l 行 → (线段 pair 索引 u32, 顶点渐变色 f32, 发丝长度列表)。"""
    if Path(path).suffix == ".npz":
        data = np.load(path)
        strand_lens = np.asarray(data["lengths"], dtype=np.int32).tolist()
        strands = []
        start = 0
        for length in strand_lens:
            strands.append(list(range(start, start + int(length))))
            start += int(length)
        nv = start
    else:
        strands = []
        nv = 0
        for ln in open(path):
            if ln.startswith('v '):
                nv += 1
            elif ln.startswith('l '):
                strands.append([int(x) - 1 for x in ln.split()[1:]])
    seg = []
    col = np.zeros((nv, 3), dtype=np.float32)
    strand_lens = []
    for s in strands:
        n = len(s)
        strand_lens.append(n)
        for k in range(n):
            t = k / (n - 1) if n > 1 else 0.0
            col[s[k]] = ROOT_COLOR * (1 - t) + TIP_COLOR * t
        for k in range(n - 1):
            seg.append(s[k]); seg.append(s[k + 1])
    return np.asarray(seg, dtype=np.uint32), col, strand_lens


def read_face_index(path):
    """读 f 行 → 三角面索引 u32（兼容 f a/b/c，四边以上扇形三角化）。"""
    idx = []
    for ln in open(path):
        if ln.startswith('f '):
            p = [int(t.split('/')[0]) - 1 for t in ln.split()[1:]]
            for k in range(1, len(p) - 1):
                idx.append(p[0]); idx.append(p[k]); idx.append(p[k + 1])
    return np.asarray(idx, dtype=np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hair_dir", required=True)
    ap.add_argument("--body_dir", required=True)
    ap.add_argument("--out", default=str(WEB / "cache"))
    ap.add_argument("--manifest", default="frames.json",
                    help="清单文件名（写到 tools/web/ 下），多数据集用不同名以便切换")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    # 按帧号严格匹配 hair frame_X ↔ body_X（hair 可能缺帧，绝不能用排序位置配对）
    hair_paths = list(Path(args.hair_dir).glob("frame_*.obj"))
    hair_paths.extend(Path(args.hair_dir).glob("frame_*.npz"))
    # 部分正式目录会同时保存同一帧的 OBJ 与 NPZ。按 frame id 去重，优先使用
    # solver 的 NPZ，避免 viewer 将一段序列重复播放两次。
    hair_by_frame = {}
    for path in sorted(hair_paths, key=lambda p: p.suffix != ".npz"):
        fid = int(re.search(r'(\d+)', path.stem).group())
        hair_by_frame.setdefault(fid, path)
    hair_paths = sorted(hair_by_frame.values(), key=natural_key)
    hair_paths = [path for path in hair_paths
                  if (lambda fid: fid >= args.start_frame
                      and (args.end_frame < 0 or fid <= args.end_frame))(
                          int(re.search(r'(\d+)', path.stem).group()))]
    hair_paths = hair_paths[::max(args.stride, 1)]
    body_dir = Path(args.body_dir)
    pairs = []
    for hp in hair_paths:
        fid = int(re.search(r'(\d+)', hp.stem).group())
        bp = body_dir / f"body_{fid}.obj"
        if bp.exists():
            pairs.append((fid, hp, bp))
    n_skip = len(hair_paths) - len(pairs)
    if n_skip:
        print(f"警告：{n_skip} 帧的 body 缺失，已跳过")
    ids = [p[0] for p in pairs]
    hair_paths = [p[1] for p in pairs]
    body_paths = [p[2] for p in pairs]
    F = len(pairs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 拓扑 + 颜色（首帧一次）
    hair_idx, hair_col, strand_lens = read_hair_topology(hair_paths[0])
    body_idx = read_face_index(body_paths[0])
    hair_idx.tofile(out / "hair_idx.u32")
    body_idx.tofile(out / "body_idx.u32")
    hair_col.tofile(out / "hair_col.f32")
    Hv = hair_col.shape[0]
    Bv = read_verts(body_paths[0]).shape[0]
    print(f"帧数={F}  发丝顶点={Hv} 线段={len(hair_idx)//2}  身体顶点={Bv} 三角={len(body_idx)//3}")

    # 顶点序列（连续大文件）
    hair_all = np.empty((F, Hv, 3), dtype=np.float32)
    body_all = np.empty((F, Bv, 3), dtype=np.float32)
    for i in range(F):
        hair_all[i] = read_verts(hair_paths[i])
        body_all[i] = read_verts(body_paths[i])
        if (i + 1) % 50 == 0:
            print(f"  解析 {i + 1}/{F}")
    hair_all.tofile(out / "hair_pos.f32")
    body_all.tofile(out / "body_pos.f32")

    try:
        rel = out.resolve().relative_to(WEB.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"--out 必须位于 {WEB} 下，viewer 才能访问") from exc
    manifest = {
        "binary": True, "fps": args.fps, "frame_count": F, "ids": ids,
        "hair": {"pos": f"{rel}/hair_pos.f32", "idx": f"{rel}/hair_idx.u32",
                 "col": f"{rel}/hair_col.f32", "vcount": Hv, "segcount": len(hair_idx),
                 "strand_lens": strand_lens},
        "body": {"pos": f"{rel}/body_pos.f32", "idx": f"{rel}/body_idx.u32",
                 "vcount": Bv, "icount": len(body_idx)},
    }
    (WEB / args.manifest).write_text(json.dumps(manifest, ensure_ascii=False))
    mb = (hair_all.nbytes + body_all.nbytes) / 1e6
    print(f"完成。二进制总量 {mb:.0f} MB → {out}")
    print(f"清单：{WEB / args.manifest}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Helpers for the motion-correction / gain / CTF test (slurm/final_mc.sh). Numbers only.
#   bin2 IN.mrc OUT.mrc            2x2x2 average of a cubic map, pixel size doubled
#   invert IN.mrc OUT.mrc          1/x of a gain image (zeros stay zero)
#   f32 IN.mrc OUT.mrc             rewrite any MRC (e.g. float16 micrographs) as float32
#   gaincheck GAIN.mrc MOVIE.tiff...  flatness of the summed raw counts after each way of
#                                  applying the gain (multiply/divide, 4 orientations, none)
#   orient MOVIE.tiff MIC.mrc      which orientation of a RELION micrograph matches the raw
#                                  TIFF frames as tifffile reads them
#   median STAR COL...             median of named columns in the last loop of a STAR file
import sys

import numpy as np

DT = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}


def read_mrc(p):
    with open(p, 'rb') as f:
        h = bytearray(f.read(1024))
        i = np.frombuffer(bytes(h), dtype=np.int32)
        nx, ny, nz, mode = (int(v) for v in i[:4])
        f.seek(1024 + int(i[23]))
        a = np.fromfile(f, dtype=DT[mode], count=nx * ny * nz).reshape(nz, ny, nx)
    apix = float(np.frombuffer(bytes(h[40:44]), dtype=np.float32)[0]) / max(int(i[7]), 1)
    return a.astype(np.float32), h, apix


def write_mrc(p, a, h, apix):
    a = np.ascontiguousarray(a, dtype=np.float32)
    nz, ny, nx = a.shape
    h = bytearray(h)
    iw = np.frombuffer(h, dtype=np.int32)  # view into h
    fw = np.frombuffer(h, dtype=np.float32)
    iw[0:3] = (nx, ny, nz)
    iw[3] = 2
    iw[7:10] = (nx, ny, nz)
    fw[10:13] = (nx * apix, ny * apix, nz * apix)
    fw[19:22] = (a.min(), a.max(), a.mean())
    iw[23] = 0
    fw[54] = a.std()
    with open(p, 'wb') as f:
        f.write(bytes(h))
        a.tofile(f)


def bin2(src, dst):
    a, h, apix = read_mrc(src)
    n = a.shape[0] // 2
    b = a[:2 * n, :2 * n, :2 * n].reshape(n, 2, n, 2, n, 2).mean(axis=(1, 3, 5))
    write_mrc(dst, b, h, apix * 2)
    print('bin2: %d -> %d px, %.4f -> %.4f A, range %.3f..%.3f' % (
        a.shape[0], n, apix, apix * 2, b.min(), b.max()))


def invert(src, dst):
    a, h, apix = read_mrc(src)
    b = np.where(a != 0, 1.0 / np.where(a != 0, a, 1), 0)
    write_mrc(dst, b, h, apix)
    print('invert: mean %.4f -> %.4f' % (a.mean(), b.mean()))


def f32(src, dst):
    a, h, apix = read_mrc(src)
    write_mrc(dst, a, h, apix)


def tiff_sum(p):
    import tifffile
    with tifffile.TiffFile(p) as t:
        s = None
        for pg in t.pages:
            fr = pg.asarray().astype(np.float32)
            s = fr if s is None else s + fr
    return s


def hp_score(c, border=100):
    from scipy import ndimage
    c = c[border:-border, border:-border].astype(np.float64)
    hp = c - ndimage.gaussian_filter(c, 16)
    return hp.std() / c.mean()


def gaincheck(gain, movies):
    from multiprocessing import Pool
    g = read_mrc(gain)[0][0]
    with Pool(min(12, len(movies))) as pool:
        sums = pool.map(tiff_sum, movies)
    s = np.sum(sums, axis=0)
    if s.shape != g.shape:
        print('gaincheck: movie %s vs gain %s, cannot compare' % (s.shape, g.shape))
        return
    print('gaincheck: %d movies, mean counts per pixel %.1f, Poisson floor %.4f' % (
        len(movies), s.mean(), 1 / np.sqrt(s.mean())))
    gs = np.where(g > 0, g, 1)
    tr = {'as-is': gs, 'flip-ud': gs[::-1], 'flip-lr': gs[:, ::-1], 'rot180': gs[::-1, ::-1]}
    rows = [(hp_score(s), 'no gain')]
    for k, t in tr.items():
        rows.append((hp_score(s * t), 'multiply ' + k))
        rows.append((hp_score(s / t), 'divide ' + k))
    for sc, k in sorted(rows):
        print('  roughness %.4f  %s' % (sc, k))


def bin8(a):
    ny, nx = (d // 8 * 8 for d in a.shape)
    return a[:ny, :nx].reshape(ny // 8, 8, nx // 8, 8).mean(axis=(1, 3))


def orient(movie, mic):
    from scipy import ndimage
    s = bin8(tiff_sum(movie))
    m = bin8(read_mrc(mic)[0][0])
    if s.shape != m.shape:
        print('orient: tiff %s vs micrograph %s' % (s.shape, m.shape))
        return

    def z(a):
        a = a - ndimage.gaussian_filter(a, 20)
        a = a[8:-8, 8:-8]
        return (a - a.mean()) / a.std()

    zs = z(s)
    for k, t in (('as-is', m), ('flip-ud', m[::-1]), ('flip-lr', m[:, ::-1]),
                 ('rot180', m[::-1, ::-1])):
        print('  orient corr %.3f  micrograph %s vs tiff' % ((z(t) * zs).mean(), k))


def median(star, cols):
    names, rows, inloop = [], [], False
    for line in open(star):
        t = line.split()
        if line.startswith('loop_'):
            names, rows, inloop = [], [], True
        elif inloop and line.startswith('_'):
            names.append(t[0])
        elif inloop and t and not line.startswith(('data_', '#')):
            rows.append(t)
    out = []
    for c in cols:
        i = names.index(c)
        v = np.array([float(r[i]) for r in rows])
        out.append('%s %.4g' % (c.replace('_rln', ''), np.median(v)))
    print('n %d  median %s' % (len(rows), '  '.join(out)))


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'bin2':
        bin2(*args)
    elif cmd == 'invert':
        invert(*args)
    elif cmd == 'f32':
        f32(*args)
    elif cmd == 'gaincheck':
        gaincheck(args[0], args[1:])
    elif cmd == 'orient':
        orient(*args)
    elif cmd == 'median':
        median(args[0], args[1:])
    else:
        sys.exit('unknown command')

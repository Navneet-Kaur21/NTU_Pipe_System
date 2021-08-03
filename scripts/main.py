# https://github.com/dranjan/python-plyfile
# https://rdrr.io/github/geomorphR/geomorph/man/read.ply.html
# https://www.programcreek.com/python/?CodeExample=read+ply

from argparse import ArgumentParser

import numpy
from mayavi import mlab

from plyfile import PlyData

def main():
    parser = ArgumentParser()
    parser.add_argument('ply_filename')

    args = parser.parse_args()

    mlab.figure(bgcolor=(0, 0, 0))
    plot(PlyData.read(args.ply_filename))
    mlab.show()

def plot(ply):
    vertex = ply['vertex']

    (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))

    mlab.points3d(x, y, z, color=(1, 1, 1), mode='point')

    if 'face' in ply:
        tri_idx = ply['face']['vertex_indices']
        triangles = numpy.vstack(tri_idx)
        mlab.triangular_mesh(x, y, z, triangles, color=(1, 0, 0.4), opacity=0.5)

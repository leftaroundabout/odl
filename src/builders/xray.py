# -*- coding: utf-8 -*-
"""
xray.py -- helper functions for X-ray tomography

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from math import pi

from RL.datamodel import ugrid as ug
from RL.datamodel import gfunc as gf
from RL.geometry import curve as crv
from RL.geometry import source as src
from RL.geometry import sample as spl
from RL.geometry import detector as det
from RL.geometry import geometry as geo
from RL.operator.projector import Projector, BackProjector
from RL.utility.utility import InputValidationError, errfmt


def xray_ct_parallel_3d_projector(geometry, backend='astra_cuda'):

    # FIXME: this construction is probably only temporary. Think properly
    # about how users would construct projectors
    return Projector(xray_ct_parallel_projection_3d, geometry,
                     backend=backend)


def xray_ct_parallel_3d_backprojector(geometry, backend='astra_cuda'):

    # FIXME: this construction is probably only temporary. Think properly
    # about how users would construct projectors
    return BackProjector(xray_ct_parallel_backprojection_3d, geometry,
                         backend=backend)


def xray_ct_parallel_geom_3d(spl_grid, det_grid, axis, angles=None,
                             rotating_sample=True, **kwargs):
    """
    Create a 3D parallel beam geometry for X-ray CT with a flat detector.

    Parameters
    ----------
    spl_grid: ugrid.Ugrid
        3D grid for the sample domain
    det_grid: ugrid.Ugrid
        2D grid for the detector domain
    axis: int or array-like
        rotation axis; if integer, interpreted as corresponding standard
        unit vecor
    angles: array-like, optional
        specifies the rotation angles
    rotating_sample: boolean, optional
        if True, the sample rotates, otherwise the source-detector system

    Keyword arguments
    -----------------
    init_rotation: matrix-like or float
        initial rotation of the sample; if float, ???

    Returns
    -------
    out: geometry.Geometry
        the new parallel beam geometry
    """

    spl_grid = ug.ugrid(spl_grid)
    det_grid = ug.ugrid(det_grid)
    if not spl_grid.dim == 3:
        raise InputValidationError(spl_grid.dim, 3, 'spl_grid.dim')
    if not det_grid.dim == 2:
        raise InputValidationError(det_grid.dim, 2, 'det_grid.dim')

    if angles is not None:
        angles = np.array(angles)

    init_rotation = kwargs.get('init_rotation', None)

    if rotating_sample:
        # TODO: make axis between source and detector flexible; now: -x axis
        direction = (1., 0., 0.)
        src_loc = (-1., 0., 0.)
        source = src.ParallelRaySource(direction, src_loc)
        sample = spl.RotatingGridSample(spl_grid, axis, init_rotation,
                                        angles=angles, **kwargs)
        det_loc = (1., 0., 0.)
        detector = det.FlatDetectorArray(det_grid, det_loc)
    else:
        src_circle = crv.Circle3D(1., axis, angles=angles, axes_map='tripod')
        source = src.ParallelRaySource((1, 0, 0), src_circle)

        sample = spl.FixedSample(spl_grid)

        det_circle = crv.Circle3D(1., axis, angle_shift=pi, angles=angles,
                                  axes_map='tripod')
        detector = det.FlatDetectorArray(det_grid, det_circle)
    return geo.Geometry(source, sample, detector)


def xray_ct_parallel_projection_3d(geometry, vol_func, backend='astra_cuda'):

    if backend == 'astra':
        proj_func = _xray_ct_par_fp_3d_astra(geometry, vol_func,
                                             use_cuda=False)
    elif backend == 'astra_cuda':
        proj_func = _xray_ct_par_fp_3d_astra(geometry, vol_func,
                                             use_cuda=True)
    else:
        raise NotImplementedError(errfmt('''\
        Only `astra` and `astra_cuda` backends supported'''))

    return proj_func


def _xray_ct_par_fp_3d_astra(geom, vol, use_cuda=True):

    import astra as at

    print('compute forward projection')

    # FIXME: we assume fixed sample rotating around z axis for now
    # TODO: include shifts (volume and detector)
    # TODO: allow custom detector grid (e.g. only partial projection)

    # Initialize volume geometry and data and wrap it into a data3d object

    # ASTRA uses a different axis labeling. We need to cycle x->y->z->x
    astra_vol = vol.fvals.swapaxes(0, 1).swapaxes(0, 2)
    astra_vol_geom = at.create_vol_geom(vol.shape)
    astra_vol_id = at.data3d.create('-vol', astra_vol_geom, astra_vol)

    # Create the ASTRA algorithm config
    if use_cuda:
        astra_algo_conf = at.astra_dict('FP3D_CUDA')
    else:
        # TODO: slice into 2D forward projections
        raise NotImplementedError('No CPU 3D forward projection available.')

    # Initialize detector geometry

    # Since ASTRA assumes voxel size (1., 1., 1.), some scaling and adaption
    # of tilt angles is necessary

    # FIXME: assuming z axis tilt
    det_grid = geom.detector.grid
    # FIXME: treat case when no discretization is given
    angles = geom.sample.angles
#    print('old angles: ', astra_angles)

    # FIXME: lots of repetition in the following lines
    # TODO: this should be written without angles, just using direction
    # vectors
    a, b, c = 1. / vol.spacing
#    print('a, b, c = ', a, b, c)
    if a != b:
        proj_fvals = np.empty((det_grid.shape[0], det_grid.shape[1],
                               len(astra_angles)))
        scaling_factors, px_scaling, astra_angles = _astra_scaling(a, b,
                                                                   angles,
                                                                   'fp')
        for i, ang in enumerate(astra_angles):
            print('[{}] ang: '.format(i), ang)
            old_dir = np.array((cos(ang), sin(ang), 0))
            print('[{}] old dir: '.format(i), old_dir)
            if old_dir[0] >= 0:
                if old_dir[1] >= 0:
                    quadrant = 1
                else:
                    quadrant = 4
            else:
                if old_dir[1] >= 0:
                    quadrant = 2
                else:
                    quadrant = 3

            print('[{}] quadrant: '.format(i), quadrant)
            scaled_dir = np.diag((a, b, c)).dot(old_dir)
            print('[{}] scaled dir: '.format(i), scaled_dir)
            norm_dir = norm(scaled_dir)
            new_dir = scaled_dir / norm_dir
            print('[{}] norm_dir: '.format(i), norm_dir)
            print('[{}] new dir: '.format(i), new_dir)

            # Use the smaller of the two values for stability
            # acos maps to [0,pi] -> subtract pi for quadrants 3 and 4
            # asin maps to [-pi/2,pi/2] -> subtract from +-pi in quadrants 2/3
            if abs(new_dir[0]) > abs(new_dir[1]):
                astra_angles[i] = asin(new_dir[1])
                if quadrant == 2:
                    astra_angles[i] = pi - astra_angles[i]
                elif quadrant == 3:
                    astra_angles[i] = -pi - astra_angles[i]
            else:
                astra_angles[i] = acos(new_dir[0])
                if quadrant in (3, 4):
                    astra_angles[i] -= pi

            print('[{}] new ang: '.format(i), astra_angles[i])

            scaling_factor = 1. / norm_dir
            print('[{}] scaling: '.format(i), scaling_factor)

            old_perp = np.cross((0, 0, 1), old_dir)
            print('[{}] old perp: '.format(i), old_perp)
            scaled_perp = np.diag((a, b, c)).dot(old_perp)
            print('[{}] scaled perp: '.format(i), scaled_perp)
            norm_perp = norm(scaled_perp)
            print('[{}] norm_perp: '.format(i), norm_perp)
            astra_pixel_spacing = det_grid.spacing.copy()
            print('[{}] orig det px size: '.format(i), astra_pixel_spacing)
            astra_pixel_spacing *= (norm_perp, c)
            print('[{}] scaled det px size: '.format(i), astra_pixel_spacing)

            # ASTRA lables detector axes as 'rows, columns', so we need to swap
            # axes 0 and 1
            # We must project one by one since pixel sizes vary
            astra_proj_geom = at.create_proj_geom('parallel3d',
                                                  astra_pixel_spacing[0],
                                                  astra_pixel_spacing[1],
                                                  det_grid.shape[1],
                                                  det_grid.shape[0],
                                                  astra_angles[i])
            # Some wrapping code
            astra_proj_id = at.data3d.create('-sino', astra_proj_geom)

            # Configure and create the algorithm
            astra_algo_conf['VolumeDataId'] = astra_vol_id
            astra_algo_conf['ProjectionDataId'] = astra_proj_id
            astra_algo_id = at.algorithm.create(astra_algo_conf)

            # Run it and remove afterwards
            at.algorithm.run(astra_algo_id)
            at.algorithm.delete(astra_algo_id)

            # Get the projection data. ASTRA creates an (nrows, 1, ncols)
            # array, so we need to squeeze and swap to get (nx, ny)
            proj_fvals[:, :, i] = at.data3d.get(
                astra_proj_id).squeeze().swapaxes(0, 1)
            proj_fvals[:, :, i] *= scaling_factor
    else:
        # ASTRA lables detector axes as 'rows, columns', so we need to swap
        # axes 0 and 1
        astra_proj_geom = at.create_proj_geom('parallel3d',
                                              astra_pixel_spacing[0],
                                              astra_pixel_spacing[1],
                                              det_grid.shape[1],
                                              det_grid.shape[0],
                                              astra_angles)

        # Some wrapping code
        astra_proj_id = at.data3d.create('-sino', astra_proj_geom)

        # Configure and create the algorithm
        astra_algo_conf['VolumeDataId'] = astra_vol_id
        astra_algo_conf['ProjectionDataId'] = astra_proj_id
        astra_algo_id = at.algorithm.create(astra_algo_conf)

        # Run it and remove afterwards
        at.algorithm.run(astra_algo_id)
        at.algorithm.delete(astra_algo_id)

        # Get the projection data. ASTRA creates an (nrows, ntilts, ncols)
        # array, so we need to cycle to the right to get (nx, ny, ntilts)
        proj_fvals = at.data3d.get(astra_proj_id)
        proj_fvals = proj_fvals.swapaxes(1, 2).swapaxes(0, 1)

        scaling_factor = 1. / a
        astra_pixel_spacing = det_grid.spacing * (a, c)

        proj_fvals *= scaling_factor

    # Create the projection grid function and return it
    proj_spacing = np.ones(3)
    proj_spacing[:-1] = det_grid.spacing
    proj_func = gf.Gfunc(proj_fvals, spacing=proj_spacing)

    return proj_func


def _astra_scaling(a, b, angles, op):

    from math import pi, sin, asin, cos, acos
    from scipy.linalg import norm

    # Forward projection:
    # - The diagonal matrix D^(-1) = diag(a, b) with D = diag(voxel_sizes)
    #   scales voxel sizes to 1 in the plane of rotation
    # - For tilt angle t, the projection direction w = (cos(t), sin(t))
    #   is changed to W = (a * cos(t), b * sin(t)) / norm with
    #   norm = sqrt(a^2 * cos(t)^2 + b^2 * sin(t)^2)
    # - A vector in w^perp can be written as v = s * w0^perp with
    #   w0 = (-sin(t), cos(t))
    # - This vector is scaled to V = s * Norm * W0 with
    #   W0 = (-a * sin(t), b * cos(t)) / Norm,
    #   Norm = sqrt(a^2 * sin(t)^2 + b^2 * cos(t)^2)
    # - Thus, the detector grid must be scaled with Norm
    # - Finally, the reparametrization produces a factor of norm^(-1)
    #
    #   The factors, scalings and angles are returned
    #
    # Backprojection:
    # - The angles are calculated as above, but with D instead of D^(-1)
    # - An integration weight on the sphere is calculated as

    scaling_factors = np.empty_like(angles)
    px_scaling = np.empty_like(angles)
    new_angles = np.empty_like(angles)

    for i, ang in enumerate(angles):
        print('[{}] ang: '.format(i), ang)
        old_dir = np.array((cos(ang), sin(ang)))
        print('[{}] old dir: '.format(i), old_dir)
        if old_dir[0] >= 0:
            if old_dir[1] >= 0:
                quadrant = 1
            else:
                quadrant = 4
        else:
            if old_dir[1] >= 0:
                quadrant = 2
            else:
                quadrant = 3

        print('[{}] quadrant: '.format(i), quadrant)
        scaled_dir = np.diag((a, b)).dot(old_dir)
        print('[{}] scaled dir: '.format(i), scaled_dir)
        norm_dir = norm(scaled_dir)
        new_dir = scaled_dir / norm_dir
        print('[{}] norm_dir: '.format(i), norm_dir)
        print('[{}] new dir: '.format(i), new_dir)

        # Use the smaller of the two values for stability
        # acos maps to [0,pi] -> subtract pi for quadrants 3 and 4
        # asin maps to [-pi/2,pi/2] -> subtract from +-pi in quadrants 2/3
        if abs(new_dir[0]) > abs(new_dir[1]):
            new_angles[i] = asin(new_dir[1])
            if quadrant == 2:
                new_angles[i] = pi - new_angles[i]
            elif quadrant == 3:
                new_angles[i] = -pi - new_angles[i]
        else:
            new_angles[i] = acos(new_dir[0])
            if quadrant in (3, 4):
                new_angles[i] -= pi

        print('[{}] new ang: '.format(i), new_angles[i])

        scaling_factors[i] = 1. / norm_dir
        print('[{}] scaling: '.format(i), scaling_factors[i])

        old_perp = np.array([-old_dir[1], old_dir[0]])
        print('[{}] old perp: '.format(i), old_perp)
        scaled_perp = np.diag((a, b)).dot(old_perp)
        print('[{}] scaled perp: '.format(i), scaled_perp)
        norm_perp = norm(scaled_perp)
        print('[{}] norm_perp: '.format(i), norm_perp)
        px_scaling[i] = norm_perp

    return scaling_factors, px_scaling, new_angles

        

def xray_ct_parallel_backprojection_3d(geometry, proj_func,
                                       backend='astra_cuda'):

    if backend == 'astra':
        vol_func = _xray_ct_par_bp_3d_astra(geometry, proj_func,
                                            use_cuda=False)
    elif backend == 'astra_cuda':
        vol_func = _xray_ct_par_bp_3d_astra(geometry, proj_func,
                                            use_cuda=True)
    else:
        raise NotImplementedError(errfmt('''\
        Only `astra` and `astra_cuda` backends supported'''))

    return vol_func


def _xray_ct_par_bp_3d_astra(geom, proj_, use_cuda=True):

    import astra as at

    print('compute backprojection')
    # FIXME: we assume fixed sample rotating around z axis for now
    # TODO: include shifts (volume and detector)
    # TODO: allow custom volume grid (e.g. partial backprojection)

    # Initialize projection geometry and data

    # Detector pixel spacing must be scaled with volume (y,z) spacing
    # since ASTRA assumes voxel size 1
    # FIXME: assuming z axis tilt
    vol_grid = geom.sample.grid
    astra_pixel_spacing = proj_.spacing[:-1] / vol_grid.spacing[1:]

    # FIXME: treat case when no discretization is given
    astra_angles = geom.sample.angles

    # ASTRA assumes a (nrows, ntilts, ncols) array. We must cycle axes to
    # the left and swap detector axes in the geometry.
    astra_proj = proj_.fvals.swapaxes(0, 2).swapaxes(0, 1)
    astra_proj_geom = at.create_proj_geom('parallel3d',
                                          astra_pixel_spacing[0],
                                          astra_pixel_spacing[1],
                                          proj_.shape[1],
                                          proj_.shape[0],
                                          astra_angles)

    # Initialize volume geometry
    astra_vol_geom = at.create_vol_geom(vol_grid.shape)

    # Some wrapping code
    astra_vol_id = at.data3d.create('-vol', astra_vol_geom)
    astra_proj_id = at.data3d.create('-sino', astra_proj_geom, astra_proj)

    # Create the ASTRA algorithm
    if use_cuda:
        astra_algo_conf = at.astra_dict('BP3D_CUDA')
    else:
        # TODO: slice into 2D forward projections
        raise NotImplementedError('No CPU 3D backprojection available.')

    astra_algo_conf['ReconstructionDataId'] = astra_vol_id
    astra_algo_conf['ProjectionDataId'] = astra_proj_id
    astra_algo_id = at.algorithm.create(astra_algo_conf)

    # Run it and remove afterwards
    at.algorithm.run(astra_algo_id)
    at.algorithm.delete(astra_algo_id)

    # Get the volume data and cycle back axes to translate from ASTRA axes
    # convention
    vol_fvals = at.data3d.get(astra_vol_id)
    vol_fvals = vol_fvals.swapaxes(0, 2).swapaxes(0, 1)

    # Create the volume grid function and return it
    vol_func = gf.Gfunc(vol_fvals, spacing=vol_grid.spacing)

    return vol_func

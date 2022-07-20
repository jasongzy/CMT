from __future__ import absolute_import, division, print_function, unicode_literals, with_statement

__version__ = "2.1.1"

try:
    __POINTNET2_SETUP__
except NameError:
    __POINTNET2_SETUP__ = False

if not __POINTNET2_SETUP__:
    from pointnet2 import utils

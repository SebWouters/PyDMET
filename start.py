'''
    PyDMET: a python implementation of density matrix embedding theory
    Copyright (C) 2014 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import numpy as np
import GroundState2D
import LDOS2D
import LDDR2D

case2run = 3

if ( case2run == 1 ):

    HubbardU = 4.0
    Filling, Energy1, Energy4 = GroundState2D.CalculateEnergies( HubbardU )

    print "Hubbard U =",HubbardU
    print "Filling ; Energy/site for 1x1 cluster ; Energy/site for 2x2 cluster"
    print np.column_stack((Filling, Energy1, Energy4))

if ( case2run == 2 ):

    HubbardU   = 10.0
    eta        = 0.05
    lowerbound = -3.5
    upperbound = 3.5
    stepsize   = 0.2
    # Local density of states spectral function
    Omegas, LDOS = LDOS2D.CalculateLDOS( HubbardU, lowerbound, upperbound, stepsize, eta )
    print np.column_stack((Omegas, LDOS))

if ( case2run == 3 ):

    HubbardU   = 10.0
    eta        = 0.05
    maxomega   = 14.0
    stepsize   = 0.2
    # Local density density response spectral function
    Omegas, LDDR = LDDR2D.CalculateLDDR( HubbardU, maxomega, stepsize, eta )
    print np.column_stack((Omegas, LDDR))
    

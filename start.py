'''
    PyDMET: a python implementation of density matrix embedding theory
    Copyright (C) 2014, 2015 Sebastian Wouters
    
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
import LDOS2D
import LDOS1D
import LDDR2D

case2run = 3

if ( case2run == 1 ):

    HubbardU = 10.0
    Omegas   = np.array([ 1.23456 ])
    eta      = 0.2
    #Local density of states spectral function
    LDOS = LDOS2D.CalculateLDOS( HubbardU, Omegas, eta )
    print np.column_stack((Omegas, LDOS))

if ( case2run == 2 ):

    HubbardU = 12.0
    Omegas   = np.array([ 1.23456 ])
    eta      = 0.2
    # Local density density response spectral function
    LDDR = LDDR2D.CalculateLDDR( HubbardU, Omegas, eta )
    print np.column_stack((Omegas, LDDR))
    
if ( case2run == 3 ):

    HubbardU = 8.0
    Omegas   = np.array([ 1.23456 ])
    eta      = 0.05
    #Local density of states spectral function
    LDOS = LDOS1D.CalculateLDOS( HubbardU, Omegas, eta )
    print np.column_stack((Omegas, LDOS))
    

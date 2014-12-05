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

HubbardU = 4.0
Filling, Energy1, Energy4 = GroundState2D.CalculateEnergies( HubbardU )

print "Hubbard U =",HubbardU
print "Filling ; Energy/site for 1x1 cluster ; Energy/site for 2x2 cluster"
print np.column_stack((Filling, Energy1, Energy4))

'''import numpy as np
import HubbardDMET

lattice_size = np.array( [24, 48], dtype=int )
cluster_size = np.array( [2, 2], dtype=int )
Nelectrons   = ( np.prod( lattice_size ) * 16 ) / 24
antiPeriodic = True

HubbardU = 4.0
theDMET = HubbardDMET.HubbardDMET( lattice_size, cluster_size, HubbardU, antiPeriodic )
#EnergyPerSite = theDMET.SolveGroundState( Nelectrons )

orb_i = 0
omega = 2.345
eta   = 0.05
numBathOrbs = 6
toSolve = 'R'
EnergyPerSite, GFvalue = theDMET.SolveResponse( Nelectrons, orb_i, omega, eta, numBathOrbs, toSolve )'''


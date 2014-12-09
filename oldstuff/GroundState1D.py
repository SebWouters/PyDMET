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
import HubbardDMET

lattice_size = np.array( [ 4 ], dtype=int )
cluster_size = np.array( [ 1 ], dtype=int )
Nelectrons   = np.prod( lattice_size ) # Half-filling
antiPeriodic = True

Uvalues  = []
Energies = []

for HubbardU in np.arange( -8.0, 8.1, 1.0 ):

    theDMET = HubbardDMET.HubbardDMET( lattice_size, cluster_size, HubbardU, antiPeriodic )
    EnergyPerSite, umatrix = theDMET.SolveGroundState( Nelectrons )
    Uvalues.append( HubbardU )
    Energies.append( EnergyPerSite )

print np.column_stack((Uvalues, Energies))
    

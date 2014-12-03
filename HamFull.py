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

#  This class augments the Hamiltonian interface with the DMET mean-field potential
#  Functions which should be provided for the DMET calculations:
#     getNumOrbitals()
#     getEconst()
#     getTmat( i, j )
#     getVmat( i, j, k, l )
#  For now the Hubbard model is assumed

import numpy as np

class HamFull:

    def __init__( self, Ham, cluster_size, umat ):
        self.Ham  = Ham
        self.Tmat = self.ConstructFullTmat( cluster_size, umat )
        
    def getNumOrbitals( self ):
        return self.Ham.getNumOrbitals()

    def getIsHubbard( self ):
        return self.Ham.getIsHubbard()

    def getEconst( self ):
        return self.Ham.getEconst()
        
    def getTmat( self, i, j ):
        return self.Tmat[i, j]

    def getVmat( self, i, j, k, l ):
        return self.Ham.getVmat(i, j, k, l)

    def ConstructFullTmat( self, cluster_size, umat ):
    
        # Make a copy of the original hopping matrix
        Tmat = np.array( self.Ham.Tmat, copy=True )
        
        # Check that the cluster fits an integer times on the lattice
        steps = np.zeros( [self.Ham.dim], dtype=int )
        for dim in range(0, self.Ham.dim):
            steps[dim] = self.Ham.size[dim] / cluster_size[dim]
            assert( self.Ham.size[dim] % cluster_size[dim] == 0 )
        
        # For each of the shifts of the cluster, copy the umat
        numSteps = np.prod( steps )
        numClusterOrbs = np.prod( cluster_size )
        for count in range(0, numSteps):
            macroshifts = np.zeros( [self.Ham.dim], dtype=int )
            copycount = count
            for dim in range(0, self.Ham.dim):
                macroshifts[dim] = copycount % steps[dim]
                copycount = ( copycount - macroshifts[dim] ) / steps[dim]
            #print "Plaquette from",np.multiply(macroshifts, cluster_size),"to",np.multiply(macroshifts+1, cluster_size)
            for count2 in range(0, numClusterOrbs):
                microshifts = np.zeros( [self.Ham.dim], dtype=int )
                copycount2 = count2
                for dim in range(0, self.Ham.dim):
                    microshifts[dim] = copycount2 % cluster_size[dim]
                    copycount2 = ( copycount2 - microshifts[dim] ) / cluster_size[dim]
                coordinate1 = self.Ham.getLinearCoordinate( np.multiply(macroshifts, cluster_size) + microshifts )
                #print "Element",np.multiply(macroshifts, cluster_size)+microshifts
                for count3 in range(0, numClusterOrbs):
                    microshifts2 = np.zeros( [self.Ham.dim], dtype=int )
                    copycount3 = count3
                    for dim in range(0, self.Ham.dim):
                        microshifts2[dim] = copycount3 % cluster_size[dim]
                        copycount3 = ( copycount3 - microshifts2[dim] ) / cluster_size[dim]
                    coordinate2 = self.Ham.getLinearCoordinate( np.multiply(macroshifts, cluster_size) + microshifts2 )
                    Tmat[ coordinate1, coordinate2 ] = self.Ham.getTmat( coordinate1, coordinate2 ) + umat[ count2, count3 ]
        return Tmat
        
    def printer( self ):
        numOrbs = self.getNumOrbitals()
        for row in range(0, numOrbs):
            for col in range(0, numOrbs):
                element = self.getTmat(row, col)
                if ( element != 0.0 ):
                    print "Tmat[",self.Ham.getLatticeCoordinate(row),",",self.Ham.getLatticeCoordinate(col),"] =",element


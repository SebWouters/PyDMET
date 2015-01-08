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

#  This class augments the Hamiltonian interface with the DMET mean-field potential
#  Functions which should be provided for the DMET calculations:
#     getNumOrbitals()
#     getEconst()
#     getTmat( i, j )
#     getVmat( i, j, k, l )
#  For now the Hubbard model is assumed

import numpy as np

class HamFull:

    def __init__( self, Ham, cluster_size, umat, skew2by2cell ):
        self.Ham = Ham
        if skew2by2cell:
            self.Tmat = self.ConstructFullTmatSkew2by2cell( cluster_size, umat )
        else:
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
        
    def ConstructFullTmatSkew2by2cell( self, cluster_size, umat ):
    
        assert( self.Ham.dim    == 2 )
        assert( cluster_size[0] == 2 )
        assert( cluster_size[1] == 2 )
        latticevec1 = np.array( [1, 0], dtype=int )
        latticevec2 = np.array( [1, 1], dtype=int )
        numClusterOrbs = np.prod(cluster_size)
        
        # Tiling has unit vectors (1,0) and (1,1)
        # with counting convention
        #    ______
        #   / 2 3 /
        # / 0 1 /
        # ------
    
        # Make a copy of the original hopping matrix
        Tmat = np.array( self.Ham.Tmat, copy=True )
        
        # Check that the cluster fits an integer times on the lattice
        assert( self.Ham.size[0] % cluster_size[0] == 0 )
        assert( self.Ham.size[1] % cluster_size[1] == 0 )
        steps = np.array( [ self.Ham.size[0] / cluster_size[0], self.Ham.size[1] / cluster_size[1] ], dtype=int )
        
        #totalNumOrbs = self.Ham.getNumOrbitals()
        #check = np.zeros( [totalNumOrbs], dtype=int )
        
        # For each of the shifts of the cluster, copy the umat
        for stepx in range(0, steps[0]):
            for stepy in range(0, steps[1]):
                corner = cluster_size[0] * stepx * latticevec1 + cluster_size[1] * stepy * latticevec2
                for count1 in range(0, numClusterOrbs):
                    orbx = np.zeros( [self.Ham.dim], dtype=int )
                    orbx[0] = ( corner[0] + (count1 + 1) / 2 ) % self.Ham.size[0]
                    orbx[1] = ( corner[1] + (count1    ) / 2 ) % self.Ham.size[1]
                    coordinate1 = self.Ham.getLinearCoordinate( orbx )
                    #check[coordinate1] = 1
                    for count2 in range(0, numClusterOrbs):
                        orby = np.zeros( [self.Ham.dim], dtype=int )
                        orby[0] = ( corner[0] + (count2 + 1) / 2 ) % self.Ham.size[0]
                        orby[1] = ( corner[1] + (count2    ) / 2 ) % self.Ham.size[1]
                        coordinate2 = self.Ham.getLinearCoordinate( orby )
                        Tmat[ coordinate1, coordinate2 ] = self.Ham.getTmat( coordinate1, coordinate2 ) + umat[ count1, count2 ]
        #assert( np.sum(check) == totalNumOrbs )
        return Tmat
        
    def printer( self ):
        numOrbs = self.getNumOrbitals()
        for row in range(0, numOrbs):
            for col in range(0, numOrbs):
                element = self.getTmat(row, col)
                if ( element != 0.0 ):
                    print "Tmat[",self.Ham.getLatticeCoordinate(row),",",self.Ham.getLatticeCoordinate(col),"] =",element


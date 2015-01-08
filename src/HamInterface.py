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

#  This class defines the original Hamiltonian in the impurity basis
#  Functions which should be provided for the DMET calculations:
#     getNumOrbitals()
#     getEconst()
#     getTmat( i, j )
#     getVmat( i, j, k, l )
#  For now the Hubbard model is assumed

import numpy as np

class HamInterface:

    def __init__( self, lattice_size, HubbardU, antiPeriodic ):
        self.dim     = len( lattice_size )
        self.size    = lattice_size
        self.U       = HubbardU
        self.NumOrbs = np.prod( self.size )
        self.Hubbard = True
        self.antiPer = antiPeriodic # Antiperiodic boundary conditions in 1D and 2D in PRL 109, 186404 (2012)
        self.Tmat    = self.buildTmat2()

    def getNumOrbitals( self ):
        return self.NumOrbs

    def getIsHubbard( self ):
        return self.Hubbard

    def getEconst( self ):
        return 0.0
        
    def getTmat( self, i, j ):
        return self.Tmat[i, j]

    def getVmat( self, i, j, k, l ):
        if ((i==j) and (i==k) and (i==l)):
            return self.U
        else:
            return 0.0
            
    def getLatticeCoordinate( self, i ):
        co = np.zeros([ self.dim ], dtype=int)
        copy_i = i
        for cnt in range(0, self.dim):
            co[ cnt ] = copy_i % self.size[ cnt ]
            copy_i    = ( copy_i - co[ cnt ] ) / self.size[ cnt ]
        assert ( copy_i == 0 )
        return co
        
    def getLinearCoordinate( self, co ):
        linco = co[0]
        factor = self.size[0]
        for cnt in range(1, self.dim):
            linco += factor * co[ cnt ]
            factor *= self.size[ cnt ]
        return linco
        
    def buildTmat2( self ):
        Tmat = np.zeros([ self.NumOrbs , self.NumOrbs ], dtype=float)
        for i in range(0, self.NumOrbs):
            co_i = self.getLatticeCoordinate( i )
            co_j = np.array( co_i, copy=True )
            for dimension in range(0, self.dim):
                # First consider the neighbour at one less
                if ( co_i[ dimension ] == 0 ):
                    co_j[ dimension ] = self.size[ dimension ] - 1
                    j = self.getLinearCoordinate( co_j )
                    if ( self.antiPer ):
                        Tmat[i, j] =  1.0
                    else:
                        Tmat[i, j] = -1.0
                else:
                    co_j[ dimension ] = co_i[ dimension ] - 1
                    j = self.getLinearCoordinate( co_j )
                    Tmat[i, j] = -1.0
                co_j[ dimension ] = co_i[ dimension ] # Restore the original copy
                # Then consider the neighbour at one more
                if ( co_i[ dimension ] == self.size[ dimension ] - 1 ):
                    co_j[ dimension ] = 0
                    j = self.getLinearCoordinate( co_j )
                    if ( self.antiPer ):
                        Tmat[i, j] =  1.0
                    else:
                        Tmat[i, j] = -1.0
                else:
                    co_j[ dimension ] = co_i[ dimension ] + 1
                    j = self.getLinearCoordinate( co_j )
                    Tmat[i, j] = -1.0
                co_j[ dimension ] = co_i[ dimension ] # Restore the original copy
        return Tmat
        
    def printer( self ):
        print "Econstant  =", self.getEconst()
        print "# orbitals =", self.NumOrbs
        for orb1 in range(0, self.NumOrbs):
            for orb2 in range(0, self.NumOrbs):
                element = self.getTmat( orb1, orb2 )
                if ( element != 0.0 ):
                    print "Tmat(",orb1,";",orb2,") =", element
        for orb1 in range(0, self.NumOrbs):
            for orb2 in range(0, self.NumOrbs):
                for orb3 in range(0, self.NumOrbs):
                    for orb4 in range(0, self.NumOrbs):
                        element = self.getVmat( orb1, orb2, orb3, orb4 )
                        if ( element != 0.0 ):
                            print "Vmat(",orb1,";",orb2,";",orb3,";",orb4,") =", element
        

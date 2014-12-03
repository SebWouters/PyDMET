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
        self.Tmat    = self.buildTmat()

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
        
    def getTmatFunction( self, i, j ):
        co_i = self.getLatticeCoordinate( i )
        co_j = self.getLatticeCoordinate( j )
        difference = np.zeros([ self.dim ], dtype=int)
        numDiffer  = 0
        lastDiffer = -1
        neighBours = True
        for cnt in range(0, self.dim):
            if ( co_i[ cnt ] >= co_j[ cnt ] ):
                difference[ cnt ] = co_i[ cnt ] - co_j[ cnt ]
            else:
                difference[ cnt ] = co_j[ cnt ] - co_i[ cnt ]
            if ( difference[ cnt ] == 1 ) or ( difference[ cnt ] == self.size[ cnt ] - 1 ):
                lastDiffer = cnt
                numDiffer += 1
            if ( difference[ cnt ] > 1 ) and ( difference[ cnt ] < self.size[ cnt ] - 1 ):
                neighBours = False
        if ( neighBours == False ) or ( numDiffer == 0 ) or ( numDiffer > 1 ):
            return 0.0
        else: # numDiffer == 1 and neighBours == True
            if ( difference[ lastDiffer ] == 1 ):
                return -1.0
            else:
                if ( self.antiPer ):
                    return 1.0
                else:
                    return -1.0
    
    def buildTmat( self ):
        Tmat = np.zeros([ self.NumOrbs , self.NumOrbs ], dtype=float)
        for i in range(0, self.NumOrbs):
            for j in range(i+1, self.NumOrbs):
                Tmat[i, j] = self.getTmatFunction( i , j )
                Tmat[j, i] = Tmat[i, j]
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
        

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

# This class contains the matrix elements relevant for the DMET impurity + bath problem

import numpy as np

class DMETham:

    def __init__( self, HamOrig, HamAugment, dmetOrbs, impurityOrbs, numImpOrbs, numBathOrbs ):
        self.HamOrig      = HamOrig
        self.HamAugment   = HamAugment
        self.dmetOrbs     = dmetOrbs
        self.impurityOrbs = impurityOrbs
        self.numImpOrbs   = numImpOrbs
        self.numBathOrbs  = numBathOrbs
        self.conversions  = self.ConstructConversion()
        self.Tmat         = self.ConstructTmat()
        
    def getEconst( self ):
        return self.HamOrig.getEconst()

    def getOrigNumOrbitals( self ):
        return self.HamOrig.getNumOrbitals()
        
    def getDmetNumOrbitals( self ):
        return self.numImpOrbs + self.numBathOrbs
        
    def getIsHubbard( self ):
        return self.HamOrig.getIsHubbard()
        
    def getTmatMF( self, i, j ):
        return self.Tmat[i, j]
        
    def getTmatCorr( self, i, j ):
        if ( i < self.numImpOrbs ) and ( j < self.numImpOrbs ):
            return self.HamOrig.getTmat(self.conversions[i], self.conversions[j])
        else:
            return self.Tmat[i, j].real

    def getVmatCorr( self, i, j, k, l ):
        if ( i < self.numImpOrbs ) and ( j < self.numImpOrbs ) and ( k < self.numImpOrbs ) and ( l < self.numImpOrbs ):
            return self.HamOrig.getVmat( self.conversions[i], self.conversions[j], self.conversions[k], self.conversions[l] )
        else:
            return 0.0
            
    def ConstructConversion( self ):
        counter = 0
        conversions = np.zeros( [self.numImpOrbs], dtype=int )
        for count in range(0, len(self.impurityOrbs)):
            if ( self.impurityOrbs[count] == 1 ):
                conversions[ counter ] = count
                counter += 1
        assert( counter == self.numImpOrbs )
        return conversions
        
    def ConstructTmat( self ):
        numCols = self.getDmetNumOrbitals()
        return np.dot( np.dot( self.dmetOrbs[:,0:numCols].T , self.HamAugment.Tmat ) , self.dmetOrbs[:,0:numCols] )
        
    def OverwriteImpurityUmat( self, umat_new ):
        for row in range(0, self.numImpOrbs):
            for col in range(0, self.numImpOrbs):
                self.Tmat[row, col] = self.HamOrig.getTmat(self.conversions[row], self.conversions[col]) + umat_new[row, col]
                
    def printer( self ):
        print self.Tmat
        numCols = self.getDmetNumOrbitals()
        for row in range(0, numCols):
            print "VmatCorr[",row,",",row,",",row,",",row,"] =",self.getVmatCorr(row, row, row, row)
            
        

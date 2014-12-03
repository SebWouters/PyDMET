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

#  This class performs a restricted HF calculation

import HamInterface
import DIIS
import numpy as np
import LinalgWrappers

class RestrictedHF:

    def __init__( self, Ham , Nelec ):
        self.Ham = Ham
        assert ( Nelec % 2 == 0 )
        self.Npairs = Nelec/2
        self.L = Ham.getNumOrbitals()
        assert ( self.L >= self.Npairs )
        assert ( self.L > 0 )
        assert ( self.Npairs > 0 )
        
    def Solve( self, Orbitals=None ):
    
        if ( Orbitals == None ):
            Tmat = np.array( self.Ham.Tmat, copy=True )
            eigvals, Orbitals = LinalgWrappers.SortedEigSymmetric( Tmat )
            if ( eigvals[ self.Npairs ] - eigvals[ self.Npairs-1 ] < 1e-8 ):
                print "WARNING: The single particle gap is zero!"
                assert( eigvals[ self.Npairs ] - eigvals[ self.Npairs-1 ] >= 1e-8 )
                
        DoublyOccOrbs = Orbitals[:, 0:self.Npairs]
        OneRDM = 2 * np.dot( DoublyOccOrbs , DoublyOccOrbs.T )
        
        errorNorm = 1.0
        iteration = 0
        threshold = 1e-8
        maxiter   = 100
        theDIIS = DIIS.DIIS()
        Energy = self.CalcEnergy( OneRDM )
        EnergyPrevious = Energy + 100.0 * threshold
        
        while (( errorNorm > threshold ) or ( abs( EnergyPrevious - Energy ) > threshold )) and ( iteration < maxiter ):
        
            iteration += 1
            EnergyPrevious = Energy
            Fock = self.BuildFock( OneRDM )
            eigvals, Orbitals = LinalgWrappers.SortedEigSymmetric( Fock )
            DoublyOccOrbs = Orbitals[:, 0:self.Npairs]
            OneRDM = 2 * np.dot( DoublyOccOrbs , DoublyOccOrbs.T )
            Energy = self.CalcEnergy( OneRDM )
            error = np.dot( Fock, OneRDM ) - np.dot( OneRDM, Fock )
            errorNorm = np.linalg.norm( error )
            print "RHF :: At iteration", iteration, "the commutator [F,D] has norm", errorNorm, "and the energy is",Energy
            theDIIS.append( error, OneRDM )
            OneRDM = theDIIS.Solve()
        
        return Orbitals
        
    def BuildFock( self, OneRDM ):
        if ( self.Ham.getIsHubbard() ):
            return self.BuildFockHubbard( OneRDM )
        else:
            return self.BuildFockRegular( OneRDM )
        
    def BuildFockHubbard( self, OneRDM ):
        Fock = np.zeros([self.L, self.L], dtype=float)
        for i in range(0, self.L):
            for j in range(0, self.L):
                Fock[i, j] = self.Ham.getTmat(i, j)
            Fock[i, i] += 0.5 * OneRDM[i, i] * self.Ham.getVmat(i, i, i, i)
        return Fock
        
    def BuildFockRegular( self, OneRDM ):
        Fock = np.zeros([self.L, self.L], dtype=float)
        for i in range(0, self.L):
            for j in range(0, self.L):
                temp = 0.0
                for k in range(0, self.L):
                    for l in range(0, self.L):
                        # In phsyics notation V(i, j, k, l) = i(r1) k(r1) j(r2) l(r2) / |r1-r2|
                        temp += OneRDM[k, l] * ( self.Ham.getVmat(i, k, j, l) - 0.5 * self.Ham.getVmat(i, k, l, j) )
                Fock[i, j] = self.Ham.getTmat(i, j) + temp
        return Fock
        
    def CalcEnergy( self, OneRDM ):
        if ( self.Ham.getIsHubbard() ):
            return self.CalcEnergyHubbard( OneRDM )
        else:
            return self.CalcEnergyRegular( OneRDM )
    
    def CalcEnergyHubbard( self, OneRDM ):
        Energy = 0.0
        for i in range(0, self.L):
            for j in range(0, self.L):
                Energy += self.Ham.getTmat(i, j) * OneRDM[i, j]
            Energy += 0.25 * self.Ham.getVmat(i, i, i, i) * OneRDM[i, i] * OneRDM[i, i]
        return Energy
        
    def CalcEnergyRegular( self, OneRDM ):
        Energy = 0.0
        for i in range(0, self.L):
            for j in range(0, self.L):
                temp = 0.0
                for k in range(0, self.L):
                    for l in range(0, self.L):
                        # In phsyics notation V(i, j, k, l) = i(r1) k(r1) j(r2) l(r2) / |r1-r2|
                        temp += OneRDM[k, l] * ( self.Ham.getVmat(i, k, j, l) - 0.5 * self.Ham.getVmat(i, k, l, j) )
                Energy += ( self.Ham.getTmat(i, j) + 0.5 * temp ) * OneRDM[i, j]
        return Energy
        
    

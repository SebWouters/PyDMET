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

#  This class performs the direct inversion of iterative subspaces

import numpy as np

class DIIS:

    def __init__( self, numVecs=8 ):
        self.errors  = []
        self.states  = []
        self.numVecs = numVecs
        
    def append( self, error, state ):
        self.errors.append(error)
        self.states.append(state)
        
        if len( self.errors ) > self.numVecs:
            self.errors.pop(0)
            self.states.pop(0)
        
    def Solve( self ):
        nStates = len( self.errors )
        mat = np.zeros([ nStates+1 , nStates+1 ], dtype=float)
        for cnt in range(0, nStates):
            mat[ cnt, nStates ] = 1.0
            mat[ nStates, cnt ] = 1.0
            for cnt2 in range(cnt, nStates):
                mat[ cnt, cnt2 ] = np.trace( np.dot( self.errors[cnt] , self.errors[cnt2].T ) )
                mat[ cnt2, cnt ] = mat[ cnt, cnt2 ]
        vec = np.zeros([ nStates+1 ], dtype=float)
        vec[ nStates ] = 1.0
        coeff = np.linalg.tensorsolve(mat, vec)
        coeff = coeff[:-1]
        theerror = coeff[0] * self.errors[0]
        thestate = coeff[0] * self.states[0]
        for cnt in range(1, len(coeff)):
            theerror += coeff[cnt] * self.errors[cnt]
            thestate += coeff[cnt] * self.states[cnt]
        RMSerror = np.linalg.norm(theerror)
        #print "DIIS coefficients = ",coeff,"and the remaining estimated error norm =",RMSerror
        return thestate

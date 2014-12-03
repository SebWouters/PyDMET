PyDMET: a python implementation of density matrix embedding theory
==================================================================

Copyright (C) 2014 Sebastian Wouters <sebastianwouters@gmail.com>

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


Building and testing
--------------------

PyDMET requires python, scipy, and libchemps2. libchemps2 can be downloaded
from [github](https://github.com/SebWouters/CheMPS2). Follow the installation
instructions on that page. Once PyCheMPS2 is built, the path can be adjusted
in SolveCorrelatedProblem.py.

Start from start.py and Redo2D_PRL.py; and adjust these files to start using
PyDMET.


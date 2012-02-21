r"""
This module contains the implicit sharpclaw solvers which requires PETSc toolkit
library for the solution of the nonlinear system of ODEs aring from an implicit
time stepping scheme.
"""

# Import modules
################

# Solver superclass
# Solver superclass
from pyclaw.solver import Solver, CFLError
import pyclaw.sharpclaw


# Reconstructor
try:
    # load c-based WENO reconstructor (PyWENO)
    from pyclaw.limiters import reconstruct as recon
except ImportError:
    # load old WENO5 reconstructor
    from pyclaw.limiters import recon

def before_step(solver,solution):
    r"""
    Dummy routine called before each step
    
    Replace this routine if you want to do something before each time step.
    """
    pass


# ============================================================================
#  Generic implicit SharpClaw solver class
# ============================================================================
class ImplicitSharpClawSolver(Solver):
    r"""
    Superclass for all ImplicitSharpClawND solvers.

    Implements implicit time stepping and the basic form of a 
    semi-discrete step (the dq() function).  If another method-of-lines
    solver is implemented in the future, it should be based on this class,
    which then ought to be renamed to something like "MOLSolver".

    .. attribute:: before_step
    
        Function called before each time step is taken.
        The required signature for this function is:
        
        def before_step(solver,solution)

    .. attribute:: lim_type

        Limiter(s) to be used.
        0: No limiting.
        1: TVD reconstruction.
        2: WENO reconstruction.
        ``Default = 2``

    .. attribute:: weno_order

        Order of the WENO reconstruction. From 1st to 17th order (PyWENO)
        ``Default = 5``

    .. attribute:: time_integrator

        Time integrator to be used.
        BEuler: Backward Euler method.
        ``Default = 'BEuler'``

    .. attribute:: char_decomp

        Type of WENO reconstruction.
        0: conservative variables WENO reconstruction (standard).
        1: characteristic-wise WENO reconstruction.
        2: transmission-based WENO reconstruction.
        ``Default = 0``

    .. attribute:: tfluct_solver

        Whether a total fluctuation solver have to be used. If True the function
        that calculates the total fluctuation must be provided.
        ``Default = False``

    .. attribute:: aux_time_dep

        Whether the auxiliary array is time dependent.
        ``Default = False``
    
    .. attribute:: kernel_language

        Specifies whether to use wrapped Fortran routines ('Fortran')
        or pure Python ('Python').  
        ``Default = 'Fortran'``.

    .. attribute:: num_ghost

        Number of ghost cells.
        ``Default = 3``

    .. attribute:: fwave
    
        Whether to split the flux jump (rather than the jump in Q) into waves; 
        requires that the Riemann solver performs the splitting.  
        ``Default = False``

    .. attribute:: cfl_desired

        Desired CFL number.
        ``Default = 2.45``

    .. attribute:: cfl_max

        Maximum CFL number.
        ``Default = 2.50``

    .. attribute:: dq_src

        Whether a source term is present. If it is present the function that 
        computes its contribution must be provided.
        ``Default = None``
    """
    
    # ========================================================================
    #   Initialization routines
    # ========================================================================
    def __init__(self):
        r"""
        Set default options for ImplicitSharpClawSolvers and call the super's 
        __init__().
        """
        self.limiters = [1]
        self.before_step = before_step
        self.lim_type = 2
        self.weno_order = 5
        self.time_integrator = 'BEuler'
        self.char_decomp = 0
        self.tfluct_solver = False
        self.aux_time_dep = False
        self.kernel_language = 'Fortran'
        self.num_ghost = 3
        self.fwave = False
        self.cfl_desired = 2.45
        self.cfl_max = 2.5
        self.dq_src = None
        self._mthlim = self.limiters
        self._method = None
        self.bVec = None
        self.fVec = None
        self.Jac = None
        self.snes = None

        so_name = 'pyclaw.sharpclaw.sharpclaw'+str(self.num_dim)
        self.fmod = __import__(so_name,fromlist=['pyclaw.sharpclaw'])

        
        # Call general initialization function
        super(ImplicitSharpClawSolver,self).__init__()

    # ========== Time stepping routines ======================================

    def initiate(self,solution):
        r"""
        Called before any set of time steps.
        """
        
        # Import modules
        from petsc4py import PETSc
        from numpy import empty 

        state = solution.state
    
        # Set up a DA with the appropriate stencil width.
        state.set_num_ghost(self.num_ghost)

        # Set mthlim
        self.set_mthlim()

        # Create PETSc vectors in charge of containig:
        # bVec: the constant part of the nonlinear algebraic system of equations
        # fVec: nonlinear vector-valued function
        self.bVec    = state.gqVec.duplicate()
        self.fVec    = state.gqVec.duplicate()
        

        # Create Jacobian matrix
        self.Jac     = PETSc.Mat().create()
        self.Jac.setSizes((self.bVec.size,self.bVec.size))
        

        # Create PETSc nonlinear solver
        self.snes    = PETSc.SNES().create()
        self.Jac = state.q_da.getMatrix()
        self.snes.setJacobian(None, self.Jac) 


        # Ought to implement a copy constructor for State
        #self.impsol_stage = State(state.grid)
        #self.impsol_stage.num_eqn             = state.num_eqn
        #self.impsol_stage.num_aux             = state.num_aux
        #self.impsol_stage.aux_global          = state.aux_global
        #self.impsol_stage.t                   = state.t
        #if state.num_aux > 0:
        #    self.impsol_stage.aux          = state.aux
        
        
    def step(self,solution):
        """
        Evolve q over one time step.

        """
        state = solution.state

        self.before_step(self,solution)


        ########################################
        # Compute solution at the new time level
        ########################################
         
        # Set the constant part of the equation and register the function in 
        # charge of computing the nonlinear residual specified by
        # self.time_integrator.
        if self.time_integrator=='BEuler':
            self.set_bVecBE(state)
            self.snes.setFunction(self.nonlinearfunctionBE, self.fVec)
        else:
            raise Exception('Unrecognized time integrator!')


        # Configure the nonlinear solver to use a matrix-free Jacobian
        #self.snes.setUseMF(True)
        self.snes.setUseFD(True)
        self.snes.setFromOptions()

        # Pass additinal properties to SNES.
        self.snes.appctx=(state)

        # Solve the nonlinear problem
        self.snes.solve(self.bVec, state.gqVec)

        from petsc4py import PETSc
        PETSc.Options().setValue('snes_monitor',1)
        PETSc.Options().setValue('ksp_monitor',1)
        PETSc.Options().setValue('snes_converged_reason',1)
        PETSc.Options().setValue('ksp_converged_reason',1)

        #PETSc.Options().setValue('snes_ls_type','basic')
        #PETSc.Options().setValue('ksp_view',1)
        #PETSc.Options().setValue('snes_view',1)
        #PETSc.Options().setValue('log_summary',1)
 

    def set_bVecBE(self,state):
        r"""
        Set the constant part of the nonlinear algebraic system arising from the
        implicit time discretization  specified by self.time_integrator.
        """

        # Set the constant part of the nonlinear algebraic system equal to the 
        # solution at the current time level.
        self.bVec.setArray(state.q)        

        

    def set_mthlim(self):
        self._mthlim = self.limiters
        if not isinstance(self.limiters,list): self._mthlim=[self._mthlim]
        if len(self._mthlim)==1: self._mthlim = self._mthlim * self.num_waves
        if len(self._mthlim)!=self.num_waves:
            raise Exception('Length of solver.limiters is not equal to 1 or to solver.num_waves')


    def set_fortran_parameters(self,state,clawparams,workspace,reconstruct):
        """
        Set parameters for Fortran modules used by SharpClaw.
        The modules should be imported and passed as arguments to this function.

        """
        grid = state.grid
        clawparams.num_dim       = grid.num_dim
        clawparams.lim_type      = self.lim_type
        clawparams.weno_order    = self.weno_order
        clawparams.char_decomp   = self.char_decomp
        clawparams.tfluct_solver = self.tfluct_solver
        clawparams.fwave         = self.fwave
        clawparams.index_capa    = state.index_capa+1

        clawparams.num_waves     = self.num_waves
        clawparams.alloc_clawparams()
        for idim in range(grid.num_dim):
            clawparams.xlower[idim]=grid.dimensions[idim].lower
            clawparams.xupper[idim]=grid.dimensions[idim].upper
        clawparams.dx       =grid.delta
        clawparams.mthlim   =self._mthlim

        maxnx = max(grid.num_cells)+2*self.num_ghost
        workspace.alloc_workspace(maxnx,self.num_ghost,state.num_eqn,self.num_waves,self.char_decomp)
        reconstruct.alloc_recon_workspace(maxnx,self.num_ghost,state.num_eqn,self.num_waves,
                                            clawparams.lim_type,clawparams.char_decomp)



# ============================================================================
#  Implicit SharpClaw 1d Solver Class
# ============================================================================
class ImplicitSharpClawSolver1D(ImplicitSharpClawSolver):
    """
    Implicit SharpClaw solver for one-dimensional problems.
    
    Used to solve 1D hyperbolic systems using WENO reconstruction and implicit 
    time stepping technique.
    """

    def __init__(self):
        
        # Set physical dimensions
        self.num_dim = 1

        # Call superclass __init__
        super(ImplicitSharpClawSolver1D,self).__init__()



    def setup(self,solution):
        r"""
        Perform essential solver setup. This routine must be called before
        solver.step() may be called.

        Set Fortran data structures (for Clawpack). 
        """ 
        
        # Call parent's "setup" function
        self.initiate(solution)

        state = solution.state        
        
        # Set Fortran data structure for the 1D implicit SharpClaw solver
        if(self.kernel_language == 'Fortran'):
            state.set_cparam(self.fmod)
            state.set_cparam(self.rp)
            self.set_fortran_parameters(state,self.fmod.clawparams,self.fmod.workspace,self.fmod.reconstruct)

        self.allocate_bc_arrays(state)

    def teardown(self):
        r"""
        Deallocate F90 module arrays.
        Also delete Fortran objects, which otherwise tend to persist in Python sessions.
        """
        if self.kernel_language=='Fortran':
            self.fmod.clawparams.dealloc_clawparams()
            self.fmod.workspace.dealloc_workspace(self.char_decomp)
            self.fmod.reconstruct.dealloc_recon_workspace(self.fmod.clawparams.lim_type,self.fmod.clawparams.char_decomp)
            del self.fmod


    # ========== Backward Euler time stepping functions =======================
    def nonlinearfunctionBE(self,snes,qin,nlf):
        r"""
        Computes the nonlinear function for the backward Euler scheme.

        :Input:
         - *qin* - Current approximation of the solution at the next time level,
         i.e. solution of the previous nonlinear solver's iteration.
        """

        # Import modules
        import numpy as np
        from numpy import zeros, reshape, empty

        # Get state
        state = snes.appctx

        # Get some quantities used later on.
        num_cells = state.grid.num_cells[0]
        dx = state.grid.delta[0]
        num_ghost = self.num_ghost
        dt = self.dt
        
        # Define and set to zero the ratio between dt and dx 
        dtdx = np.zeros( (num_cells+2*num_ghost) ) + dt/dx

        # Auxbc is set here and not outside of this function because it is 
        # more general. Indeed aux could depend on q which changes at each 
        # nonlinear iteration!
        if state.num_aux>0:
            state.aux = self.auxbc(state)
        else:
            aux = np.empty((state.num_aux,num_cells+2*num_ghost), order='F')

        # Have to do this because of how qbc works...
        state.q = reshape(qin,(state.num_eqn,num_cells),order='F') 
        qapprox = self.qbc
        

        # Import module
        #from sharpclaw1 import flux1

        # Call fortran routine 
        rp1 = self.rp.rp1._cpointer
        ixy = 1
        sd,self.cfl=flux1(qapprox,aux,dt,state.t,ixy,num_cells,num_ghost,num_cells,rp1)
 
        # Compute the nonlinear vector-valued function
        assert sd.flags['F_CONTIGUOUS']
        nlf.setArray(qapprox[:,num_ghost:-num_ghost]-sd[:,num_ghost:-num_ghost])
        

    


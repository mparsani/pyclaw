#!/usr/bin/env python
# encoding: utf-8
def advection(kernel_language='Fortran',iplot=False,htmlplot=False,use_petsc=True,solver_type='sharpclawBE',outdir='./_output'):
    """
    Example python script for solving the 1d advection equation.
    """
    import numpy as np

    #===========================================================================
    # Import libraries
    #===========================================================================
    # Import only petclaw because the implicit LW always uses SNES to solve the
    # nonlinear algebraic system
    if use_petsc:
        import petclaw as pyclaw
    else:
        raise Exception('Implicit SharpClaw solvers require PETSc!!')


    #===========================================================================
    # Setup solver and solver parameters
    #=========================================================================== 
    solver = pyclaw.ImplicitSharpClawSolver1D()

    solver.kernel_language = kernel_language
    from riemann import rp_advection
    solver.num_waves = rp_advection.num_waves
    
    solver.bc_lower[0] = 2
    solver.bc_upper[0] = 2

    solver.time_integrator = 'BEuler'
    solver.cfl_max = 1.1
    solver.cfl_desired = 1.0
    

    x = pyclaw.Dimension('x',0.0,1.0,100)
    domain = pyclaw.Domain(x)
    num_eqn = 1
    state = pyclaw.State(domain,num_eqn)
    state.problem_data['u']=1.

    grid = state.grid
    xc=grid.x.centers
    beta=100; gamma=0; x0=0.75
    state.q[0,:] = np.exp(-beta * (xc-x0)**2) * np.cos(gamma * (xc - x0))

    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir

    claw.tfinal =1.0
    status = claw.run()

    if htmlplot:  pyclaw.plot.html_plot(outdir=outdir)
    if iplot:     pyclaw.plot.interactive_plot(outdir=outdir)

if __name__=="__main__":
    from pyclaw.util import run_app_from_main
    output = run_app_from_main(advection)


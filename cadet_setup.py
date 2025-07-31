import numpy as np


def create_base_system(model, ncomp, init_c):
    """Create a basic CSTR system with inlet, CSTR and outlet"""
    # CSTR
    model.root.input.model.nunits = 3
    
    # Inlet
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = ncomp
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    
    # CSTR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = ncomp
    model.root.input.model.unit_001.init_liquid_volume = 1.0
    model.root.input.model.unit_001.init_c = init_c
    model.root.input.model.unit_001.const_solid_volume = 1.0
    model.root.input.model.unit_001.use_analytic_jacobian = 1
    
    # Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = ncomp

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 1
    model.root.input['return'].unit_000.write_solution_inlet = 1
    model.root.input['return'].unit_000.write_solution_outlet = 1

    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

def setup_solver(model, sim_time=300.0):
    """Configure solver settings"""
    model.root.input.solver.user_solution_times = np.linspace(0, sim_time, 1000)
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, sim_time]
    model.root.input.solver.sections.section_continuity = []
    
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8
    
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-6
    model.root.input.solver.time_integrator.max_steps = 1000000
    model.root.input.solver.consistent_init_mode = 1

def setup_connections( model, ncomp):
    """Connect the units together"""
    # Connections
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, 0.0,  # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1]
        1, 2, -1, -1, 0.0   # [unit_001, unit_002, all components, all components, Q/ m^3*s^-1]
    ]

    # Inlet coefficients - no inflow
    model.root.input.model.unit_000.sec_000.const_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0] * ncomp
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0] * ncomp
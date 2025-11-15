import numpy as np
import matplotlib.pyplot as plt
from cadet import Cadet
import pickle
import os
import shutil

import pandas as pd

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
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
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

def add_reaction_system(model, parameters):
    
    Km_PD, Km_NAD, Km_Lactol, Km_NADH = parameters['KmPD'], parameters['KmNAD'], parameters['KmLactol'], parameters['KmNADH']

    Ki_PD = parameters['KiPD']

    vmax1_per_mg, vmax2_per_mg, vmax3_per_mg = parameters['Vmax1'], parameters['Vmax2'], parameters['Vmax3']

    model.root.input.model.unit_001.reaction_model = 'MICHAELIS_MENTEN'

    # Km values - single reaction, A is the substrate
    model.root.input.model.unit_001.reaction_bulk.mm_kmm = [
        [Km_PD, Km_NAD, 0.0, 0.0, 0.0],
        [0.0, 0.0, Km_Lactol, Km_NADH, 0.0],
        [0.0, Km_NAD, Km_Lactol, 0.0, 0.0]
    ]

    # Competitive inhibition constants - 3D array [reaction][substrate][inhibitor]
    model.root.input.model.unit_001.reaction_bulk.mm_ki_c = [
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],   # PD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NAD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # LTOL
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NADH
            [0.0, 0.0, 0.0, 0.0, 0.0]    # LTO
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],   # PD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NAD
            [0, 0.0, 0.0, 0.0, 0.0],  # LTOL
            [0.0, 0, 0.0, 0.0, 0.0], # NADH
            [0.0, 0.0, 0.0, 0.0, 0.0]    # LTOL
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],   # PD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NAD
            [0.0, 0.0, 0.0, 0.0, 0],   # LTOL
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NADH
            [0.0, 0.0, 0.0, 0.0, 0.0]    # LTOL
        ]
    ]

    # Uncompetitive inhibition constants (not used in this test)
    model.root.input.model.unit_001.reaction_bulk.mm_ki_uc = [
            [
            [0.0, 0.0, 0.0, 0.0, 0.0],   # PD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NAD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # LTOL
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NADH
            [0.0, 0.0, 0.0, 0.0, 0.0]    # LTON
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],   # PD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NAD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # LTOL
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NADH
            [0.0, 0.0, 0.0, 0.0, 0.0]    # LTON
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],   # PD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NAD
            [0.0, 0.0, 0.0, 0.0, 0.0],   # LTOL
            [0.0, 0.0, 0.0, 0.0, 0.0],   # NADH
            [0.0, 0.0, 0.0, 0.0, 0.0]    # LTON
        ]
    ]

    # Vmax and stoichiometry
    model.root.input.model.unit_001.reaction_bulk.mm_vmax = [vmax1_per_mg, vmax2_per_mg, vmax3_per_mg]
    model.root.input.model.unit_001.reaction_bulk.mm_stoichiometry_bulk = [
            [-1, 1, 0],     # PD
            [-1, 1,-1],     # NAD
            [ 1,-1,-1],     # LTOL
            [ 1,-1, 1],     # NADH
            [ 0, 0, 1]      # LTON
    ]

def cadet_simulation_full_system(parameters, sim_time=300.0, cm_iteration = 0):
    
    install_path = r'C:\Users\berger\CADET-Core\out\install\DEBUG\bin\cadet-cli.exe'
    model_do = Cadet(install_path)

    nComp = 5 
    sim_time = 500.0
    # c_PD0, c_NAD0, c_Lactol0, c_NADH0, c_Lacton0
    init_c = [100, 8, 1e-6, 1e-6, 1e-6] 

    create_base_system(model=model_do, ncomp=nComp, init_c=init_c)
    setup_solver(model_do, sim_time=sim_time)
    setup_connections(model_do, nComp)
    add_reaction_system(model_do, parameters)

    model_do.filename = "do_enzyme_system.h5"
    model_do.save()
    data_mm = model_do.run()
    model_do.load()
    
    if (data_mm.return_code != 0):
        print(f" Simulation {cm_iteration} fehlgeschlagen mit Rückgabecode: {data_mm.return_code}")
        return data_mm.return_code
    
    # Speichere die Ergebnisse in einer Datei
    simulation_dir = "Results/Simulations"  # Konsistent mit Results-Struktur
    os.makedirs(simulation_dir, exist_ok=True)
    


    try:
        if os.path.exists(model_do.filename):
            os.remove(model_do.filename)
    except:
        pass  # Ignoriere Aufräum-Fehler

    
    return data_mm


if __name__ == "__main__":

    df_parameters = pd.read_csv("Results/results.csv")

    results = []
    for index, row in df_parameters.iterrows():
        parameters = {
            'KmPD': row['KmPD'],
            'KmNAD': row['KmNAD'],
            'KmLactol': row['KmLactol'],
            'KmNADH': row['KmNADH'],
            'KiPD': row['KiPD'],
            'Vmax1': row['Vmax1'],
            'Vmax2': row['Vmax2'],
            'Vmax3': row['Vmax3']
        }
        
        result = cadet_simulation_full_system(parameters, sim_time=500.0, cm_iteration=index)
        results.append(result)

    # Plot results for each component
    if isinstance(results[0], int):
        print("All simulations failed")
    else:
        # Create plots for each component
        component_names = ['PD', 'NAD', 'LTOL', 'NADH', 'LTON']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for comp_idx in range(5):  # 5 components
            ax = axes[comp_idx]
            
            for sim_idx, result in enumerate(results):
                if not isinstance(result, int):  # Skip failed simulations
                    time = result.solution.solution_times
                    concentration = result.solution.unit_001.solution_outlet[:, comp_idx]
                    ax.plot(time, concentration, alpha=0.7, label=f'Sim {sim_idx}')
            
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Concentration [mol/L]')
            ax.set_title(f'Component: {component_names[comp_idx]}')
            ax.grid(True)
            ax.legend()
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig('Results/Simulations/component_profiles.png', dpi=300, bbox_inches='tight')
        plt.show()
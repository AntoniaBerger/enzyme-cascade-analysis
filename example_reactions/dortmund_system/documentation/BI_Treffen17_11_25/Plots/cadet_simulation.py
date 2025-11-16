import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


MC_RESULTS_PATH = r"C:\Users\berger\Documents\Projekts\enzyme-cascade-analysis\example_reactions\dortmund_system\results"


# Reaction 1

df_reaction1_NAD_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction1_full_experiment_NAD.csv"))
df_reaction1_NAD_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction1_rate_noise_NAD.csv"))

vmax1_full_NAD = df_reaction1_NAD_full['Vmax'].mean()
vmax1_rate_NAD = df_reaction1_NAD_rate['Vmax'].mean()
km11_full_NAD = df_reaction1_NAD_full['Km1'].mean()
km11_rate_NAD = df_reaction1_NAD_rate['Km1'].mean()

df_reaction1_PD_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction1_full_experiment_PD.csv"))
df_reaction1_PD_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction1_rate_noise_PD.csv"))

vmax1_full_PD = df_reaction1_PD_full['Vmax'].mean()
vmax1_rate_PD = df_reaction1_PD_rate['Vmax'].mean()
km21_full_PD = df_reaction1_PD_full['Km2'].mean()
km21_rate_PD = df_reaction1_PD_rate['Km2'].mean()

# Reaction 2
df_reaction2_HP_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_full_experiment_HP.csv"))
df_reaction2_HP_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_rate_noise_HP.csv"))

vmax2_full_HP = df_reaction2_HP_full['Vmax'].mean()
vmax2_rate_HP = df_reaction2_HP_rate['Vmax'].mean()
km12_full_HP = df_reaction2_HP_full['Km1'].mean()
km12_rate_HP = df_reaction2_HP_rate['Km1'].mean()



df_reaction2_NADH_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_full_experiment_NADH.csv"))
df_reaction2_NADH_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_rate_noise_NADH.csv"))

vmax2_full_NADH = df_reaction2_NADH_full['Vmax'].mean()
vmax2_rate_NADH = df_reaction2_NADH_rate['Vmax'].mean()
km22_full_NADH = df_reaction2_NADH_full['Km2'].mean()
km22_rate_NADH = df_reaction2_NADH_rate['Km2'].mean()


df_reaction2_PD_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_full_experiment_PD.csv"))
df_reaction2_PD_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction2_rate_noise_PD.csv"))

vmax2_full_PD = df_reaction2_PD_full['Vmax'].mean()
vmax2_rate_PD = df_reaction2_PD_rate['Vmax'].mean()
km12_full_PD = df_reaction2_PD_full['Km1'].mean()
km12_rate_PD = df_reaction2_PD_rate['Km1'].mean()
ki_full_PD = df_reaction2_PD_full['Ki'].mean()
ki_rate_PD = df_reaction2_PD_rate['Ki'].mean()

# Reaction 3
df_reaction3_LACTOL_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction3_full_experiment_LACTOL.csv"))
df_reaction3_LACTOL_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction3_rate_noise_LACTOL.csv"))

vmax3_full_LACTOL = df_reaction3_LACTOL_full['Vmax'].mean()
vmax3_rate_LACTOL = df_reaction3_LACTOL_rate['Vmax'].mean()
km13_full_LACTOL = df_reaction3_LACTOL_full['Km1'].mean()
km13_rate_LACTOL = df_reaction3_LACTOL_rate['Km1'].mean()




df_reaction3_NAD_full = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction3_full_experiment_NAD.csv"))
df_reaction3_NAD_rate = pd.read_csv(os.path.join(MC_RESULTS_PATH, "MC_reaction3_rate_noise_NAD.csv"))

vmax3_full_NAD = df_reaction3_NAD_full['Vmax'].mean()
vmax3_rate_NAD = df_reaction3_NAD_rate['Vmax'].mean()
km23_full_NAD = df_reaction3_NAD_full['Km2'].mean()
km23_rate_NAD = df_reaction3_NAD_rate['Km2'].mean()



from cadet import Cadet


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
    
    install_path = r'C:\Users\berger\CADET-Corev5\out\install\aRELEASE\bin\cadet-cli.exe'
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
        print( data_mm.log )
        input("Press Enter to continue...")
    
    
    return model_do


parameters_full = {
    'Vmax1': vmax1_full_PD,
    'KmPD': km12_full_PD,
    'KmNAD': km11_full_NAD,
    'Vmax2': vmax2_full_NADH,
    'KmLactol': km13_full_LACTOL,
    'KmNADH': km22_full_NADH,
    'KiPD': ki_full_PD,
    'Vmax3': vmax3_full_NAD,
}


simulation_full_experiment = cadet_simulation_full_system(parameters_full, sim_time=1000.0)

zeit = simulation_full_experiment.root.output.solution.solution_times
konzentrationen_full = simulation_full_experiment.root.output.solution.unit_001.solution_outlet

parameters_rate = {
    'Vmax1': vmax1_rate_PD,
    'KmPD': km12_rate_PD,
    'KmNAD': km11_rate_NAD,
    'Vmax2': vmax2_rate_NADH,
    'KmLactol': km13_rate_LACTOL,
    'KmNADH': km22_rate_NADH,
    'KiPD': ki_rate_PD,
    'Vmax3': vmax3_rate_NAD,
}

simulation_rate_noise = cadet_simulation_full_system(parameters_rate, sim_time=1000.0)

zeit_rate = simulation_rate_noise.root.output.solution.solution_times
konzentrationen_rate = simulation_rate_noise.root.output.solution.unit_001.solution_outlet

# Plot for all substrates
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

substrate_names = ['PD', 'NAD', 'LTOL', 'NADH', 'LTON']
colors_full = ['blue', 'green', 'red', 'purple', 'orange']
colors_rate = ['lightblue', 'lightgreen', 'pink', 'plum', 'peachpuff']

for i, (name, color_full, color_rate) in enumerate(zip(substrate_names, colors_full, colors_rate)):
    if i < 5:  # We have 5 substrates
        axes[i].plot(zeit, konzentrationen_full[:,i], label=f'{name} Full Experiment', 
                    color=color_full, linewidth=2)
        axes[i].plot(zeit_rate, konzentrationen_rate[:,i], label=f'{name} Rate Noise', 
                    color=color_rate, linestyle='--', linewidth=2)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Concentration (mM)')
        axes[i].set_title(f'{name} Concentration: Full Experiment vs Rate Noise')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

# Remove the empty subplot
axes[5].remove()

plt.tight_layout()
plt.savefig("example_reactions/dortmund_system/documentation/BI_Treffen17_11_25/Plots/cadet_simulation_full_vs_rate_noise.png")
plt.show()

import time
from lsdo_geo.core.geometry.geometry_functions import import_geometry
import m3l
import numpy as np
import python_csdl_backend
import lsdo_geo as lg
from lsdo_rotor import BEMParameters, evaluate_multiple_BEM_models, BEM
from VAST import FluidProblem, VASTFluidSover, VASTNodalForces
import caddee.api as cd
from aframe import BeamMassModel, EBBeam, EBBeamForces
from caddee.utils.helper_functions.geometry_helpers import (make_rotor_mesh, make_vlm_camber_mesh, make_1d_box_beam_mesh, 
                                                            compute_component_surface_area, BladeParameters)
from darpa.payload import Payload
from darpa.battery import Battery
from darpa.motor import Motor
from darpa.fuselage import Fuselage
from darpa.heleeos import HELEEOS
from darpa.pod import Pod

import sys
sys.setrecursionlimit(100000000)

caddee = cd.CADDEE()
geometry = lg.import_geometry('darpa4.stp', parallelize=True)
geometry.refit(parallelize=True, fit_resolution=(50, 50), num_coefficients=(30, 30),)
# geometry.plot()
# m3l_model = m3l.Model()
system_model = m3l.Model()

wing = geometry.declare_component(component_name='wing', b_spline_search_names=['Wing'])
htail = geometry.declare_component(component_name='htail', b_spline_search_names=['OldTail'])
lfuse = geometry.declare_component(component_name='lfuse', b_spline_search_names=['Left_Fuselage'])
rfuse = geometry.declare_component(component_name='rfuse', b_spline_search_names=['Right_Fuselage'])
lprop = geometry.declare_component(component_name='lprop', b_spline_search_names=['Left_Prop'])
rprop = geometry.declare_component(component_name='rprop', b_spline_search_names=['Right_Prop'])
ldisk = geometry.declare_component(component_name='ldisk', b_spline_search_names=['Left_Disk'])
rdisk = geometry.declare_component(component_name='rdisk', b_spline_search_names=['Right_Disk'])
pod = geometry.declare_component(component_name='pod', b_spline_search_names=['Pod'])



# wing_te_right = wing.project(np.array([16.553, 29.986, 1.447]), plot=False)
# wing_te_left = wing.project(np.array([16.553, -29.986, 1.447]), plot=False)
# wing_te_center = wing.project(np.array([18.05, 0, 0.637]), plot=False)
# wing_le_left = wing.project(np.array([14.803, -29.986, 1.455]), plot=False)
# wing_le_right = wing.project(np.array([14.803, 29.986, 1.455]), plot=False)
# wing_le_center = wing.project(np.array([14, 0, 0.654]), plot=False)

cg = wing.project(np.array([14.867, 0, 0.777]), plot=False)

num_spanwise_vlm = 25
num_chordwise_vlm = 8

wing_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=wing,
    num_spanwise=num_spanwise_vlm,
    num_chordwise=num_chordwise_vlm,
    te_right=np.array([16.553, 29.986, 1.447]),
    te_left=np.array([16.553, -29.986, 1.447]),
    te_center=np.array([18.05, 0, 0.637]),
    le_left=np.array([14.803, -29.986, 1.455]),
    le_right=np.array([14.803, 29.986, 1.455]),
    le_center=np.array([14, 0, 0.654]),
    grid_search_density_parameter=100,
    off_set_x=0.2,
    bunching_cos=True,
    plot=False,
    mirror=True,
)

num_spanwise_vlm_htail = 8
num_chordwise_vlm_htail = 4

tail_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=htail, 
    num_spanwise=num_spanwise_vlm_htail,
    num_chordwise=num_chordwise_vlm_htail,
    te_right=np.array([27.5, 5.5, 4.539]),
    te_left=np.array([27.5, -5.5, 4.539]),
    le_right=np.array([24.5, 5.5, 4.487]),
    le_left=np.array([24.5, -5.5, 4.487]),
    plot=False,
    mirror=True,
)



# # wing beam mesh (old way)
# eps = 2
# num_wing_beam = 15
# leading_edge = wing.project(np.linspace(np.array([14.803-eps, -29.986, 1.455]), np.array([14.803-eps, 29.986, 1.455]), num_wing_beam), direction=np.array([0., 0., -1.]), plot=True)
# trailing_edge = wing.project(np.linspace(np.array([16.553+eps, -29.986, 1.447]), np.array([16.553+eps, 29.986, 1.447]), num_wing_beam), direction=np.array([0., 0., -1.]), plot=True)
# wing_beam = m3l.linear_combination(geometry.evaluate(leading_edge), geometry.evaluate(trailing_edge), 1, 
#                                    start_weights=np.ones((num_wing_beam,))*0.75, stop_weights=np.ones((num_wing_beam,))*0.25)
# width = m3l.norm((geometry.evaluate(leading_edge) - geometry.evaluate(trailing_edge))*0.5)
# # geometry.plot_meshes([wing_beam.reshape((-1,3))])

# wing beam mesh (helper)
num_wing_beam = 15
wing_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=wing,
    num_beam_nodes=num_wing_beam,
    te_right=np.array([16.553, 29.986, 1.447]),
    te_left=np.array([16.553, -29.986, 1.447]),
    te_center=np.array([18.05, 0, 0.637]),
    le_left=np.array([14.803, -29.986, 1.455]),
    le_right=np.array([14.803, 29.986, 1.455]),
    le_center=np.array([14, 0, 0.654]),
    grid_search_density_parameter=100,
    beam_width=0.5,
    node_center=0.5,
    le_interp='linear',
    te_interp='linear',
    plot=False,
    )

num_tail_beam = 9
tail_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=htail,
    num_beam_nodes=num_tail_beam,
    te_right=np.array([27.5, 5.5, 4.539]),
    te_left=np.array([27.5, -5.5, 4.539]),
    le_right=np.array([24.5, 5.5, 4.487]),
    le_left=np.array([24.5, -5.5, 4.487]),
    grid_search_density_parameter=100,
    beam_width=0.5,
    node_center=0.5,
    le_interp='linear',
    te_interp='linear',
    plot=False,
    )


num_fuse_beam = 9
lfuse_front = lfuse.project(np.array([0, -7, 0.5]), plot=False)
lfuse_back = lfuse.project(np.array([27, -7, 0.81]), plot=False)
lfuse_beam_mesh = m3l.linspace(geometry.evaluate(lfuse_front), geometry.evaluate(lfuse_back), num_fuse_beam)
# geometry.plot_meshes([lfuse_beam_mesh.reshape((-1,3))])

rfuse_front = rfuse.project(np.array([0, 7, 0.5]), plot=False)
rfuse_back = rfuse.project(np.array([27, 7, 0.81]), plot=False)
rfuse_beam_mesh = m3l.linspace(geometry.evaluate(rfuse_front), geometry.evaluate(rfuse_back), num_fuse_beam)
# geometry.plot_meshes([rfuse_beam_mesh.reshape((-1,3))])







# rotor parameters
num_radial = 20
# prop_radius = system_model.create_input(name='prop_radius', val=6.5, dv_flag=False, lower=5.5, upper=7.5, scaler=1e-1)

left_prop_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=ldisk,
    origin=np.array([28., -7., 0.75]),
    y1=np.array([28., -10.25, 0.75]),
    y2=np.array([28., -3.75, 0.75]),
    z1=np.array([28., -7., -2.5]),
    z2=np.array([28., -7., 4.]),
    create_disk_mesh=False,
    plot=False,
    # radius=prop_radius,
)

right_prop_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=rdisk,
    origin=np.array([28., 7., 0.75]),
    y1=np.array([28., 3.75, 0.75]),
    y2=np.array([28., 10.25, 0.75]),
    z1=np.array([28., 7., -2.5]),
    z2=np.array([28., 7., 4.]),
    create_disk_mesh=False,
    plot=False,
    # radius=prop_radius,
)


bem_left_rotor_parameters = BEMParameters(
    num_blades=4,
    num_radial=num_radial,
    num_tangential=1,
    airfoil='NACA_4412',
    use_custom_airfoil_ml=False,
    mesh_units='ft',
)

bem_right_rotor_parameters = BEMParameters(
    num_blades=4,
    num_radial=num_radial,
    num_tangential=1,
    airfoil='NACA_4412',
    use_custom_airfoil_ml=False,
    mesh_units='ft',
)







# Aframe dictionaries
joints, bounds, beams = {}, {}, {}
youngs_modulus = 69E9
poisson_ratio = 0.33
shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))
material_density = 2780 

beams['wing_beam'] = {'E': youngs_modulus, 'G': shear_modulus, 'rho': material_density, 'cs': 'box', 'nodes': list(range(num_wing_beam))}
bounds['left_wing_root'] = {'beam': 'wing_beam','node': 5,'fdim': [1,1,1,0,1,1]} # with 15 nodes physically accurate
bounds['right_wing_root'] = {'beam': 'wing_beam','node': 9,'fdim': [1,1,1,0,1,1]} # with 15 nodes physically accurate

wing_beam_t_top = system_model.create_input(name='wing_beam_ttop' ,val=0.005 * np.ones((num_wing_beam, )), dv_flag=False, lower=0.0008, upper=0.02, scaler=10)
wing_beam_t_bot = system_model.create_input(name='wing_beam_tbot' ,val=0.005 * np.ones((num_wing_beam, )), dv_flag=False, lower=0.0008, upper=0.02, scaler=10)
wing_beam_tweb = system_model.create_input(name='wing_beam_tweb' ,val=0.005 * np.ones((num_wing_beam, )), dv_flag=False, lower=0.000508, upper=0.02, scaler=10)

beam_mass_model = BeamMassModel(beams=beams, name='wing_beam_mass_model')
wing_beam_mass_props = beam_mass_model.evaluate(beam_nodes=wing_beam_mesh.beam_nodes,
                                        width=wing_beam_mesh.width, height=wing_beam_mesh.height, 
                                        t_top=wing_beam_t_top, t_bot=wing_beam_t_bot ,t_web=wing_beam_tweb)

system_model.register_output(wing_beam_mass_props)







total_mass_props_model = cd.TotalMassPropertiesM3L(name=f"total_mass_properties_model")
total_mass_props = total_mass_props_model.evaluate(component_mass_properties=[wing_beam_mass_props])
system_model.register_output(total_mass_props)
# system_model.add_objective(total_mass_props.mass, scaler=1e-3)









cruise = True
plus_3g = True

if cruise:
    cruise_condition = cd.CruiseCondition(
        name='steady_cruise',
        num_nodes=1,
        stability_flag=True,
    )

    cruise_M = system_model.create_input('cruise_mach', val=0.195)
    cruise_h = system_model.create_input('cruise_altitude', val=1000)
    cruise_range = system_model.create_input('cruise_range', val=60000)
    cruise_pitch = system_model.create_input('cruise_pitch', val=np.deg2rad(0), dv_flag=True, lower=np.deg2rad(-5), upper=np.deg2rad(5), scaler=10)
    cruise_ac_states, cruise_atmos = cruise_condition.evaluate(mach_number=cruise_M, pitch_angle=cruise_pitch, altitude=cruise_h, cruise_range=cruise_range)

    system_model.register_output(cruise_ac_states)
    system_model.register_output(cruise_atmos)





    left_cruise_bem = BEM(
        name='left_cruise_bem',
        num_nodes=1, 
        BEM_parameters=bem_left_rotor_parameters,
        rotation_direction='ignore',
    )
    left_cruise_rpm = system_model.create_input('left_cruise_rpm', val=2000, dv_flag=True, lower=600, upper=2500, scaler=1e-3)
    left_cruise_chord_cp = system_model.create_input('left_cruise_chord_cp', val=np.linspace(0.2, 0.1, 4))
    left_cruise_blade_twist_cp = system_model.create_input('left_cruise_blade_twist_cp', val=np.linspace(np.deg2rad(45), np.deg2rad(10), 4))
    left_cruise_bem_outputs = left_cruise_bem.evaluate(ac_states=cruise_ac_states, rpm=left_cruise_rpm, rotor_radius=left_prop_mesh.radius, thrust_vector=left_prop_mesh.thrust_vector,
                                                    thrust_origin=left_prop_mesh.thrust_origin, atmosphere=cruise_atmos, blade_chord_cp=left_cruise_chord_cp, blade_twist_cp=left_cruise_blade_twist_cp)
    system_model.register_output(left_cruise_bem_outputs)

    right_cruise_bem = BEM(
        name='right_cruise_bem',
        num_nodes=1, 
        BEM_parameters=bem_right_rotor_parameters,
        rotation_direction='ignore',
    )
    right_cruise_rpm = system_model.create_input('right_cruise_rpm', val=2000, dv_flag=True, lower=600, upper=2500, scaler=1e-3)
    right_cruise_chord_cp = system_model.create_input('right_cruise_chord_cp', val=np.linspace(0.2, 0.1, 4))
    right_cruise_blade_twist_cp = system_model.create_input('right_cruise_blade_twist_cp', val=np.linspace(np.deg2rad(45), np.deg2rad(10), 4))
    right_cruise_bem_outputs = right_cruise_bem.evaluate(ac_states=cruise_ac_states, rpm=right_cruise_rpm, rotor_radius=right_prop_mesh.radius, thrust_vector=right_prop_mesh.thrust_vector,
                                                    thrust_origin=right_prop_mesh.thrust_origin, atmosphere=cruise_atmos, blade_chord_cp=right_cruise_chord_cp, blade_twist_cp=right_cruise_blade_twist_cp)
    system_model.register_output(right_cruise_bem_outputs)




    # VAST VLM model
    vlm_model = VASTFluidSover(
        name='cruise_vlm_model',
        surface_names=[
            'cruise_wing_mesh',
            'cruise_tail_mesh',
        ],
        surface_shapes=[
            (1, ) + wing_meshes.vlm_mesh.shape[1:],
            (1, ) + tail_meshes.vlm_mesh.shape[1:],
        ],
        fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake', symmetry=True),
        mesh_unit='ft',
        cl0=[0., 0.,]
    )

    vlm_outputs = vlm_model.evaluate(
        atmosphere=cruise_atmos,
        ac_states=cruise_ac_states,
        meshes=[wing_meshes.vlm_mesh, tail_meshes.vlm_mesh],
        # deflections=[None, None],
        # wing_AR=wing_AR,
        eval_pt=geometry.evaluate(cg),
    )
    system_model.register_output(vlm_outputs)

    # # VAST nodal forces
    # vlm_force_mapping_model = VASTNodalForces(
    #     name='vast_cruise_nodal_forces',
    #     surface_names=[
    #         f'cruise_wing_mesh',
    #         f'cruise_tail_mesh',
    #     ],
    #     surface_shapes=[
    #         (1, ) + wing_meshes.vlm_mesh.shape[1:],
    #         (1, ) + tail_meshes.vlm_mesh.shape[1:],
    #     ],
    #     initial_meshes=[
    #         wing_meshes.vlm_mesh,
    #         tail_meshes.vlm_mesh
    #     ]
    # )


    # oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=vlm_outputs.panel_forces, nodal_force_meshes=[wing_meshes.oml_mesh, tail_meshes.oml_mesh])
    # wing_oml_forces = oml_forces[0]
    # tail_oml_forces = oml_forces[1]

    # system_model.register_output(wing_oml_forces)
    # system_model.register_output(tail_oml_forces)




    # # beam model
    # beam_force_map_model = EBBeamForces(name='eb_beam_force_map_cruise', beams=beams, exclude_middle=True)

    # structural_wing_mesh_forces_cruise = beam_force_map_model.evaluate(
    #     beam_mesh=wing_beam_mesh.beam_nodes,
    #     nodal_forces=wing_oml_forces,
    #     nodal_forces_mesh=wing_meshes.oml_mesh
    # )

    # beam_displacement_model = EBBeam(
    #     name='eb_beam_cruise',
    #     beams=beams,
    #     bounds=bounds,
    #     joints=joints,
    #     mesh_units='ft',
    # )

    # cruise_eb_beam_outputs = beam_displacement_model.evaluate(beam_mesh=wing_beam_mesh, 
    #                                                            t_top=wing_beam_t_top, 
    #                                                            t_bot=wing_beam_t_bot, 
    #                                                            t_web=wing_beam_tweb, 
    #                                                            forces=structural_wing_mesh_forces_cruise)


    # # trim computation
    # cruise_trim_variables = cruise_condition.assemble_trim_residual(
    #     mass_properties=[wing_beam_mass_props],
    #     aero_propulsive_outputs=[vlm_outputs, left_cruise_bem_outputs, right_cruise_bem_outputs],
    #     ac_states=cruise_ac_states,
    #     load_factor=1.,
    #     ref_pt=geometry.evaluate(cg),
    # )
    # system_model.register_output(cruise_trim_variables)
    # system_model.add_constraint(cruise_trim_variables.accelerations, equals=0, scaler=5.)






caddee_csdl_model = system_model.assemble_csdl()
sim = python_csdl_backend.Simulator(caddee_csdl_model, analytics=True)
sim.run()
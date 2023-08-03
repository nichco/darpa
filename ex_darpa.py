# imports
import caddee.api as cd
import m3l
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
import array_mapper as am
import lsdo_geo as lg
from lsdo_rotor.core.BEM_caddee.BEM_caddee import BEM, BEMMesh
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from caddee.utils.aircraft_models.pav.pav_weight import PavMassProperties
from aframe.core.mass import Mass, MassMesh
from VAST.core.generate_mappings_m3l import VASTNodalForces
from aframe.core.beam_module import EBBeam, LinearBeamMesh
import aframe.core.beam_module as ebbeam
from caddee import GEOMETRY_FILES_FOLDER
import numpy as np
import pandas as pd
from darpa.payload import Payload
from darpa.battery import Battery
from darpa.motor import Motor
from darpa.fuselage import Fuselage
from darpa.heleeos import HELEEOS
from darpa.pod import Pod



file_name = 'darpa/darpa2.stp'

caddee = cd.CADDEE()
caddee.system_model = system_model = cd.SystemModel()
caddee.system_representation = sys_rep = cd.SystemRepresentation()
caddee.system_parameterization = sys_param = cd.SystemParameterization(system_representation=sys_rep)

spatial_rep = sys_rep.spatial_representation
spatial_rep.import_file(file_name=file_name)
# spatial_rep.refit_geometry(file_name=GEOMETRY_FILES_FOLDER / file_name)
spatial_rep.refit_geometry(file_name=file_name)
# spatial_rep.plot(plot_types=['mesh'])


# wing
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['wing']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)
sys_rep.add_component(wing)
# wing.plot()

# tail
tail_primitive_names = list(spatial_rep.get_primitives(search_names=['tail']).keys())
tail = cd.LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)
sys_rep.add_component(tail)
# tail.plot()

# right prop
right_prop_primitive_names = list(spatial_rep.get_primitives(search_names=['rdisk']).keys())
right_prop = cd.Rotor(name='right_prop', spatial_representation=spatial_rep, primitive_names=right_prop_primitive_names)
sys_rep.add_component(right_prop)
# right_prop.plot()

# left prop
left_prop_primitive_names = list(spatial_rep.get_primitives(search_names=['ldisk']).keys())
left_prop = cd.Rotor(name='left_prop', spatial_representation=spatial_rep, primitive_names=left_prop_primitive_names)
sys_rep.add_component(left_prop)
# left_prop.plot()

# left fuselage
left_fuse_primitive_names = list(spatial_rep.get_primitives(search_names=['leftfuse']).keys())
left_fuse = Component(name='left_fuse', spatial_representation=spatial_rep, primitive_names=left_fuse_primitive_names)
sys_rep.add_component(left_fuse)
# left_fuse.plot()

# right fuselage
right_fuse_primitive_names = list(spatial_rep.get_primitives(search_names=['rightfuse']).keys())
right_fuse = Component(name='right_fuse', spatial_representation=spatial_rep, primitive_names=right_fuse_primitive_names)
sys_rep.add_component(right_fuse)
# right_fuse.plot()

# right prop blades (for visualization, BEM uses lprop and rprop)
right_rotor_primitive_names = list(spatial_rep.get_primitives(search_names=['rprop']).keys())
right_rotor = cd.Rotor(name='right_rotor', spatial_representation=spatial_rep, primitive_names=right_rotor_primitive_names)
sys_rep.add_component(right_rotor)
# right_rotor.plot()

# left prop blades (for visualization, BEM uses lprop and rprop)
left_rotor_primitive_names = list(spatial_rep.get_primitives(search_names=['lprop']).keys())
left_rotor = cd.Rotor(name='left_rotor', spatial_representation=spatial_rep, primitive_names=left_rotor_primitive_names)
sys_rep.add_component(left_rotor)
# left_rotor.plot()




# FFD
# wing FFD
wing_geometry_primitives = wing.get_geometry_primitives()
wing_ffd_bspline_volume = cd.create_cartesian_enclosure_volume(wing_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
wing_ffd_block = cd.SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_geometry_primitives)
wing_ffd_block.add_translation_u(name='wing_span_scaling', connection_name='wing_span_scaling', order=2,num_dof=2, value=np.array([0., 0.]))
wing_ffd_block.add_rotation_u(name='wing_incidence', connection_name='wing_incidence', order=1, num_dof=1, value=np.array([0.]))
#wing_ffd_block.plot_sections()
#wing_ffd_bspline_volume.plot()

# tail FFD
tail_geometry_primitives = tail.get_geometry_primitives()
tail_ffd_bspline_volume = cd.create_cartesian_enclosure_volume(tail_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
tail_ffd_block = cd.SRBGFFDBlock(name='tail_ffd_block', primitive=tail_ffd_bspline_volume, embedded_entities=tail_geometry_primitives)
tail_ffd_block.add_translation_u(name='tail_span_scaling', connection_name='tail_span_scaling', order=2,num_dof=2)
tail_ffd_block.add_rotation_u(name='tail_twist_distribution', connection_name='h_tail_act', order=1, num_dof=1, value=np.array([np.deg2rad(0)]))
#tail_ffd_block.plot_sections()
#tail_ffd_bspline_volume.plot()

# left fuse FFD
left_fuse_geometry_primitives = left_fuse.get_geometry_primitives()
left_fuse_ffd_bspline_volume = cd.create_cartesian_enclosure_volume(left_fuse_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(1,0,2))
left_fuse_ffd_block = cd.SRBGFFDBlock(name='left_fuse_ffd_block', primitive=left_fuse_ffd_bspline_volume, embedded_entities=left_fuse_geometry_primitives)
left_fuse_ffd_block.add_translation_u(name='left_fuse_scaling', connection_name='left_fuse_scaling', order=1,num_dof=1)
left_fuse_ffd_block.add_rotation_u(name='left_fuse_rotation', connection_name='left_fuse_rotation', order=1, num_dof=1, value=np.array([0.]))


# right fuse FFD
right_fuse_geometry_primitives = right_fuse.get_geometry_primitives()
right_fuse_ffd_bspline_volume = cd.create_cartesian_enclosure_volume(right_fuse_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(1,0,2))
right_fuse_ffd_block = cd.SRBGFFDBlock(name='right_fuse_ffd_block', primitive=right_fuse_ffd_bspline_volume, embedded_entities=right_fuse_geometry_primitives)
right_fuse_ffd_block.add_translation_u(name='right_fuse_scaling', connection_name='right_fuse_scaling', order=1,num_dof=1)
right_fuse_ffd_block.add_rotation_u(name='right_fuse_rotation', connection_name='right_fuse_rotation', order=1, num_dof=1, value=np.array([0.]))



# left prop FFD
left_prop_geometry_primitives = left_prop.get_geometry_primitives()
left_prop_ffd_bspline_volume = cd.create_cartesian_enclosure_volume(left_prop_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(1,0,2))
left_prop_ffd_block = cd.SRBGFFDBlock(name='left_prop_ffd_block', primitive=left_prop_ffd_bspline_volume, embedded_entities=left_prop_geometry_primitives)
left_prop_ffd_block.add_translation_u(name='left_prop_pos', connection_name='left_prop_pos', order=1,num_dof=1)

# right prop FFD
right_prop_geometry_primitives = right_prop.get_geometry_primitives()
right_prop_ffd_bspline_volume = cd.create_cartesian_enclosure_volume(right_prop_geometry_primitives, num_control_points=(2, 2, 2), order=(2,2,2), xyz_to_uvw_indices=(1,0,2))
right_prop_ffd_block = cd.SRBGFFDBlock(name='right_prop_ffd_block', primitive=right_prop_ffd_bspline_volume, embedded_entities=right_prop_geometry_primitives)
right_prop_ffd_block.add_translation_u(name='right_prop_pos', connection_name='right_prop_pos', order=1,num_dof=1)


ffd_set = cd.SRBGFFDSet(name='ffd_set',ffd_blocks={wing_ffd_block.name : wing_ffd_block,
                                                   tail_ffd_block.name : tail_ffd_block,
                                                   left_fuse_ffd_block.name : left_fuse_ffd_block,
                                                   right_fuse_ffd_block.name : right_fuse_ffd_block,
                                                   left_prop_ffd_block.name : left_prop_ffd_block,
                                                   right_prop_ffd_block.name : right_prop_ffd_block})
sys_param.add_geometry_parameterization(ffd_set)
sys_param.setup()




# wing mesh
num_spanwise_vlm = 18
num_chordwise_vlm = 5
offset = 2
leading_edge = wing.project(np.linspace(np.array([14.873 - offset, -31.966, 1.874]), np.array([14.873 - offset, 31.966, 1.874]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
trailing_edge = wing.project(np.linspace(np.array([16.622 + offset, -31.966, 1.813]), np.array([16.622 + offset, 31.966, 1.813]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=False)
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
# spatial_rep.plot_meshes([chord_surface])

wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=30, plot=False)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=30, plot=False)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1
# spatial_rep.plot_meshes([wing_camber_surface])

wing_vlm_mesh_name = 'wing_vlm_mesh'
sys_rep.add_output(wing_vlm_mesh_name, wing_camber_surface)
wing_oml_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
wing_oml_mesh_name = 'wing_oml_mesh'
sys_rep.add_output(wing_oml_mesh_name, wing_oml_mesh)


# tail mesh
tail_leading_edge = tail.project(np.linspace(np.array([27.501, -5.5, 4.7]), np.array([27.501, 5.5, 4.7]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_n=15, plot=False)
tail_trailing_edge = tail.project(np.linspace(np.array([30.497, -5.5, 4.857]), np.array([30.497, 5.5, 4.857]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_n=15, plot=False)
tail_chord_surface = am.linspace(tail_leading_edge, tail_trailing_edge, num_chordwise_vlm)
# spatial_rep.plot_meshes([tail_chord_surface])

tail_upper_surface_wireframe = tail.project(tail_chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=30, plot=False)
tail_lower_surface_wireframe = tail.project(tail_chord_surface.value - np.array([0., 0., 0.5]), direction=np.array([0., 0., 1.]), grid_search_n=50, plot=False)
tail_camber_surface = am.linspace(tail_upper_surface_wireframe, tail_lower_surface_wireframe, 1) # this linspace will return average when n=1
# spatial_rep.plot_meshes([tail_camber_surface])

tail_vlm_mesh_name = 'tail_vlm_mesh'
sys_rep.add_output(tail_vlm_mesh_name, tail_camber_surface)






# left prop mesh
y11 = left_prop.project(np.array([30.5, -7.0, -3.0]), direction=np.array([-1., 0., 0.]), plot=False) # bottom
y12 = left_prop.project(np.array([30.5, -7.0, 4.5]), direction=np.array([-1., 0., 0.]), plot=False) # top
y21 = left_prop.project(np.array([30.5, -10.75, 0.75]), direction=np.array([-1., 0., 0.]), plot=False) # left
y22 = left_prop.project(np.array([30.5, -3.25, 0.75]), direction=np.array([-1., 0., 0.]), plot=False) # right
left_prop_in_plane_y = am.subtract(y11, y12)
left_prop_in_plane_x = am.subtract(-1*y21, -1*y22)
left_prop_origin = left_prop.project(np.array([30.5, -7, 0.75]), plot=False) # center
sys_rep.add_output(f"{left_prop.parameters['name']}_in_plane_1", left_prop_in_plane_y)
sys_rep.add_output(f"{left_prop.parameters['name']}_in_plane_2", left_prop_in_plane_x)
sys_rep.add_output(f"{left_prop.parameters['name']}_origin", left_prop_origin)

# right prop mesh
y11 = right_prop.project(np.array([30.5, 7.0, -3.0]), direction=np.array([-1., 0., 0.]), plot=False) # bottom
y12 = right_prop.project(np.array([30.5, 7.0, 4.5]), direction=np.array([-1., 0., 0.]), plot=False) # top
y21 = right_prop.project(np.array([30.5, 10.75, 0.75]), direction=np.array([-1., 0., 0.]), plot=False) # left
y22 = right_prop.project(np.array([30.5, 3.25, 0.75]), direction=np.array([-1., 0., 0.]), plot=False) # right
right_prop_in_plane_y = am.subtract(y11, y12)
right_prop_in_plane_x = am.subtract(1*y21, 1*y22)
right_prop_origin = right_prop.project(np.array([30.5, 7, 0.75]), plot=False) # center
sys_rep.add_output(f"{right_prop.parameters['name']}_in_plane_1", right_prop_in_plane_y)
sys_rep.add_output(f"{right_prop.parameters['name']}_in_plane_2", right_prop_in_plane_x)
sys_rep.add_output(f"{right_prop.parameters['name']}_origin", right_prop_origin)

"""
print(left_prop_in_plane_y)
print(right_prop_in_plane_y)
print(left_prop_in_plane_x)
print(right_prop_in_plane_x)
exit()
"""

# wing beam mesh
num_wing_beam = 15
leading_edge = wing.project(np.linspace(np.array([14.873 - offset, -31.966, 1.874]), np.array([14.873 - offset, 31.966, 1.874]), num_wing_beam), direction=np.array([0., 0., -1.]), plot=False)
trailing_edge = wing.project(np.linspace(np.array([16.622 + offset, -31.966, 1.813]), np.array([16.622 + offset, 31.966, 1.813]), num_wing_beam), direction=np.array([0., 0., -1.]), plot=False)
wing_beam = am.linear_combination(leading_edge,trailing_edge,1,start_weights=np.ones((num_wing_beam,))*0.75,stop_weights=np.ones((num_wing_beam,))*0.25)
width = am.norm((leading_edge - trailing_edge)*0.5)
# spatial_rep.plot_meshes([wing_beam])
print(leading_edge - trailing_edge)

beam_offset = np.array([0,0,0.5])
top = wing.project(wing_beam.value + beam_offset, direction=np.array([0., 0., -1.]), plot=False)
bot = wing.project(wing_beam.value - beam_offset, direction=np.array([0., 0., 1.]), plot=False)

top = top.reshape((num_wing_beam,3))
bot = bot.reshape((num_wing_beam,3))
height = am.norm((top - bot)*1.0)

sys_rep.add_output(name='wing_beam_mesh', quantity=wing_beam)
sys_rep.add_output(name='wing_beam_width', quantity=width)
sys_rep.add_output(name='wing_beam_height', quantity=height)


# aframe meshes
beam_mesh = LinearBeamMesh(
meshes = dict(
wing_beam = wing_beam,
wing_beam_width = width,
wing_beam_height = height,)
)

# pass the beam meshes to the aframe mass model:
beam_mass_mesh = MassMesh(
meshes = dict(
wing_beam = wing_beam,
wing_beam_width = width,
wing_beam_height = height,)
)










# design scenario
design_scenario = cd.DesignScenario(name='cruise')

# cruise
cruise_model = m3l.Model()
cruise_condition = cd.CruiseCondition(name='cruise')
cruise_condition.atmosphere_model = cd.SimpleAtmosphereModel()
cruise_condition.set_module_input(name='altitude', val=5000) # 18288
cruise_condition.set_module_input(name='mach_number', val=0.15, dv_flag=True, lower=0.1, upper=0.3)
cruise_condition.set_module_input(name='range', val=10000)
cruise_condition.set_module_input(name='pitch_angle', val=np.deg2rad(0), dv_flag=True, lower=np.deg2rad(-10), upper=np.deg2rad(10))
cruise_condition.set_module_input(name='flight_path_angle', val=0)
cruise_condition.set_module_input(name='roll_angle', val=0)
cruise_condition.set_module_input(name='yaw_angle', val=0)
cruise_condition.set_module_input(name='wind_angle', val=0)
cruise_condition.set_module_input(name='observer_location', val=np.array([0, 0, 1000]))

ac_states = cruise_condition.evaluate_ac_states()
cruise_model.register_output(ac_states)



# BEM meshes
left_prop_mesh = BEMMesh(
    meshes=dict(
        left_prop_in_plane_y=left_prop_in_plane_y,
        left_prop_in_plane_x=left_prop_in_plane_x,
        left_prop_origin=left_prop_origin,),
    num_blades=3,
    num_radial=30,
    airfoil='NACA_4412',
    mesh_units='ft',
    use_rotor_geometry=False,
    use_airfoil_ml=False,
    airfoil_polar=None,
    chord_b_spline_rep=True,
    twist_b_spline_rep=True,)

right_prop_mesh = BEMMesh(
    meshes=dict(
        right_prop_in_plane_y=right_prop_in_plane_y,
        right_prop_in_plane_x=right_prop_in_plane_x,
        right_prop_origin=right_prop_origin,),
    num_blades=3,
    num_radial=30,
    airfoil='NACA_4412',
    mesh_units='ft',
    use_rotor_geometry=False,
    use_airfoil_ml=False,
    airfoil_polar=None,
    chord_b_spline_rep=True,
    twist_b_spline_rep=True,)


left_bem_model = BEM(disk_prefix='left_prop', blade_prefix='left_prop', component=left_prop, mesh=left_prop_mesh)
left_bem_model.set_module_input('rpm', val=1500, dv_flag=True, lower=500, upper=3000, scaler=1e-3)
left_bem_model.set_module_input('propeller_radius', val=1.0, dv_flag=False)
left_bem_model.set_module_input('chord_cp', val=np.linspace(0.15, 0.05, 4), dv_flag=False)
left_bem_model.set_module_input('twist_cp', val=np.deg2rad(np.linspace(50, 15, 4)), dv_flag=False)
left_bem_model.set_module_input('thrust_vector', val=np.array([1., 0., 0.]))
left_bem_model.set_module_input('thrust_origin', val=np.array([30.5, -7, 0.75]))
left_bem_forces, left_bem_moments, _, _, _, _, _, _ = left_bem_model.evaluate(ac_states=ac_states)
cruise_model.register_output(left_bem_forces)
cruise_model.register_output(left_bem_moments)

right_bem_model = BEM(disk_prefix='right_prop', blade_prefix='right_prop', component=right_prop, mesh=right_prop_mesh)
right_bem_model.set_module_input('rpm', val=1500, dv_flag=True, lower=500, upper=3000, scaler=1e-3)
right_bem_model.set_module_input('propeller_radius', val=1.0, dv_flag=False)
right_bem_model.set_module_input('chord_cp', val=np.linspace(0.15, 0.05, 4), dv_flag=False)
right_bem_model.set_module_input('twist_cp', val=np.deg2rad(np.linspace(50, 15, 4)), dv_flag=False)
right_bem_model.set_module_input('thrust_vector', val=np.array([1., 0., 0.]))
right_bem_model.set_module_input('thrust_origin', val=np.array([30.5, 7, 0.75]))
right_bem_forces, right_bem_moments, _, _, _, _, _, _ = right_bem_model.evaluate(ac_states=ac_states)
cruise_model.register_output(right_bem_forces)
cruise_model.register_output(right_bem_moments)




# VLM solver
vlm_model = VASTFluidSover(
    surface_names=[
        wing_vlm_mesh_name,
        tail_vlm_mesh_name,
    ],
    surface_shapes=[
        (1, ) + wing_camber_surface.evaluate().shape[1:],
        (1, ) + tail_camber_surface.evaluate().shape[1:],
        ],
    fluid_problem=FluidProblem(solver_option='VLM', problem_type='fixed_wake'),
    mesh_unit='ft',
    cl0=[0.25, 0],

)
vlm_panel_forces, vlm_forces, vlm_moments = vlm_model.evaluate(ac_states=ac_states)
cruise_model.register_output(vlm_forces)
cruise_model.register_output(vlm_moments)

# VLM force mapping model
vlm_force_mapping_model = VASTNodalForces(
    surface_names=[
        wing_vlm_mesh_name,
        tail_vlm_mesh_name,
    ],
    surface_shapes=[
        (1, ) + wing_camber_surface.evaluate().shape[1:],
        (1, ) + tail_camber_surface.evaluate().shape[1:],
        ],
    initial_meshes=[
        wing_camber_surface,
        tail_camber_surface]
)

oml_forces = vlm_force_mapping_model.evaluate(vlm_forces=vlm_panel_forces, nodal_force_meshes=[wing_oml_mesh, wing_oml_mesh])
wing_forces = oml_forces[0]
htail_forces = oml_forces[1]




# create the aframe dictionaries
joints, bounds, beams = {}, {}, {}
beams['wing_beam'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'box','nodes': list(range(num_wing_beam))}
# bounds['wing_root'] = {'beam': 'wing_beam','node': 5,'fdim': [1,1,1,1,1,1]} # with 11 nodes
# bounds['wing_root'] = {'beam': 'wing_beam','node': 7,'fdim': [1,1,1,1,1,1]} # with 15 nodes
bounds['left_wing_root'] = {'beam': 'wing_beam','node': 5,'fdim': [1,1,1,0,1,1]} # with 15 nodes physically accurate
bounds['right_wing_root'] = {'beam': 'wing_beam','node': 9,'fdim': [1,1,1,0,1,1]} # with 15 nodes physically accurate

# create the beam model
beam = EBBeam(component=wing, mesh=beam_mesh, beams=beams, bounds=bounds, joints=joints, mesh_units='ft')
beam_mass = Mass(component=wing, mesh=beam_mass_mesh, beams=beams, mesh_units='ft') # the separate mass model thingy
beam_mass.set_module_input('wing_beam_tcap', val=0.004, dv_flag=True, lower=0.001, upper=0.04, scaler=1E3)
beam_mass.set_module_input('wing_beam_tweb', val=0.004, dv_flag=True, lower=0.001, upper=0.04, scaler=1E3)


cruise_wing_structural_nodal_displacements_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
cruise_wing_aero_nodal_displacements_mesh = cruise_wing_structural_nodal_displacements_mesh
cruise_wing_structural_nodal_force_mesh = cruise_wing_structural_nodal_displacements_mesh
cruise_wing_aero_nodal_force_mesh = cruise_wing_structural_nodal_displacements_mesh

dummy_b_spline_space = lg.BSplineSpace(name='dummy_b_spline_space', order=(3, 1), control_points_shape=((35, 1)))
dummy_function_space = lg.BSplineSetSpace(name='dummy_space', spaces={'dummy_b_spline_space': dummy_b_spline_space})

cruise_wing_displacement_coefficients = m3l.Variable(name='cruise_wing_displacement_coefficients', shape=(35, 3))
cruise_wing_displacement = m3l.Function(name='cruise_wing_displacement', space=dummy_function_space, coefficients=cruise_wing_displacement_coefficients)

beam_force_map_model = ebbeam.EBBeamForces(component=wing,beam_mesh=beam_mesh,beams=beams,exclude_middle=False)
cruise_structural_wing_mesh_forces = beam_force_map_model.evaluate(nodal_forces=wing_forces,nodal_forces_mesh=wing_oml_mesh)

beam_displacements_model = ebbeam.EBBeam(component=wing, mesh=beam_mesh, beams=beams, bounds=bounds, joints=joints)
# beam_displacements_model.set_module_input('wing_beamt_cap_in', val=0.01, dv_flag=True, lower=0.001, upper=0.04, scaler=1E3)
# beam_displacements_model.set_module_input('wing_beamt_web_in', val=0.01, dv_flag=True, lower=0.001, upper=0.04, scaler=1E3)

cruise_structural_wing_mesh_displacements, cruise_structural_wing_mesh_rotations, wing_mass, wing_cg, wing_inertia_tensor = beam_displacements_model.evaluate(forces=cruise_structural_wing_mesh_forces)
cruise_model.register_output(cruise_structural_wing_mesh_displacements)

print(cruise_structural_wing_mesh_displacements)







# index functions
from m3l.utils.utils import index_functions
nodal_displacement = ebbeam.EBBeamNodalDisplacements(component=wing, beam_mesh=beam_mesh, beams=beams)
surface_names = list(wing.get_primitives().keys())
grid_num = 10
para_grid = []
for name in surface_names:
    for u in np.linspace(0,1,grid_num):
        for v in np.linspace(0,1,grid_num):
            para_grid.append((name, np.array([u,v]).reshape((1,2))))


evaluated_parametric_map = sys_rep.spatial_representation.evaluate_parametric(para_grid)
#nodal_displacements = nodal_displacement.evaluate(cruise_structural_wing_mesh_displacements.reshape((1,15,3)), evaluated_parametric_map)
#cruise_model.register_output(nodal_displacements)
















# separate wing mass model
mass_model_wing_mass = beam_mass.evaluate()
cruise_model.register_output(mass_model_wing_mass)

# the payload mass model
payload = Payload(component=wing, payload=50, payload_cg=np.array([4.6,0,0.2]))
payload_mass, payload_cg, payload_inertia = payload.evaluate()
cruise_model.register_output(payload_mass)

# the battery sizing model
battery = Battery(component=wing, esb=200, eta=0.75, endurance=0.5, battery_cg=np.array([4.6,0,0.2]))
battery_mass, battery_cg, battery_inertia = battery.evaluate()
cruise_model.register_output(battery_mass)

# the motor model
motor = Motor(component=wing, motor_power_density=3.1, eta=0.75, ratio=2.5, motor_cg=np.array([4.6,0,0.2])) # 3.1
motor_mass, motor_cg, motor_inertia = motor.evaluate()
cruise_model.register_output(motor_mass)

# the fuselage mass model
fuselage = Fuselage(component=wing, fuse_mass=150, fuse_cg=np.array([4.5,0,0.2]), fuse_cd=0.4, area=0.25)
fuse_mass, fuse_cg, fuse_inertia, fuse_forces, fuse_moments = fuselage.evaluate()
cruise_model.register_output(fuse_mass)

# HELEEOS
heleeos = HELEEOS(component=wing,)
heleeos.set_module_input('horizontal_distance', val=250000, dv_flag=False)
pid = heleeos.evaluate()
cruise_model.register_output(pid)

# pod model
pod = Pod(component=wing, cd=0.4, length=2.0, density=100.0, pod_cg=np.array([4.5,0,0.2]))
pod.set_module_input('aperture_diameter', val=0.75, dv_flag=True, lower=0.25, upper=1.0, scaler=1)
pod_mass, pod_cg, pod_inertia, pod_forces, pod_moments = pod.evaluate()
cruise_model.register_output(pod_mass)



# m3l mass properties
total_mass_properties = cd.TotalMassPropertiesM3L()
total_mass, total_cg, total_inertia = total_mass_properties.evaluate(mass_model_wing_mass, wing_cg, wing_inertia_tensor, 
                                                                     payload_mass, payload_cg, payload_inertia,
                                                                     battery_mass, battery_cg, battery_inertia,
                                                                     motor_mass, motor_cg, motor_inertia,
                                                                     fuse_mass, fuse_cg, fuse_inertia,
                                                                     pod_mass, pod_cg, pod_inertia)

cruise_model.register_output(total_mass)
cruise_model.register_output(total_cg)
cruise_model.register_output(total_inertia)

# inertial forces and moments
inertial_loads_model = cd.InertialLoadsM3L(load_factor=1.)
inertial_forces, inertial_moments = inertial_loads_model.evaluate(total_cg_vector=total_cg, totoal_mass=total_mass, ac_states=ac_states)
cruise_model.register_output(inertial_forces)
cruise_model.register_output(inertial_moments)

# total forces and moments
total_forces_moments_model = cd.TotalForcesMomentsM3L()
total_forces, total_moments = total_forces_moments_model.evaluate(vlm_forces, vlm_moments, 
                                                                  left_bem_forces, left_bem_moments, 
                                                                  right_bem_forces, right_bem_moments, 
                                                                  inertial_forces, inertial_moments,
                                                                  pod_forces, pod_moments,
                                                                  fuse_forces, fuse_moments)
cruise_model.register_output(total_forces)
cruise_model.register_output(total_moments)

# pass total forces/moments + mass properties into EoM model
eom_m3l_model = cd.EoMM3LEuler6DOF()
trim_residual = eom_m3l_model.evaluate(
    total_mass=total_mass,
    total_cg_vector=total_cg,
    total_inertia_tensor=total_inertia,
    total_forces=total_forces,
    total_moments=total_moments,
    ac_states=ac_states)

cruise_model.register_output(trim_residual)

# add the cruise m3l model to the cruise condition
cruise_condition.add_m3l_model('cruise_model', cruise_model)

# add the design condition to the design scenario
design_scenario.add_design_condition(cruise_condition)

system_model.add_design_scenario(design_scenario=design_scenario)

caddee_csdl_model = caddee.assemble_csdl()



# beam model connections
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.mass_model.wing_beam_tweb', 'system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.wing_beam_tweb')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.mass_model.wing_beam_tcap', 'system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.wing_beam_tcap')

# battery model connections
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.left_prop_bem_model.rpm', 'system_model.cruise.cruise.cruise.battery_mass_model.left_prop_rpm')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.right_prop_bem_model.rpm', 'system_model.cruise.cruise.cruise.battery_mass_model.right_prop_rpm')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.left_prop_bem_model.induced_velocity_model.C_Q', 'system_model.cruise.cruise.cruise.battery_mass_model.left_prop_C_Q')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.right_prop_bem_model.induced_velocity_model.C_Q', 'system_model.cruise.cruise.cruise.battery_mass_model.right_prop_C_Q')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.left_prop_bem_model.propeller_radius', 'system_model.cruise.cruise.cruise.battery_mass_model.left_prop_radius')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.right_prop_bem_model.propeller_radius', 'system_model.cruise.cruise.cruise.battery_mass_model.right_prop_radius')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.atmosphere_model.cruise_density', 'system_model.cruise.cruise.cruise.battery_mass_model.density')

# motor model connections
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.battery_mass_model.required_power', 'system_model.cruise.cruise.cruise.motor_mass_model.required_power')

# HELEEOS connections
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.battery_mass_model.required_power', 'system_model.cruise.cruise.cruise.heleeos.required_power')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.cruise_altitude', 'system_model.cruise.cruise.cruise.heleeos.altitude')

# pod connections
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.u', 'system_model.cruise.cruise.cruise.pod_model.u')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.v', 'system_model.cruise.cruise.cruise.pod_model.v')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.w', 'system_model.cruise.cruise.cruise.pod_model.w')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.pod_model.aperture_diameter', 'system_model.cruise.cruise.cruise.heleeos.aperture_size')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.atmosphere_model.cruise_density', 'system_model.cruise.cruise.cruise.pod_model.density')

# fuselage model connections for drag
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.u', 'system_model.cruise.cruise.cruise.fuselage_mass_model.u')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.v', 'system_model.cruise.cruise.cruise.fuselage_mass_model.v')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.w', 'system_model.cruise.cruise.cruise.fuselage_mass_model.w')
caddee_csdl_model.connect('system_model.cruise.cruise.cruise.cruise_ac_states_operation.atmosphere_model.cruise_density', 'system_model.cruise.cruise.cruise.fuselage_mass_model.density')



# tail rotation design variable
h_tail_act = caddee_csdl_model.create_input('h_tail_act', val=np.deg2rad(0))
caddee_csdl_model.add_design_variable('h_tail_act', lower=np.deg2rad(-10), upper=np.deg2rad(10), scaler=1,)

# wing span design variable
wing_span_scaling = caddee_csdl_model.create_input('wing_span_scaling', val=np.array([0., 0.]))
caddee_csdl_model.add_design_variable('wing_span_scaling', lower=-10, upper=10, scaler=1,) # original span is 64 ft
# caddee_csdl_model.add_design_variable('wing_span_scaling', scaler=1,)
# [-2, 2] means the wing is getting smaller
caddee_csdl_model.register_output('wing_span_residual', wing_span_scaling[1] + wing_span_scaling[0])
caddee_csdl_model.add_constraint('wing_span_residual', equals=0)

caddee_csdl_model.print_var(wing_span_scaling)

# tail span scaling
tail_span_scaling = caddee_csdl_model.create_input('tail_span_scaling', val=np.array([0., 0.]))
caddee_csdl_model.add_design_variable('tail_span_scaling', scaler=1, lower=-4, upper=4)
caddee_csdl_model.register_output('tail_span_residual', tail_span_scaling[1] + tail_span_scaling[0])
caddee_csdl_model.add_constraint('tail_span_residual', equals=0)

# connect the tail scaling variable to the left/right fuselages and the rotors (for visualization)
caddee_csdl_model.register_output('left_fuse_scaling', 1*tail_span_scaling[0])
caddee_csdl_model.register_output('right_fuse_scaling', 1*tail_span_scaling[1])
caddee_csdl_model.register_output('left_prop_pos', 1*tail_span_scaling[0])
caddee_csdl_model.register_output('right_prop_pos', 1*tail_span_scaling[1])


# beam symmetry constraint
# system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.wing_beam_tcap
tcap = caddee_csdl_model.declare_variable('system_model.cruise.cruise.cruise.mass_model.wing_beam_tcap', shape=(num_wing_beam - 1))
tcap_a = tcap[0:6]
tcap_b = tcap[8:14]
caddee_csdl_model.register_output('tcap_residual', tcap_b - tcap_a)
# caddee_csdl_model.add_constraint('tcap_residual', equals=0)

tweb = caddee_csdl_model.declare_variable('system_model.cruise.cruise.cruise.mass_model.wing_beam_tweb', shape=(num_wing_beam - 1))
tweb_a = tweb[0:6]
tweb_b = tweb[8:14]
caddee_csdl_model.register_output('tweb_residual', tweb_b - tweb_a)
# caddee_csdl_model.add_constraint('tweb_residual', equals=0)


# wing stress constraint
caddee_csdl_model.add_constraint('system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.new_stress', upper=500E6/3, scaler=1E-8)

# HELEEOS power residual constraint
caddee_csdl_model.add_constraint('system_model.cruise.cruise.cruise.heleeos.power_residual', lower=0, scaler=1E-3)

# trim constraint
caddee_csdl_model.add_constraint('system_model.cruise.cruise.cruise.euler_eom_gen_ref_pt.trim_residual', equals=0, scaler=1E-1)

# the objective:
# caddee_csdl_model.add_objective('system_model.cruise.cruise.cruise.euler_eom_gen_ref_pt.trim_residual', scaler=1E-1)
caddee_csdl_model.add_objective('system_model.cruise.cruise.cruise.total_constant_mass_properties.total_mass', scaler=1E-2)







"""
# simulator
sim = Simulator(caddee_csdl_model, analytics=True)
#sim.run()


prob = CSDLProblem(problem_name='lpc', simulator=sim)
optimizer = SLSQP(prob, maxiter=500, ftol=1E-8)
optimizer.solve()
optimizer.print_results()





print('Mach: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.cruise_mach_number'])
print('U: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.u'])
print('V: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.v'])
print('W: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.w'])
print('Pitch Angle: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.cruise_pitch_angle'])
print('Left Prop RPM: ', sim['system_model.cruise.cruise.cruise.left_prop_bem_model.rpm'])
print('Right Prop RPM: ', sim['system_model.cruise.cruise.cruise.right_prop_bem_model.rpm'])
print('H-Tail Act: ', sim['system_parameterization.ffd_set.rotational_section_properties_model.h_tail_act'])
print('Total Mass: ', sim['system_model.cruise.cruise.cruise.total_constant_mass_properties.total_mass'])
print('CG: ', sim['system_model.cruise.cruise.cruise.total_constant_mass_properties.total_cg_vector'])
print('Total Forces: ', sim['system_model.cruise.cruise.cruise.total_forces_moments_model.total_forces'])
print('Total Moments: ', sim['system_model.cruise.cruise.cruise.total_forces_moments_model.total_moments'])
print('Left Prop Power: ', sim['system_model.cruise.cruise.cruise.battery_mass_model.left_prop_power'])
print('Right Prop Power: ', sim['system_model.cruise.cruise.cruise.battery_mass_model.right_prop_power'])
print('Battery Mass: ', sim['system_model.cruise.cruise.cruise.battery_mass_model.mass'])
print('Wing Mass: ', sim['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.MassProp.struct_mass'])
print('Motor Mass: ', sim['system_model.cruise.cruise.cruise.motor_mass_model.mass'])
print('Pod Mass: ', sim['system_model.cruise.cruise.cruise.pod_model.mass'])
print('Fuselage Mass: ', sim['system_model.cruise.cruise.cruise.fuselage_mass_model.mass'])
print('Payload Mass: ', sim['system_model.cruise.cruise.cruise.payload_mass_model.mass'])
print('Aperture Diameter: ', sim['system_model.cruise.cruise.cruise.pod_model.aperture_diameter'])
print('PID: ', sim['system_model.cruise.cruise.cruise.heleeos.pid'])
print('HELEEOS Residual: ', sim['system_model.cruise.cruise.cruise.heleeos.power_residual'])
print('Wing Span Scaling: ', sim['system_parameterization.ffd_set.affine_section_properties_model.wing_span_scaling'])
print('Tail Span Scaling: ', sim['system_parameterization.ffd_set.affine_section_properties_model.tail_span_scaling'])
print('Left prop radius: ', sim['system_model.cruise.cruise.cruise.left_prop_bem_model.propeller_radius'])
print('Right prop radius: ', sim['system_model.cruise.cruise.cruise.right_prop_bem_model.propeller_radius'])
print('Trim Residual: ', sim['system_model.cruise.cruise.cruise.euler_eom_gen_ref_pt.trim_residual'])

# print('New Stress: ', sim['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.new_stress'])
# print('Wing Mesh: ', sim['system_representation.outputs_model.design_outputs_model.wing_beam_mesh'])

print('Buckle Ratio: ', sim['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.bkl'])

stress = sim['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.new_stress']
disp = sim['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.wing_beam_displacement']
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

plt.plot(stress)
plt.show()

plt.plot(disp[:,2])
plt.show()
"""




















# VISUALIZATION
import lsdo_dash.api as ld
caddee_viz = ld.caddee_plotters.CaddeeViz(
    caddee = caddee,
    system_m3l_model = system_model,
    design_configuration_map={},
    system_prefix='',
)
import csdl
rep = csdl.GraphRepresentation(caddee_csdl_model)

class TC2DB(ld.DashBuilder):
    def define(self, *args):


        geo_frame = self.add_frame(
            'full_geometry_frame',
            height_in=16.,
            width_in=24.,
            ncols=130,
            nrows = 80,
            wspace=0.4,
            hspace=0.4,
         )


        # +=+=+=+=+=+=+=+=+=+=+=+= HERE IS THE GEOMETRY PLOTTING STUFF +=+=+=+=+=+=+=+=+=+=+=+=
        def z_reverser(locations, state):
            # Reverse the z axis of the geometry because beam solver
            state[:,2] = -state[:,2]
            return locations, state

        center_x = 20  # eyeballing center x coordinate of geometry
        center_z = 8  # eyeballing center z coordinate of geometry
        camera_settings = {
            'pos': (-40, -22, 37),
            'viewup': (0, 0, 1),
            'focalPoint': (center_x+8, 0, center_z-20)
        }

        geo_elements = []
        geo_elements.append(caddee_viz.build_geometry_plotter(show = False, opacity = 0.3))
        geo_elements.append(caddee_viz.build_state_plotter(
            wing_displacement,
            rep = rep,
            displacements= 10,
            state_callback = z_reverser
        ))
        geo_frame[0:50,40:] = caddee_viz.build_vedo_renderer(geo_elements, camera_settings = camera_settings, show = 0)
        # +=+=+=+=+=+=+=+=+=+=+=+= HERE IS THE GEOMETRY PLOTTING STUFF +=+=+=+=+=+=+=+=+=+=+=+=

        # plot mass
        geo_frame[0:20,0:40] = ld.default_plotters.build_historic_plotter(
            'system_model.cruise.cruise.cruise.total_constant_mass_properties.total_mass',
            title = 'Total Mass',
            legend = False,
            # xlim = [0, 200], #SET XLIM YLIM HERE
            # ylim = [0, 200], #SET XLIM YLIM HERE
        )

        # plot trim residual
        geo_frame[25:45,0:40] = ld.default_plotters.build_historic_plotter(
            'system_model.cruise.cruise.cruise.euler_eom_gen_ref_pt.trim_residual',
            title = 'Trim Residual',
            legend = False,
            plot_type = 'semilogy',
        )

        # plot mesh vs span
        def plot_2D(ax_subplot, data_dict, data_dict_history):
            y_axis = data_dict['system_model.cruise.cruise.cruise.mass_model.wing_beam_tcap'].flatten()
            x_axis_but_wrong = data_dict['system_representation.outputs_model.design_outputs_model.wing_beam_mesh'][:,:,1].flatten()
            x_axis = []
            for i in range(x_axis_but_wrong.size-1):
                x_axis.append((x_axis_but_wrong[i] + x_axis_but_wrong[i+1])/2.0)
            ax_subplot.plot(x_axis, y_axis)
            ax_subplot.set_ylabel('Cap Thickness')
            # ax_subplot.set_ylim([0.0, 5.0])
            # ax_subplot.set_xlim([0.0, 5.0])
            ax_subplot.set_xlabel('Span (ft)')
            ax_subplot.set_title('Thickness (m) vs Span')

        geo_frame[50:70,0:40] = ld.BaseAxesPlotter(
            required_vars = [
                'system_representation.outputs_model.design_outputs_model.wing_beam_mesh',
                'system_model.cruise.cruise.cruise.mass_model.wing_beam_tcap'],
            plot_function = plot_2D,
        )

        # plot mesh vs stress
        def plot_2D(ax_subplot, data_dict, data_dict_history):
            y_axis = data_dict['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.new_stress']
            x_axis_but_wrong = data_dict['system_representation.outputs_model.design_outputs_model.wing_beam_mesh'][:,:,1].flatten()
            x_axis = []
            for i in range(x_axis_but_wrong.size-1):
                x_axis.append((x_axis_but_wrong[i] + x_axis_but_wrong[i+1])/2.0)
            ax_subplot.plot(x_axis, y_axis)
            ax_subplot.set_ylabel('Stress')
            # ax_subplot.set_ylim([0.0, 5.0])
            # ax_subplot.set_xlim([0.0, 5.0])
            ax_subplot.set_xlabel('Span (ft)')
            ax_subplot.set_title('Stress vs Span')

        geo_frame[50:70,45:85] = ld.BaseAxesPlotter(
            required_vars = [
                'system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.new_stress',
                'system_representation.outputs_model.design_outputs_model.wing_beam_mesh'],
            plot_function = plot_2D,
        )

        # plot something here?
        geo_frame[50:70,90:130] = ld.BaseAxesPlotter(
            required_vars = [
                'system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.new_stress',
                'system_representation.outputs_model.design_outputs_model.wing_beam_mesh'],
            plot_function = plot_2D,
        )

if __name__ == '__main__':
    dashbuilder = TC2DB()
    # dashbuilder = None

    # simulator
    sim = Simulator(caddee_csdl_model, analytics=True, dashboard=dashbuilder)
    # sim.run()

    # exit()
    prob = CSDLProblem(problem_name='lpc', simulator=sim)
    optimizer = SLSQP(prob, maxiter=200, ftol=1E-8)
    optimizer.solve()
    optimizer.print_results()

    print('Mach: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.cruise_mach_number'])
    print('U: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.u'])
    print('V: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.v'])
    print('W: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.w'])
    print('Pitch Angle: ', sim['system_model.cruise.cruise.cruise.cruise_ac_states_operation.cruise_pitch_angle'])
    print('Left Prop RPM: ', sim['system_model.cruise.cruise.cruise.left_prop_bem_model.rpm'])
    print('Right Prop RPM: ', sim['system_model.cruise.cruise.cruise.right_prop_bem_model.rpm'])
    print('H-Tail Act: ', sim['system_parameterization.ffd_set.rotational_section_properties_model.h_tail_act'])
    print('Total Mass: ', sim['system_model.cruise.cruise.cruise.total_constant_mass_properties.total_mass'])
    print('CG: ', sim['system_model.cruise.cruise.cruise.total_constant_mass_properties.total_cg_vector'])
    print('Total Forces: ', sim['system_model.cruise.cruise.cruise.total_forces_moments_model.total_forces'])
    print('Total Moments: ', sim['system_model.cruise.cruise.cruise.total_forces_moments_model.total_moments'])
    print('Left Prop Power: ', sim['system_model.cruise.cruise.cruise.battery_mass_model.left_prop_power'])
    print('Right Prop Power: ', sim['system_model.cruise.cruise.cruise.battery_mass_model.right_prop_power'])
    print('Battery Mass: ', sim['system_model.cruise.cruise.cruise.battery_mass_model.mass'])
    print('Wing Mass: ', sim['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.MassProp.struct_mass'])
    # print('Wing Beam Forces: ', sim['system_model.cruise.cruise.cruise.wing_eb_beam_force_mapping.wing_beam_forces'])
    print('Motor Mass: ', sim['system_model.cruise.cruise.cruise.motor_mass_model.mass'])
    print('Pod Mass: ', sim['system_model.cruise.cruise.cruise.pod_model.mass'])
    print('Fuselage Mass: ', sim['system_model.cruise.cruise.cruise.fuselage_mass_model.mass'])
    print('Payload Mass: ', sim['system_model.cruise.cruise.cruise.payload_mass_model.mass'])
    print('Aperture Diameter: ', sim['system_model.cruise.cruise.cruise.pod_model.aperture_diameter'])
    print('PID: ', sim['system_model.cruise.cruise.cruise.heleeos.pid'])
    print('HELEEOS Residual: ', sim['system_model.cruise.cruise.cruise.heleeos.power_residual'])
    print('Wing Span Scaling: ', sim['system_parameterization.ffd_set.affine_section_properties_model.wing_span_scaling'])
    print('Tail Span Scaling: ', sim['system_parameterization.ffd_set.affine_section_properties_model.tail_span_scaling'])
    print('Trim Residual: ', sim['system_model.cruise.cruise.cruise.euler_eom_gen_ref_pt.trim_residual'])

    # print('New Stress: ', sim['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.new_stress'])

    # print('Wing Mesh: ', sim['system_representation.outputs_model.design_outputs_model.wing_beam_mesh'])

    """
    f = sim['system_model.cruise.cruise.cruise.wing_eb_beam_force_mapping.wing_beam_forces']
    fz = f[:,:,2]
    lift = np.sum(fz)
    lift_mass = lift/9.81
    print(lift_mass)
    """

    stress = sim['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.new_stress']
    disp = sim['system_model.cruise.cruise.cruise.wing_eb_beam_model.Aframe.wing_beam_displacement']
    import matplotlib.pyplot as plt
    plt.rcParams.update(plt.rcParamsDefault)

    plt.plot(stress)
    plt.show()

    plt.plot(disp[:,2])
    plt.show()
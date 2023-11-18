import caddee.api as cd
import m3l
from python_csdl_backend import Simulator
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component, Rotor
import array_mapper as am
import lsdo_geo as lg
from lsdo_rotor.core.BEM_caddee.BEM_caddee import BEM, BEMMesh
from VAST.core.vast_solver import VASTFluidSover
from VAST.core.fluid_problem import FluidProblem
from aframe.core.mass import Mass, MassMesh
from VAST.core.generate_mappings_m3l import VASTNodalForces
from aframe.core.beam_module import EBBeam, LinearBeamMesh
import aframe.core.beam_module as ebbeam
import numpy as np
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
spatial_rep.refit_geometry(file_name=file_name)
# spatial_rep.plot(plot_types=['mesh'])



def build_lifting_surface(name, search_names):
    primitive_names = list(spatial_rep.get_primitives(search_names=search_names).keys())
    component = LiftingSurface(name=name, spatial_representation=spatial_rep, primitive_names=primitive_names)
    sys_rep.add_component(component)
    return component

wing = build_lifting_surface('wing', ['wing'])
# wing.plot()
tail = build_lifting_surface('tail', ['tail'])
# tail.plot()

def build_component(name, search_names):
    primitive_names = list(spatial_rep.get_primitives(search_names=search_names).keys())
    component = Component(name=name, spatial_representation=spatial_rep, primitive_names=primitive_names)
    sys_rep.add_component(component)
    return component

left_fuse = build_component('left_fuse', ['leftfuse'])
# left_fuse.plot()
right_fuse = build_component('right_fuse', ['rightfuse'])
# right_fuse.plot()

def build_rotor(name, search_names):
    primitive_names = list(spatial_rep.get_primitives(search_names=search_names).keys())
    component = Rotor(name=name, spatial_representation=spatial_rep, primitive_names=primitive_names)
    sys_rep.add_component(component)
    return component

right_prop = build_rotor('right_prop', ['rdisk'])
# right_prop.plot()
left_prop = build_rotor('left_prop', ['ldisk'])
# left_prop.plot()
right_rotor = build_rotor('right_rotor', ['rprop'])
# right_rotor.plot()
left_rotor = build_rotor('left_rotor', ['lprop'])
# right_rotor.plot()


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


# wing beam mesh
num_wing_beam = 15
leading_edge = wing.project(np.linspace(np.array([14.873 - offset, -31.966, 1.874]), np.array([14.873 - offset, 31.966, 1.874]), num_wing_beam), direction=np.array([0., 0., -1.]), plot=False)
trailing_edge = wing.project(np.linspace(np.array([16.622 + offset, -31.966, 1.813]), np.array([16.622 + offset, 31.966, 1.813]), num_wing_beam), direction=np.array([0., 0., -1.]), plot=False)
wing_beam = am.linear_combination(leading_edge,trailing_edge,1,start_weights=np.ones((num_wing_beam,))*0.75,stop_weights=np.ones((num_wing_beam,))*0.25)
width = am.norm((leading_edge - trailing_edge)*0.5)
# spatial_rep.plot_meshes([wing_beam])

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
beam_mesh = LinearBeamMesh(meshes = dict(
wing_beam = wing_beam,
wing_beam_width = width,
wing_beam_height = height,))

# pass the beam meshes to the aframe mass model:
beam_mass_mesh = MassMesh(meshes = dict(
wing_beam = wing_beam,
wing_beam_width = width,
wing_beam_height = height,))



# design scenario
design_scenario = cd.DesignScenario(name='cruise')
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
left_bem_model.set_module_input('propeller_radius', val=1.0)
left_bem_model.set_module_input('chord_cp', val=np.linspace(0.15, 0.05, 4), dv_flag=False)
left_bem_model.set_module_input('twist_cp', val=np.deg2rad(np.linspace(50, 15, 4)), dv_flag=False)
left_bem_model.set_module_input('thrust_vector', val=np.array([1., 0., 0.]))
left_bem_model.set_module_input('thrust_origin', val=np.array([30.5, -7, 0.75]))
left_bem_forces, left_bem_moments, _, _, _, _, _, _ = left_bem_model.evaluate(ac_states=ac_states)
cruise_model.register_output(left_bem_forces)
cruise_model.register_output(left_bem_moments)

right_bem_model = BEM(disk_prefix='right_prop', blade_prefix='right_prop', component=right_prop, mesh=right_prop_mesh)
right_bem_model.set_module_input('rpm', val=1500, dv_flag=True, lower=500, upper=3000, scaler=1e-3)
right_bem_model.set_module_input('propeller_radius', val=1.0)
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
    cl0=[0.25, 0]
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
beam_mass.set_module_input('wing_beam_tcap', val=0.001, dv_flag=True, lower=0.0005, upper=0.04, scaler=1E3)
beam_mass.set_module_input('wing_beam_tweb', val=0.001, dv_flag=True, lower=0.0005, upper=0.04, scaler=1E3)



cruise_wing_structural_nodal_displacements_mesh = am.vstack((wing_upper_surface_wireframe, wing_lower_surface_wireframe))
cruise_wing_aero_nodal_displacements_mesh = cruise_wing_structural_nodal_displacements_mesh
cruise_wing_structural_nodal_force_mesh = cruise_wing_structural_nodal_displacements_mesh
cruise_wing_aero_nodal_force_mesh = cruise_wing_structural_nodal_displacements_mesh

# dummy_b_spline_space = lg.BSplineSpace(name='dummy_b_spline_space', order=(3, 1), control_points_shape=((35, 1)))
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
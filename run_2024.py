import time
from lsdo_geo.core.geometry.geometry_functions import import_geometry
import m3l
import numpy as np
from python_csdl_backend import Simulator
import lsdo_geo as lg
from caddee.utils.helper_functions.geometry_helpers import (make_rotor_mesh, make_vlm_camber_mesh, make_1d_box_beam_mesh, 
                                                            compute_component_surface_area, BladeParameters)

geometry = lg.import_geometry('darpa4.stp', parallelize=True)
geometry.refit(parallelize=True, fit_resolution=(50, 50), num_coefficients=(30, 30),)
# geometry.plot()
m3l_model = m3l.Model()

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



# # wing beam mesh
# eps = 2
# num_wing_beam = 15
# leading_edge = wing.project(np.linspace(np.array([14.803-eps, -29.986, 1.455]), np.array([14.803-eps, 29.986, 1.455]), num_wing_beam), direction=np.array([0., 0., -1.]), plot=True)
# trailing_edge = wing.project(np.linspace(np.array([16.553+eps, -29.986, 1.447]), np.array([16.553+eps, 29.986, 1.447]), num_wing_beam), direction=np.array([0., 0., -1.]), plot=True)
# wing_beam = m3l.linear_combination(geometry.evaluate(leading_edge), geometry.evaluate(trailing_edge), 1, 
#                                    start_weights=np.ones((num_wing_beam,))*0.75, stop_weights=np.ones((num_wing_beam,))*0.25)
# width = m3l.norm((geometry.evaluate(leading_edge) - geometry.evaluate(trailing_edge))*0.5)
# # geometry.plot_meshes([wing_beam.reshape((-1,3))])


wing_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=wing,
    num_beam_nodes=15,
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

tail_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=htail,
    num_beam_nodes=9,
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
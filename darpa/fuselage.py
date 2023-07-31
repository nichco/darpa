import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l





class Fuselage(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        # self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)
        self.num_nodes = None
        self.parameters.declare('fuse_mass', default=100)
        self.parameters.declare('fuse_cg', default=np.array([0,0,0]))
        self.parameters.declare('fuse_cd', default=0.4)
        self.parameters.declare('area', default=1.0)

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        # self.struct_solver = self.parameters['struct_solver']
        self.compute_mass_properties = self.parameters['compute_mass_properties']
        self.fuse_mass = self.parameters['fuse_mass']
        self.fuse_cg = self.parameters['fuse_cg']
        self.fuse_cd = self.parameters['fuse_cd']
        self.area = self.parameters['area']

    def compute(self):
        fuse_mass = self.parameters['fuse_mass']
        fuse_cg = self.parameters['fuse_cg']
        fuse_cd = self.parameters['fuse_cd']
        area = self.parameters['area']
        csdl_model = FuselageCSDL(module=self, fuse_mass=fuse_mass, fuse_cg=fuse_cg, fuse_cd=fuse_cd, area=area)
        return csdl_model
    
    def evaluate(self):

        self.name = 'fuselage_mass_model'
        self.arguments = {}
        
        mass = m3l.Variable('mass', shape=(1,), operation=self)
        cg_vector = m3l.Variable('cg_vector', shape=(1,), operation=self)
        inertia_tensor = m3l.Variable('inertia_tensor', shape=(1,), operation=self)

        F = m3l.Variable('F', shape=(1,), operation=self)
        M = m3l.Variable('M', shape=(1,), operation=self)

        return mass, cg_vector, inertia_tensor, F, M




class FuselageCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('fuse_mass', default=50) # (kg)
        self.parameters.declare('fuse_cg', default=np.array([0,0,0]))
        self.parameters.declare('fuse_cd', default=0.4)
        self.parameters.declare('area', default=1.0)
        self.parameters.declare('fuse_inertia', default=np.zeros((3,3)))

    def define(self):
        fuse_mass = self.parameters['fuse_mass']
        fuse_cg = self.parameters['fuse_cg']
        cd = self.parameters['fuse_cd']
        area = self.parameters['area']
        fuse_inertia = self.parameters['fuse_inertia']

        fuse_mass = self.create_input('mass', fuse_mass)
        fuse_cg = self.create_input('cg_vector', fuse_cg)
        fuse_inertia = self.create_input('inertia_tensor', fuse_inertia)


        u = self.declare_variable('u', shape=(1,1))
        v = self.declare_variable('v', shape=(1,1))
        w = self.declare_variable('w', shape=(1,1))
        velocity = csdl.reshape((u**2 + v**2 + w**2)**0.5, (1,))

        rho = self.declare_variable('density')

        drag = 0.5*rho*(velocity**2)*area*cd
        self.register_output('fuse_drag', drag)

        zero_vec = self.declare_variable('zero_vec', shape=(3), val=0)
        self.register_output('M', 1*zero_vec) # no moments

        F = self.create_output('F', shape=(3), val=0)
        F[0] = -1*drag # Fx is the drag force (check signs!!)
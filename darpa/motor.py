import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l





class Motor(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        # self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)
        self.num_nodes = None
        self.parameters.declare('motor_power_density', default=3.1)
        self.parameters.declare('eta', default=0.75)
        self.parameters.declare('ratio', default=1.5)
        self.parameters.declare('motor_cg', default=np.array([0,0,0]))

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        # self.struct_solver = self.parameters['struct_solver']
        self.compute_mass_properties = self.parameters['compute_mass_properties']
        self.motor_power_density = self.parameters['motor_power_density']
        self.eta = self.parameters['eta']
        self.ratio = self.parameters['ratio']
        self.motor_cg = self.parameters['motor_cg']

    def compute(self):
        motor_power_density = self.parameters['motor_power_density']
        eta = self.parameters['eta']
        ratio = self.parameters['ratio']
        motor_cg = self.parameters['motor_cg']
        csdl_model = MotorCSDL(module=self, motor_power_density=motor_power_density, eta=eta, ratio=ratio, motor_cg=motor_cg)
        return csdl_model
    
    def evaluate(self):

        self.name = 'motor_mass_model'
        self.arguments = {}
        
        mass = m3l.Variable('mass', shape=(1,), operation=self)
        cg_vector = m3l.Variable('cg_vector', shape=(1,), operation=self)
        inertia_tensor = m3l.Variable('inertia_tensor', shape=(1,), operation=self)

        return mass, cg_vector, inertia_tensor




class MotorCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('motor_power_density', default=3.1) # (kg/kW)
        self.parameters.declare('eta', default=0.8)
        self.parameters.declare('ratio', default=1.5)
        self.parameters.declare('motor_cg', default=np.array([0,0,0]))
        self.parameters.declare('motor_inertia', default=np.zeros((3,3)))

    def define(self):
        motor_power_density = self.parameters['motor_power_density']
        eta = self.parameters['eta']
        ratio = self.parameters['ratio']
        motor_cg = self.parameters['motor_cg']
        motor_inertia = self.parameters['motor_inertia']

        required_power = self.declare_variable('required_power')/1000 # (kW)

        motor_mass = ratio*required_power*motor_power_density/eta
        self.register_module_output('mass', motor_mass)

        motor_cg = self.create_input('cg_vector', motor_cg)
        motor_inertia = self.create_input('inertia_tensor', motor_inertia)
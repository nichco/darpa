import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l





class Battery(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        # self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)
        self.num_nodes = None
        self.parameters.declare('esb', default=200)
        self.parameters.declare('eta', default=0.75)
        self.parameters.declare('endurance', default=0.5)
        self.parameters.declare('battery_cg', default=np.array([0,0,0]))

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        # self.struct_solver = self.parameters['struct_solver']
        self.compute_mass_properties = self.parameters['compute_mass_properties']
        self.esb = self.parameters['esb']
        self.eta = self.parameters['eta']
        self.endurance = self.parameters['endurance']
        self.battery_cg = self.parameters['battery_cg']

    def compute(self):
        esb = self.parameters['esb']
        eta = self.parameters['eta']
        endurance = self.parameters['endurance']
        battery_cg = self.parameters['battery_cg']
        csdl_model = BatteryCSDL(module=self, esb=esb, eta=eta, endurance=endurance, battery_cg=battery_cg)
        return csdl_model
    
    def evaluate(self):

        self.name = 'battery_mass_model'
        self.arguments = {}
        
        mass = m3l.Variable('mass', shape=(1,), operation=self)
        cg_vector = m3l.Variable('cg_vector', shape=(1,), operation=self)
        inertia_tensor = m3l.Variable('inertia_tensor', shape=(1,), operation=self)

        return mass, cg_vector, inertia_tensor




class BatteryCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('esb', default=200) # (wh/kg)
        self.parameters.declare('eta', default=0.75)
        self.parameters.declare('endurance', default=0.5) # (hr)
        self.parameters.declare('battery_cg', default=np.array([0,0,0]))
        self.parameters.declare('battery_inertia', default=np.zeros((3,3)))

    def define(self):
        esb = self.parameters['esb']
        eta = self.parameters['eta']
        endurance = self.parameters['endurance']
        battery_cg = self.parameters['battery_cg']
        battery_inertia = self.parameters['battery_inertia']

        p_req_left = self.declare_variable('required_power_left_rotor')
        p_req_right = self.declare_variable('required_power_right_rotor')

        p_req = p_req_left + p_req_right # the total power required

        battery_mass = (endurance*p_req/(esb*eta))
        self.register_output('mass', battery_mass)

        battery_cg = self.create_input('cg_vector', battery_cg)
        battery_inertia = self.create_input('inertia_tensor', battery_inertia)
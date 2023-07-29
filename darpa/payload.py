import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l





class Payload(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        # self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)
        self.num_nodes = None
        self.parameters.declare('payload', default=50)
        self.parameters.declare('payload_cg', default=np.array([0,0,0]))

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        # self.struct_solver = self.parameters['struct_solver']
        self.compute_mass_properties = self.parameters['compute_mass_properties']
        self.payload = self.parameters['payload']
        self.payload_cg = self.parameters['payload_cg']

    def compute(self):
        payload = self.parameters['payload']
        payload_cg = self.parameters['payload_cg']
        csdl_model = PayloadCSDL(module=self, payload=payload, payload_cg=payload_cg)
        return csdl_model
    
    def evaluate(self):

        self.name = 'payload_mass_model'
        self.arguments = {}
        
        mass = m3l.Variable('mass', shape=(1,), operation=self)
        cg_vector = m3l.Variable('cg_vector', shape=(1,), operation=self)
        inertia_tensor = m3l.Variable('inertia_tensor', shape=(1,), operation=self)

        return mass, cg_vector, inertia_tensor




class PayloadCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('payload', default=50) # (kg)
        self.parameters.declare('payload_cg', default=np.array([0,0,0]))
        self.parameters.declare('payload_inertia', default=np.zeros((3,3)))

    def define(self):
        payload_mass = self.parameters['payload']
        payload_cg = self.parameters['payload_cg']
        payload_inertia = self.parameters['payload_inertia']

        payload_mass = self.create_input('mass', payload_mass)
        payload_cg = self.create_input('cg_vector', payload_cg)
        payload_inertia = self.create_input('inertia_tensor', payload_inertia)
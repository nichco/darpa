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
        # self.parameters.declare('compute_mass_properties', default=True, types=bool)
        self.num_nodes = None
        self.parameters.declare('payload', default=50)

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        # self.struct_solver = self.parameters['struct_solver']
        # self.compute_mass_properties = self.parameters['compute_mass_properties']
        self.payload = self.parameters['payload']

    def compute(self):
        payload = self.parameters['payload']
        csdl_model = PayloadCSDL(module=self, payload=payload)
        return csdl_model
    
    def evaluate(self):

        self.name = 'payload'
        self.arguments = {}
        
        payload_mass = m3l.Variable('payload_mass', shape=(1,), operation=self)

        return payload_mass




class PayloadCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('payload', default=50) # (kg)

    def define(self):
        payload_mass = self.parameters['payload']

        payload_mass = self.create_input('payload_mass', payload_mass)
        # self.register_output('mass', 1*payload_mass)
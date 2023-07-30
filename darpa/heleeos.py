import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l
from smt.surrogate_models import KRG
import scipy.io as sio


heleeos_data = sio.loadmat('new_heleeos_data.mat')



class HELEEOS(m3l.ExplicitOperation):
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
        csdl_model = HELEEOSCSDL(module=self, payload=payload, payload_cg=payload_cg)
        return csdl_model
    
    def evaluate(self):

        self.name = 'payload_mass_model'
        self.arguments = {}
        
        mass = m3l.Variable('mass', shape=(1,), operation=self)
        cg_vector = m3l.Variable('cg_vector', shape=(1,), operation=self)
        inertia_tensor = m3l.Variable('inertia_tensor', shape=(1,), operation=self)

        return mass, cg_vector, inertia_tensor




class HELEEOSCSDL(ModuleCSDL):
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






class HELEEOS_Explicit(csdl.CustomExplicitOperation):
    def initialize(self):
        # Surrogate modelling for HELEEOS
        receiver_aperture_qp = np.array([0.1, 0.325, 0.55, 0.775, 1])
        horizontal_distance_qp = np.array([0, 250000, 500000, 750000, 1000000])
        receiver_altitude_qp = np.array([1000, 5750, 10500, 15250, 20000])

        receiver_aperture_input = np.reshape(np.einsum(
            'i...,pj...->ipj...', receiver_aperture_qp, np.ones((5, 5))), (125,))

        horizontal_distance_input = np.reshape(np.einsum(
            'i...,pj...->pij...', horizontal_distance_qp, np.ones((5, 5))), (125,))

        receiver_altitude_input = np.reshape(np.einsum(
            'i...,pj...->pji...', receiver_altitude_qp, np.ones((5, 5))), (125,))
        input_data = np.zeros([125, 3])
        input_data[:, 0] = receiver_aperture_input
        input_data[:, 1] = horizontal_distance_input
        input_data[:, 2] = receiver_altitude_input

        output_data = np.reshape(np.array(heleeos_data['ans']), (125, 1))

        sm = KRG(theta0=[1e-2], print_global=False, print_solver=False,)
        sm.set_training_values(input_data, output_data)
        sm.train()
        self.sm = sm

    def define(self):
        # inputs
        self.add_input('aperture_size', shape=(1,))
        self.add_input('horizontal_distance', shape=(1,))
        self.add_input('altitude', shape=(1,))

        # output: pid
        self.add_output('pid', shape=(1,))
        self.declare_derivatives('pid', 'aperture_size')
        self.declare_derivatives('pid', 'horizontal_distance')
        self.declare_derivatives('pid', 'altitude')

    def compute(self, inputs, outputs):

        # surrogate model
        data_input = np.zeros([1, 3])
        data_input[0][0] = inputs['aperture_size']
        data_input[0][1] = inputs['horizontal_distance']
        data_input[0][2] = inputs['altitude']
        #print(data_input)
        pid = self.sm.predict_values(data_input)
        #print(pid)
        outputs['pid'] = pid

    def compute_derivatives(self, inputs, derivatives):
        data_input = np.zeros([1, 3])
        data_input[0][0] = inputs['aperture_size']
        data_input[0][1] = inputs['horizontal_distance']
        data_input[0][2] = inputs['altitude']

        dp_d1 = self.sm.predict_derivatives(data_input, 0)
        dp_d2 = self.sm.predict_derivatives(data_input, 1)
        dp_d3 = self.sm.predict_derivatives(data_input, 2)

        derivatives['pid', 'aperture_size'] = 1*dp_d1
        derivatives['pid', 'horizontal_distance'] = 1*dp_d2
        derivatives['pid', 'altitude'] = 1*dp_d3
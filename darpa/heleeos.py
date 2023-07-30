import numpy as np
import csdl
import python_csdl_backend
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
        # self.parameters.declare('compute_mass_properties', default=True, types=bool)
        self.num_nodes = None

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        # self.struct_solver = self.parameters['struct_solver']
        # self.compute_mass_properties = self.parameters['compute_mass_properties']

    def compute(self):
        csdl_model = HELEEOSCSDL(module=self,)
        return csdl_model
    
    def evaluate(self):

        self.name = 'heleeos'
        self.arguments = {}
        
        pid = m3l.Variable('pid', shape=(1,), operation=self)

        return pid




class HELEEOSCSDL(ModuleCSDL):
    def initialize(self):
        pass

    def define(self):
        # aperture_size = self.declare_variable('aperture_size')  # aperture_size (m)
        aperture_size = self.register_module_input('aperture_size', shape=(1,), promotes=True)
        # self.print_var(aperture_size)

        horizontal_distance = self.register_module_input('horizontal_distance', shape=(1,), computed_upstream=False)
        #horizontal_distance = self.create_input('horizontal_distance',val=300000)  # engagement distance (m)
        # self.print_var(horizontal_distance)
        
        altitude = self.register_module_input('altitude', shape=(1,), vectorized=True)


        u = self.register_module_input('u', shape=(1,1), vectorized=True)
        v = self.register_module_input('v', shape=(1,1), vectorized=True)
        w = self.register_module_input('w', shape=(1,1), vectorized=True)

        velocity = self.register_output('velocity', csdl.reshape((u**2 + v**2 + w**2)**0.5, (1,)))

        pid = csdl.custom(aperture_size, horizontal_distance, altitude, op=HELEEOS_Explicit())
        self.register_output('pid', pid)

        # the laser power must be greater than the required power:
        required_power = self.declare_variable('required_power')

        power_residual = pid - required_power # must be >= 0
        self.register_output('power_residual', power_residual)






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






if __name__ == '__main__':
    name = 'cruise'
    sim = python_csdl_backend.Simulator(HELEEOS())
    sim['aperture_size'] = 0.5
    sim['altitude'] = 10000
    sim['u'] = 100
    sim['v'] = 0
    sim['w'] = 0

    sim.run()

    pid = sim['pid']

    print('pid = ', pid)
import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l





class Pod(m3l.ExplicitOperation):
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
        csdl_model = PodCSDL(module=self, payload=payload, payload_cg=payload_cg)
        return csdl_model
    
    def evaluate(self):

        self.name = 'payload_mass_model'
        self.arguments = {}
        
        mass = m3l.Variable('mass', shape=(1,), operation=self)
        cg_vector = m3l.Variable('cg_vector', shape=(1,), operation=self)
        inertia_tensor = m3l.Variable('inertia_tensor', shape=(1,), operation=self)

        return mass, cg_vector, inertia_tensor




class PodCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('cd', default=0.4)
        self.parameters.declare('length', default=2.0) # (m)
        self.parameters.declare('density', default=75) # (kg/m^3)
        self.parameters.declare('pod_cg', default=np.array([0,0,0]))
        self.parameters.declare('pod_inertia', default=np.zeros((3,3)))

    def define(self):
        cd = self.parameters['cd']
        length = self.parameters['length']
        density = self.parameters['density']
        pod_cg = self.parameters['pod_cg']
        pod_inertia = self.parameters['pod_inertia']


        d = self.register_module_input('aperture_diameter',shape=(1,), computed_upstream=False)

        # velocity:
        # v = self.declare_variable('velocity')
        u = self.register_module_input('u',shape=(1,1),vectorized=True)
        v = self.register_module_input('v',shape=(1,1),vectorized=True)
        w = self.register_module_input('w',shape=(1,1),vectorized=True)

        velocity = csdl.reshape((u**2 + v**2 + w**2)**0.5, (1,))

        # density:
        #rho = self.declare_variable('rho')
        density = self.register_module_input('density', shape=(1,1), vectorized=True)
        # self.print_var(density)
        rho = csdl.reshape(density, (1,))

        # the pod's frontal area:
        area = np.pi*(d/2)**2


        # compute the pod drag
        drag = 0.5*rho*(velocity**2)*area*cd
        self.register_output('pod_drag', drag)

        zero_vec = self.declare_variable('zero_vec',shape=(3),val=0)
        self.register_output('M',1*zero_vec) # no moments
        F = self.create_output('F',shape=(3),val=0) # (about 0,0,0)
        F[0] = -1*drag # Fx is the drag force (check signs!!)
        #self.print_var(F)

        volume = area*length
        pod_mass = volume*density




        pod_mass = self.create_input('mass', pod_mass)
        pod_cg = self.create_input('cg_vector', pod_cg)
        pod_inertia = self.create_input('inertia_tensor', pod_inertia)
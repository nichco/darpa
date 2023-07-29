import csdl
import python_csdl_backend
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.sizing_group.sizing_models.sizing_model import SizingModel
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
import numpy as np





class PayloadModel(SizingModel):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        #self.parameters.declare('mesh', default=None)
        self.num_nodes = None
    def _assemble_csdl(self):
        csdl_model = Payload(module=self,)
        return csdl_model


class PayloadMechModel(MechanicsModel):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)
        self.num_nodes = None
    def _assemble_csdl(self):
        csdl_model = Payload(module=self,)
        return csdl_model
    
class PayloadMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict)



class Payload(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('payload',default=50) # (kg)
    def define(self):
        payload = self.parameters['payload']

        payload_mass = self.create_input('payload_mass',payload)
        self.register_output('mass', 1*payload_mass)


        # compute the cg:
        pay_cgx = self.register_module_input('pay_cgx',shape=(1,),computed_upstream=False)
        pay_cgy = self.create_input('pay_cgy',val=0)
        pay_cgz = self.create_input('pay_cgz',val=0.218)
        self.register_module_output('cgx',1*pay_cgx)
        self.register_module_output('cgy',1*pay_cgy)
        self.register_module_output('cgz',1*pay_cgz)

        # compute the moi:
        x = pay_cgx
        y = pay_cgy
        z = pay_cgz

        rxx = y**2 + z**2
        ryy = x**2 + z**2
        rzz = x**2 + y**2
        rxz = x*z

        ixx = payload_mass*rxx
        iyy = payload_mass*ryy
        izz = payload_mass*rzz
        ixz = payload_mass*rxz

        self.register_module_output('ixx',ixx)
        self.register_module_output('iyy',iyy)
        self.register_module_output('izz',izz)
        self.register_module_output('ixz',ixz)

        zero = self.declare_variable('zero_vec',shape=(3),val=0)
        self.register_module_output('F', 1*zero)
        self.register_module_output('M', 1*zero)
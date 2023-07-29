import csdl
import python_csdl_backend
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.sizing_group.sizing_models.sizing_model import SizingModel
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
import numpy as np




"""
class BatteryModel(SizingModel):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.num_nodes = None
    def _assemble_csdl(self):
        csdl_model = Battery()
        return csdl_model
"""

class BatteryModel(MechanicsModel):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)
        self.num_nodes = None
    def _assemble_csdl(self):
        csdl_model = Battery(module=self,)
        return csdl_model
    


class BatteryMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict)




class Battery(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('battery_specific_energy',default=200) # (wh/kg)
        self.parameters.declare('efficiency',default=0.75)
        self.parameters.declare('required_endurance',default=0.5) # (hr)
        self.parameters.declare('system_mass',default=10)
    def define(self):
        esb = self.parameters['battery_specific_energy']
        eta = self.parameters['efficiency']
        required_endurance = self.parameters['required_endurance']
        system_mass = self.parameters['system_mass']



        p_req_left = self.declare_variable('required_power_left_rotor') # required power for the left rotor
        p_req_right = self.declare_variable('required_power_right_rotor') # required power for the left rotor

        p_req = p_req_left + p_req_right # the total power required

        battery_mass = (required_endurance*p_req/(esb*eta)) + system_mass# (kg)
        self.register_output('mass', 1*battery_mass)
        self.register_output('battery_mass', 1*battery_mass)



        # compute the cg:
        batt_cgx = self.register_module_input('batt_cgx',shape=(1,),computed_upstream=False)
        batt_cgy = self.create_input('batt_cgy',val=0)
        batt_cgz = self.create_input('batt_cgz',val=0.218)
        #r_batt = self.create_input('r_batt',shape=(3),val=[14.5/3.281,0,0])
        self.register_module_output('cgx',1*batt_cgx)
        self.register_module_output('cgy',1*batt_cgy)
        self.register_module_output('cgz',1*batt_cgz)



        # compute the moi:
        x = batt_cgx
        y = batt_cgy
        z = batt_cgz

        rxx = y**2 + z**2
        ryy = x**2 + z**2
        rzz = x**2 + y**2
        rxz = x*z

        ixx = battery_mass*rxx
        iyy = battery_mass*ryy
        izz = battery_mass*rzz
        ixz = battery_mass*rxz

        self.register_module_output('ixx',ixx)
        self.register_module_output('iyy',iyy)
        self.register_module_output('izz',izz)
        self.register_module_output('ixz',ixz)



        zero = self.declare_variable('zero_vec',shape=(3),val=0)
        self.register_module_output('F', 1*zero)
        self.register_module_output('M', 1*zero)
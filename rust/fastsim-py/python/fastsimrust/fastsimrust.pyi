from typing_extensions import Self


class VehicleThermal(object):
    fc_l: float
    cab_c_kj__k: float
    fc_exp_minimum: float
    orphaned: bool
    fc_coeff_from_comb: float
    tstat_te_sto_deg_c: float
    fc_c_kj__k: float
    tstat_te_delta_deg_c: float
    cab_l_width: float
    cab_r_to_amb: float
    cab_l_length: float
    cat_c_kj__K: float
    exhport_ha_int: float
    cat_l: float
    fc_exp_offset: float
    cat_htc_to_amb_stop: float
    cat_fc_eta_coeff: float
    fc_htc_to_amb_stop: float
    exhport_ha_to_amb: float
    fc_exp_lag: float
    ess_c_kj_k: float
    cat_te_lightoff_deg_c: float
    cab_htc_to_amb_stop: float
    exhport_c_kj__k: float
    rad_eps: float
    ess_htc_to_amb: float

    @classmethod
    def default(cls) -> Self: ...

    def copy(self) -> Self: ...

    def set_cabin_model_internal(
        self,
        te_set_deg_c: float,
        p_cntrl_kw_per_deg_c: float,
        i_cntrl_kw_per_deg_c_scnds: float,
        i_cntrl_max_kw: float,
        te_deadband_deg_c: float,
    ): ...

    def set_cabin_model_external(self): ...

    def set_fc_model_internal_exponential(
        self,
        offset: float,
        lag: float,
        minimum: float,
        fc_temp_eff_component: str,
    ): ...

    def reset_orphaned(self): ...

    def to_json(self) -> str: ...

    @classmethod
    def from_json(cls, json_str: str) -> Self: ...

    def to_file(self, filename: str) -> Self: ...

    @classmethod
    def from_file(cls, filename: str) -> Self: ...

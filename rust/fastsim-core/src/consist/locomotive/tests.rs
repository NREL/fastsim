use super::*;
use crate::si;

#[test]
fn test_conv_loco() {
    let mut loco = Locomotive::default();
    loco.assert_limits = false; // todo: make this true and fix test

    match loco.loco_type {
        LocoType::ConventionalLoco(_) => {}
        _ => panic!("Invalid loco type for conventional loco test!"),
    }

    assert_eq!(loco.state.pwr_out_max, si::Power::ZERO);
    assert_eq!(loco.state.pwr_rate_out_max, si::PowerRate::ZERO);
    assert_eq!(loco.state.pwr_regen_max, si::Power::ZERO);
    loco.set_cur_pwr_max_out(None, 1.0 * uc::S).unwrap();
    assert!(loco.state.pwr_out_max > si::Power::ZERO);
    assert!(loco.state.pwr_rate_out_max > si::PowerRate::ZERO);
    assert!(loco.state.pwr_regen_max == si::Power::ZERO);

    assert_eq!(loco.state.energy_out, si::Energy::ZERO);
    loco.solve_energy_consumption(uc::W * 900e3, uc::S * 1.0, Some(true))
        .unwrap();
    assert!(loco.state.energy_out > si::Energy::ZERO);
    if let LocoType::ConventionalLoco(lt) = &loco.loco_type {
        assert!(lt.edrv.state.energy_elec_dyn_brake == si::Energy::ZERO);
    }
    loco.solve_energy_consumption(uc::W * -900e3, uc::S * 1.0, Some(true))
        .unwrap();
    if let LocoType::ConventionalLoco(lt) = loco.loco_type {
        assert!(lt.edrv.state.energy_elec_dyn_brake > si::Energy::ZERO);
    }
}

#[test]
fn test_hybrid_loco() {
    let mut loco = Locomotive::default_hybrid_electric_loco();

    assert_eq!(loco.state.pwr_out_max, si::Power::ZERO);
    assert_eq!(loco.state.pwr_rate_out_max, si::PowerRate::ZERO);
    assert_eq!(loco.state.pwr_regen_max, si::Power::ZERO);
    loco.set_cur_pwr_max_out(None, 1.0 * uc::S).unwrap();
    assert!(loco.state.pwr_out_max > si::Power::ZERO);
    assert!(loco.state.pwr_rate_out_max > si::PowerRate::ZERO);
    assert!(loco.state.pwr_regen_max > si::Power::ZERO);

    assert_eq!(loco.state.energy_out, si::Energy::ZERO);
    loco.solve_energy_consumption(uc::W * 1e6, uc::S * 1.0, Some(true))
        .unwrap();
    assert!(loco.state.energy_out > si::Energy::ZERO);
}

#[test]
fn test_battery_electric_loco() {
    let mut loco = Locomotive::default_battery_electric_loco();

    assert_eq!(loco.state.pwr_out_max, si::Power::ZERO);
    assert_eq!(loco.state.pwr_rate_out_max, si::PowerRate::ZERO);
    assert_eq!(loco.state.pwr_regen_max, si::Power::ZERO);
    loco.set_cur_pwr_max_out(None, 1.0 * uc::S).unwrap();
    assert!(loco.state.pwr_out_max > si::Power::ZERO);
    assert!(loco.state.pwr_rate_out_max > si::PowerRate::ZERO);
    assert!(loco.state.pwr_regen_max > si::Power::ZERO);

    assert_eq!(loco.state.energy_out, si::Energy::ZERO);
    loco.solve_energy_consumption(uc::W * 1e6, uc::S * 1.0, Some(true))
        .unwrap();
    assert!(loco.state.energy_out > si::Energy::ZERO);
}

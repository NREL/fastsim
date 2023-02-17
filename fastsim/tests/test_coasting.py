"""
Tests that check the drive cycle modification functionality.
"""
import unittest
from typing import Union, List, Optional

import numpy as np
from numpy.polynomial import Chebyshev

import fastsim
from fastsim.rustext import RUST_AVAILABLE, warn_rust_unavailable
if RUST_AVAILABLE:
    import fastsimrust as fsr

DO_PLOTS = False
USE_PYTHON = True
USE_RUST = True

if USE_RUST and not RUST_AVAILABLE:
    warn_rust_unavailable(__file__)

from fastsim.auxiliaries import set_nested_values

def make_coasting_plot(
    cyc0:fastsim.cycle.Cycle,
    cyc:fastsim.cycle.Cycle,
    use_mph:bool=False,
    title:Optional[str]=None,
    save_file:Optional[str]=None,
    do_show:bool=False,
    verbose:bool=False,
    gap_offset_m:float=0.0,
    coast_brake_start_speed_m_per_s:Optional[float]=None):
    """
    - cyc0: Cycle, the reference cycle (the "shadow trace" or "lead vehicle")
    - cyc: Cycle, the actual cycle driven
    - use_mph: Bool, if True, plot in miles per hour, else m/s
    - title: None or string, if string, set the title
    - save_file: (Or None string), if specified, save the file to disk
    - do_show: Bool, whether to show the file or not
    - verbose: Bool, if True, prints out
    - gap_offset_m: number, an offset to apply to the gap metrics (m)
    - coast_brake_start_speed_m_per_s: None | number, if supplied, plots the coast-start speed (m/s)
    RETURN: None
    - saves creates the given file and shows it
    """
    import matplotlib.pyplot as plt
    ts_orig = np.array(cyc0.time_s)
    vs_orig = np.array(cyc0.mps)
    m = fastsim.params.MPH_PER_MPS if use_mph else 1.0
    ds_orig = fastsim.cycle.trapz_step_distances(cyc0).cumsum()
    ts = np.array(cyc.time_s)
    vs = np.array(cyc.mps)
    ds = fastsim.cycle.trapz_step_distances(cyc).cumsum()
    gaps = ds_orig - ds
    speed_units = "mph" if use_mph else "m/s"
    fontsize=10
    (fig, axs) = plt.subplots(nrows=3)
    ax = axs[1]
    if coast_brake_start_speed_m_per_s is not None:
        ax.plot([ts[0], ts[-1]], [coast_brake_start_speed_m_per_s, coast_brake_start_speed_m_per_s], 'y--', label='coast brake start speed')
    ax.plot(ts_orig, vs_orig * m, 'gray', label='lead')
    ax.plot(ts, vs * m, 'b-', lw=2, label='cav')
    ax.plot(ts, vs * m, 'r.', ms=1)
    ax.set_xlabel('Elapsed Time (s)', fontsize=fontsize)
    ax.set_ylabel(f'Speed ({speed_units})', fontsize=fontsize)
    ax.legend(loc=0, prop={'size': 6})
    ax = axs[2]
    ax_right = ax.twinx()
    ax_right.plot(ds_orig, np.array(cyc0.grade) * 100, 'y--', label='grade')
    ax_right.set_ylabel('Grade (%)', fontsize=fontsize)
    ax_right.grid(False)
    ax.set_zorder(ax_right.get_zorder()+1)
    ax.grid(False)
    ax.set_frame_on(False)
    ax.plot(ds_orig, vs_orig * m, 'gray', label='lead')
    ax.plot(ds, vs * m, 'b-', lw=2, label='cav')
    ax.plot(ds, vs * m, 'r.', ms=1)
    ax.set_xlabel('Distance Traveled (m)', fontsize=fontsize)
    ax.set_ylabel(f'Speed ({speed_units})', fontsize=fontsize)
    ax = axs[0]
    ax.plot(ts_orig, gaps + gap_offset_m, 'gray', label='lead')
    ax.set_xlabel('Elapsed Time (s)', fontsize=fontsize)
    ax.set_ylabel('Gap (m)', fontsize=fontsize)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if verbose:
        print(f'Distance Traveled for Coasting Vehicle: {cyc.dist_m.sum()} m')
        print(f'Distance Traveled for Cycle           : {cyc0.dist_m.sum()} m')
    if save_file is not None:
        fig.savefig(save_file, dpi=300)
    if do_show:
        plt.show()
    plt.close()


def make_dvdd_plot(
    cyc:fastsim.cycle.Cycle,
    coast_to_break_speed_m__s:Union[float, None]=None,
    use_mph:bool=False,
    save_file:Union[None,str]=None,
    do_show:bool=False,
    curve_fit:bool=True,
    additional_xs:Union[None,List[float]]=None,
    additional_ys:Union[None,List[float]]=None):
    """
    Create a change in speed (dv) by change in distance (dd) plot
    """
    if coast_to_break_speed_m__s is None:
        coast_to_break_speed_m__s = 5.0 # m/s
    TOL = 1e-6
    import matplotlib.pyplot as plt
    dvs = np.array(cyc.mps)[1:] - np.array(cyc.mps)[:-1]
    vavgs = 0.5 * (np.array(cyc.mps)[1:] + np.array(cyc.mps)[:-1])
    grades = np.array(cyc.grade)[:-1]
    unique_grades = np.sort(np.unique(grades))
    dds = vavgs * np.array(cyc.time_s)[1:]
    mask = dds < TOL
    dds[mask] = 0.5 * TOL
    ks = dvs / dds
    ks[mask] = 0.0

    fig, ax = plt.subplots()
    m = fastsim.params.MPH_PER_MPS if use_mph else 1.0
    speed_units = "mph" if use_mph else "m/s"
    c1 = None
    c2 = None
    c3 = None
    for g in unique_grades:
        grade_pct = g * 100.0 # percentage
        mask = np.logical_and(
            np.logical_and(
                grades == g,
                ks < 0.0
            ),
            vavgs >= coast_to_break_speed_m__s
        )
        ax.plot(vavgs[mask] * m, np.abs(ks[mask]), label=f'{grade_pct}%')
        if curve_fit and sum(mask) > 3:
            c1 = Chebyshev.fit(vavgs[mask], ks[mask], deg=1)
            c2 = Chebyshev.fit(vavgs[mask], ks[mask], deg=2)
            c3 = Chebyshev.fit(vavgs[mask], ks[mask], deg=3)
            print("FITS:")
            print(f"{g}: {c3}")
            colors = ['r', 'k', 'g']
            for deg, c in enumerate([c1, c2, c3]):
                if deg == 2:
                    xs, ys = c.linspace(n=25)
                    ax.plot(
                        xs,
                        np.abs(ys),
                        marker='.',
                        markerfacecolor=colors[deg],
                        markeredgecolor=colors[deg],
                        linestyle='None',
                        label=f'{grade_pct}% (fit {deg+1})')
    if additional_xs is not None and additional_ys is not None:
        ax.plot(additional_xs, additional_ys, 'r--', label='custom')
    ax.legend()
    ax.set_xlabel(f'Average Step Speed ({speed_units})')
    ax.set_ylabel('k-factor (m/s / m)')
    title = 'K by Speed and Grade'
    ax.set_title(title)
    fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, dpi=300)
    if do_show:
        plt.show()
    plt.close()


class TestCoasting(unittest.TestCase):
    def setUp(self) -> None:
        fastsim.utils.disable_logging()
        # create a trapezoidal trip shape
        # initial ramp: d(t=10s) = 100 meters distance
        # distance by time in constant speed region = d(t) = 100m + (t - 10s) * 20m/s 
        # distance of stop: 100m + (45s - 10s) * 20m/s + 0.5 * (55s - 45s) * 20m/s = 900m
        self.distance_of_stop_m = 900.0
        trapz = fastsim.cycle.make_cycle(
            [0.0, 10.0, 45.0, 55.0, 150.0],
            [0.0, 20.0, 20.0, 0.0, 0.0],
        )
        trapz = fastsim.cycle.resample(trapz, new_dt=1.0)
        if USE_PYTHON:
            self.trapz = fastsim.cycle.Cycle.from_dict(trapz)
            self.veh = fastsim.vehicle.Vehicle.from_vehdb(5)
            self.sim_drive = fastsim.simdrive.SimDrive(self.trapz, self.veh)
            self.sim_drive_coast = fastsim.simdrive.SimDrive(self.trapz, self.veh)
            self.sim_drive_coast.sim_params.coast_allow = True
            self.sim_drive_coast.sim_params.coast_start_speed_m_per_s = 17.0
        if RUST_AVAILABLE and USE_RUST:
            self.ru_trapz = fastsim.cycle.Cycle.from_dict(trapz).to_rust()
            self.ru_veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
            self.ru_sim_drive = fastsim.simdrive.RustSimDrive(self.ru_trapz, self.ru_veh)
            self.ru_sim_drive_coast = fastsim.simdrive.RustSimDrive(self.ru_trapz, self.ru_veh)
            self.ru_sim_drive_coast.sim_params = set_nested_values(self.ru_sim_drive_coast.sim_params,
                coast_allow=True,
                coast_start_speed_m_per_s=17.0,
            )
        return super().setUp()
    
    def test_cycle_reported_distance_traveled_m(self):
        ""
        # At the entering of constant-speed region
        if USE_PYTHON:
            idx = 10
            expected_time_s = 10.0
            t = self.trapz.time_s[idx]
            self.assertAlmostEqual(expected_time_s, t)
            expected_distance_m = 100.0
            dist_m = fastsim.cycle.trapz_step_start_distance(self.trapz, idx+1)
            self.assertAlmostEqual(expected_distance_m, dist_m)
            # At t=20s
            idx = 20
            expected_time_s = 20.0
            t = self.trapz.time_s[idx]
            self.assertAlmostEqual(expected_time_s, t)
            expected_distance_m = 300.0 # 100m + (20s - 10s) * 20m/s
            dist_m = fastsim.cycle.trapz_step_start_distance(self.trapz, idx + 1)
            self.assertAlmostEqual(expected_distance_m, dist_m)
            dds = self.trapz.calc_distance_to_next_stop_from(dist_m)
            dds_expected_m = 900 - dist_m
            self.assertAlmostEqual(dds_expected_m, dds, msg="Error in python version")
        if RUST_AVAILABLE and USE_RUST:
            idx = 10
            expected_time_s = 10.0
            t = self.ru_trapz.time_s[idx]
            self.assertAlmostEqual(expected_time_s, t)
            expected_distance_m = 100.0
            dist_m = fastsim.cycle.trapz_step_start_distance(self.ru_trapz, idx + 1)
            self.assertAlmostEqual(
                expected_distance_m, dist_m,
                msg=f"Error in Rust version, Rust dist: {dist_m}")
            # At t=20s
            idx = 20
            expected_time_s = 20.0
            t = self.ru_trapz.time_s[idx]
            self.assertAlmostEqual(expected_time_s, t)
            expected_distance_m = 300.0 # 100m + (20s - 10s) * 20m/s
            dist_m = fastsim.cycle.trapz_step_start_distance(self.trapz, idx + 1)
            self.assertAlmostEqual(expected_distance_m, dist_m, msg="Error in Rust version")
            dds = self.trapz.calc_distance_to_next_stop_from(dist_m)
            dds_expected_m = 900 - dist_m
            self.assertAlmostEqual(dds_expected_m, dds, msg="Error in Rust version")

    def test_cycle_modifications_with_constant_jerk(self):
        ""
        if USE_PYTHON:
            idx = 20
            n = 10
            accel = -1.0
            jerk = 0.1
            trapz = self.trapz.copy()
            self.assertEqual(type(trapz), fastsim.cycle.Cycle)
            trapz.modify_by_const_jerk_trajectory(idx, n, jerk, -1.0)
            self.assertNotEqual(self.trapz.mps[idx], trapz.mps[idx])
            self.assertEqual(len(self.trapz.mps), len(trapz.mps))
            self.assertTrue(self.trapz.mps[idx] > trapz.mps[idx])
            v0 = trapz.mps[idx-1]
            v = v0
            a = accel
            for i in range(len(self.trapz.time_s)):
                msg = f"i: {i}; idx: {idx}; idx+n: {idx+n}"
                if i < idx or i >= idx+n:
                    self.assertEqual(self.trapz.mps[i], trapz.mps[i], msg)
                else:
                    dt = trapz.dt_s_at_i(idx)
                    a_expected = fastsim.cycle.accel_for_constant_jerk(i - idx, accel, jerk, dt)
                    a = accel + (i - idx) * jerk * dt
                    v += a * dt
                    msg += f" a: {a}, v: {v}, dt: {dt}"
                    self.assertAlmostEqual(a_expected, a, msg=msg)
                    self.assertAlmostEqual(v, trapz.mps[i], msg=msg)
        if RUST_AVAILABLE and USE_RUST:
            idx = 20
            n = 10
            accel = -1.0
            jerk = 0.1
            trapz = self.ru_trapz.copy()
            self.assertEqual(type(trapz), fsr.RustCycle)
            trapz.modify_by_const_jerk_trajectory(idx, n, jerk, -1.0)
            self.assertNotEqual(self.trapz.mps[idx], trapz.mps[idx])
            self.assertEqual(len(self.trapz.mps), len(trapz.mps))
            self.assertTrue(self.trapz.mps[idx] > trapz.mps[idx])
            v0 = trapz.mps[idx-1]
            v = v0
            a = accel
            for i in range(len(self.trapz.time_s)):
                msg = f"i: {i}; idx: {idx}; idx+n: {idx+n}"
                if i < idx or i >= idx+n:
                    self.assertEqual(self.trapz.mps[i], trapz.mps[i], msg)
                else:
                    dt = trapz.dt_s_at_i(idx)
                    a_expected = fastsim.cycle.accel_for_constant_jerk(i - idx, accel, jerk, dt)
                    a = accel + (i - idx) * jerk * dt
                    v += a * dt
                    msg += f" a: {a}, v: {v}, dt: {dt}"
                    self.assertAlmostEqual(a_expected, a, msg=msg)
                    self.assertAlmostEqual(v, trapz.mps[i], msg=msg)
    
    def test_that_cycle_modifications_work_as_expected(self):
        ""
        if USE_PYTHON:
            idx = 20
            n = 10
            accel = -1.0
            jerk = 0.0
            trapz = self.trapz.copy()
            trapz.modify_by_const_jerk_trajectory(idx, n, jerk, -1.0)
            self.assertNotEqual(self.trapz.mps[idx], trapz.mps[idx])
            self.assertEqual(len(self.trapz.mps), len(trapz.mps))
            self.assertTrue(self.trapz.mps[idx] > trapz.mps[idx])
            for i in range(len(self.trapz.time_s)):
                msg = f"i: {i}; idx: {idx}; idx+n: {idx+n}"
                if i < idx or i >= idx+n:
                    self.assertEqual(self.trapz.mps[i], trapz.mps[i], msg=msg)
                else:
                    self.assertAlmostEqual(
                        self.trapz.mps[idx-1] + (accel * (i - idx + 1)),
                        trapz.mps[i],
                        msg=msg,
                    )
        if RUST_AVAILABLE and USE_RUST:
            idx = 20
            n = 10
            accel = -1.0
            jerk = 0.0
            trapz = self.ru_trapz.copy()
            trapz.modify_by_const_jerk_trajectory(idx, n, jerk, -1.0)
            self.assertNotEqual(self.ru_trapz.mps[idx], trapz.mps[idx])
            self.assertEqual(len(self.ru_trapz.mps), len(trapz.mps))
            self.assertTrue(self.ru_trapz.mps[idx] > trapz.mps[idx])
            for i in range(len(self.ru_trapz.time_s)):
                msg = f"i: {i}; idx: {idx}; idx+n: {idx+n}"
                if i < idx or i >= idx+n:
                    self.assertEqual(self.ru_trapz.mps[i], trapz.mps[i], msg=msg)
                else:
                    self.assertAlmostEqual(
                        self.ru_trapz.mps[idx-1] + (accel * (i - idx + 1)),
                        trapz.mps[i],
                        msg=msg,
                    )
    
    def test_that_we_can_coast(self):
        "Test the standard interface to Eco-Approach for 'free coasting'"
        if USE_PYTHON:
            self.assertFalse(self.sim_drive.impose_coast.any(), "All impose_coast starts out False")
            self.sim_drive.init_for_step(init_soc=self.veh.max_soc)
            while self.sim_drive_coast.i < len(self.trapz.time_s):
                self.sim_drive_coast.sim_drive_step()
            max_trace_miss_coast_m__s = np.absolute(self.trapz.mps - self.sim_drive_coast.mps_ach).max()
            self.assertTrue(max_trace_miss_coast_m__s > 1.0, f"Max trace miss: {max_trace_miss_coast_m__s} m/s")
            self.assertFalse(self.sim_drive_coast.impose_coast[0])
            if DO_PLOTS:
                make_coasting_plot(
                    self.sim_drive_coast.cyc0,
                    self.sim_drive_coast.cyc,
                    use_mph=False,
                    title="Test That We Can Coast",
                    save_file='junk-test-that-we-can-coast.png')
        if RUST_AVAILABLE and USE_RUST:
            self.assertFalse(
                self.ru_sim_drive.sim_params.coast_allow,
                "coast_allow is False by default")
            self.assertTrue(
                self.ru_sim_drive_coast.sim_params.coast_allow,
                "Ensure coast_allow is True")
            self.assertFalse(
                self.ru_sim_drive_coast.sim_params.coast_allow_passing,
                "Passing during coast is not allowed")
            self.assertEqual(17.0, self.ru_sim_drive_coast.sim_params.coast_start_speed_m_per_s)
            self.assertFalse(
                np.array(self.ru_sim_drive.impose_coast).any(),
                "All impose_coast starts out False")
            self.ru_sim_drive.init_for_step(init_soc=self.ru_veh.max_soc)
            while self.ru_sim_drive_coast.i < len(self.ru_trapz.time_s):
                self.ru_sim_drive_coast.sim_drive_step()
            max_trace_miss_coast_m__s = np.absolute(
                np.array(self.ru_trapz.mps) - np.array(self.ru_sim_drive_coast.mps_ach)).max()
            self.assertTrue(
                max_trace_miss_coast_m__s > 1.0,
                f"Max trace miss: {max_trace_miss_coast_m__s} m/s")
            self.assertFalse(self.ru_sim_drive_coast.impose_coast[0])
            if DO_PLOTS:
                make_coasting_plot(
                    self.ru_sim_drive_coast.cyc0,
                    self.ru_sim_drive_coast.cyc,
                    use_mph=False,
                    title="Test That We Can Coast",
                    save_file='junk-test-that-we-can-coast-rust.png')

    def test_eco_approach_modeling(self):
        "Test a simplified model of eco-approach"
        if USE_PYTHON:
            self.sim_drive_coast.sim_drive()
            self.assertFalse(self.sim_drive_coast.impose_coast.all(), "Assert we are not always in coast")
            self.assertTrue(self.sim_drive_coast.impose_coast.any(), "Assert we are at least sometimes in coast")
            max_trace_miss_coast_m__s = np.absolute(self.trapz.mps - self.sim_drive_coast.mps_ach).max()
            self.assertTrue(max_trace_miss_coast_m__s > 1.0, "Assert we deviate from the shadow trace")
            self.assertTrue(self.sim_drive_coast.mph_ach.max() > 20.0, "Assert we at least reach 20 mph")
            self.assertAlmostEqual(
                self.trapz.dist_m.sum(),
                self.sim_drive_coast.dist_m.sum(),
                msg="Assert the end distances are equal\n" +
                f"Got {self.trapz.dist_m.sum()} m and {self.sim_drive_coast.dist_m.sum()} m")
        if RUST_AVAILABLE and USE_RUST:
            self.ru_sim_drive_coast.sim_drive()
            self.assertFalse(np.array(self.ru_sim_drive_coast.impose_coast).all(), "Assert we are not always in coast")
            self.assertTrue(np.array(self.ru_sim_drive_coast.impose_coast).any(), "Assert we are at least sometimes in coast")
            max_trace_miss_coast_m__s = np.absolute(
                np.array(self.ru_trapz.mps) - np.array(self.ru_sim_drive_coast.mps_ach)).max()
            self.assertTrue(max_trace_miss_coast_m__s > 1.0, "Assert we deviate from the shadow trace")
            self.assertTrue(np.array(self.ru_sim_drive_coast.mph_ach).max() > 20.0, "Assert we at least reach 20 mph")
            self.assertAlmostEqual(
                np.array(self.ru_trapz.dist_m).sum(),
                np.array(self.ru_sim_drive_coast.dist_m).sum(),
                msg="Assert the end distances are equal\n" +
                f"Got {np.array(self.ru_trapz.dist_m).sum()} m and {np.array(self.ru_sim_drive_coast.dist_m).sum()} m")

    def test_consistency_of_constant_jerk_trajectory(self):
        "Confirm that acceleration, speed, and distances are as expected for constant jerk trajectory"
        if USE_PYTHON:
            n = 10 # ten time-steps
            v0 = 15.0
            vr = 7.5
            d0 = 0.0
            dr = 120.0
            dt = 1.0
            k, a0 = fastsim.cycle.calc_constant_jerk_trajectory(n, d0, v0, dr, vr, dt)
            v = v0
            d = d0
            a = a0
            for n in range(n):
                a_expected = fastsim.cycle.accel_for_constant_jerk(n, a0, k, dt)
                v_expected = fastsim.cycle.speed_for_constant_jerk(n, v0, a0, k, dt)
                d_expected = fastsim.cycle.dist_for_constant_jerk(n, d0, v0, a0, k, dt)
                if n > 0:
                    d += dt * (v + v + a * dt) / 2.0
                    v += a * dt
                # acceleration is the constant acceleration for the NEXT time-step
                a = a0 + n * k * dt
                self.assertAlmostEqual(a, a_expected)
                self.assertAlmostEqual(v, v_expected)
                self.assertAlmostEqual(d, d_expected)
        if RUST_AVAILABLE and USE_RUST:
            n = 10 # ten time-steps
            v0 = 15.0
            vr = 7.5
            d0 = 0.0
            dr = 120.0
            dt = 1.0
            k, a0 = fsr.calc_constant_jerk_trajectory(n, d0, v0, dr, vr, dt)
            v = v0
            d = d0
            a = a0
            for n in range(n):
                a_expected = fsr.accel_for_constant_jerk(n, a0, k, dt)
                v_expected = fsr.speed_for_constant_jerk(n, v0, a0, k, dt)
                d_expected = fsr.dist_for_constant_jerk(n, d0, v0, a0, k, dt)
                if n > 0:
                    d += dt * (v + v + a * dt) / 2.0
                    v += a * dt
                # acceleration is the constant acceleration for the NEXT time-step
                a = a0 + n * k * dt
                self.assertAlmostEqual(a, a_expected)
                self.assertAlmostEqual(v, v_expected)
                self.assertAlmostEqual(d, d_expected)

    def test_that_final_speed_of_cycle_modification_matches_trajectory_calcs(self):
        ""
        if USE_PYTHON:
            trapz = self.trapz.copy()
            idx = 20
            n = 20
            d0 = self.trapz.dist_m[:idx].sum()
            v0 = self.trapz.mps[idx-1]
            dt = self.trapz.dt_s_at_i(idx)
            brake_decel_m__s2 = 2.5
            dts0 = trapz.calc_distance_to_next_stop_from(d0)
            # speed at which friction braking initiates (m/s)
            brake_start_speed_m__s = 7.5
            # distance to brake (m)
            dtb = 0.5 * brake_start_speed_m__s * brake_start_speed_m__s / brake_decel_m__s2
            dtbi0 = dts0 - dtb
            jerk_m__s3, accel_m__s2 = fastsim.cycle.calc_constant_jerk_trajectory(n, d0, v0, d0 + dtbi0, brake_start_speed_m__s, dt)
            final_speed_m__s = self.trapz.modify_by_const_jerk_trajectory(
                idx,
                n,
                jerk_m__s3,
                accel_m__s2)
            self.assertAlmostEqual(final_speed_m__s, brake_start_speed_m__s)
        if RUST_AVAILABLE and USE_RUST:
            trapz = self.ru_trapz.copy()
            idx = 20
            n = 20
            d0 = np.array(self.ru_trapz.dist_m)[:idx].sum()
            v0 = self.ru_trapz.mps[idx-1]
            dt = self.ru_trapz.dt_s_at_i(idx)
            brake_decel_m__s2 = 2.5
            dts0 = trapz.calc_distance_to_next_stop_from(d0)
            # speed at which friction braking initiates (m/s)
            brake_start_speed_m__s = 7.5
            # distance to brake (m)
            dtb = 0.5 * brake_start_speed_m__s * brake_start_speed_m__s / brake_decel_m__s2
            dtbi0 = dts0 - dtb
            jerk_m__s3, accel_m__s2 = fsr.calc_constant_jerk_trajectory(n, d0, v0, d0 + dtbi0, brake_start_speed_m__s, dt)
            final_speed_m__s = self.ru_trapz.modify_by_const_jerk_trajectory(
                idx,
                n,
                jerk_m__s3,
                accel_m__s2)
            self.assertAlmostEqual(final_speed_m__s, brake_start_speed_m__s)

    def test_that_cycle_distance_reported_is_correct(self):
        "Test the reported distances via cycDistMeters"
        if USE_PYTHON:
            # total distance
            d_expected = 900.0
            d_v1 = self.trapz.dist_m.sum()
            d_v2 = fastsim.cycle.trapz_step_distances(self.trapz).sum()
            self.assertAlmostEqual(d_expected, d_v1)
            self.assertAlmostEqual(d_expected, d_v2)
            # distance traveled between 0 s and 10 s
            d_expected = 100.0 # 0.5 * (0s - 10s) * 20m/s = 100m
            d_v1 = self.trapz.dist_m[:11].sum()
            d_v2 = fastsim.cycle.trapz_step_start_distance(self.trapz, 11)
            # TODO: is there a way to get the distance from 0 to 10s using existing cycDistMeters system?
            self.assertNotEqual(d_expected, d_v1)
            self.assertAlmostEqual(d_expected, d_v2)
            # distance traveled between 10 s and 45 s
            d_expected = 700.0 # (45s - 10s) * 20m/s = 700m
            d_v1 = self.trapz.dist_m[11:46].sum()
            d_v2 = fastsim.cycle.trapz_distance_over_range(self.trapz, 11, 46)
            self.assertAlmostEqual(d_expected, d_v1)
            self.assertAlmostEqual(d_expected, d_v2)
            # distance traveled between 45 s and 55 s
            d_expected = 100.0 # 0.5 * (45s - 55s) * 20m/s = 100m
            d_v1 = self.trapz.dist_m[45:56].sum()
            d_v2 = fastsim.cycle.trapz_distance_over_range(self.trapz, 46, 56)
            # TODO: is there a way to get the distance from 45 to 55s using existing cycDistMeters system?
            self.assertNotEqual(d_expected, d_v1)
            self.assertAlmostEqual(d_expected, d_v2)
            # TRIANGLE RAMP SPEED CYCLE
            const_spd_cyc = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 20.0],
                        [0.0, 20.0]
                    ),
                    new_dt=1.0
                )
            )
            expected_dist_m = 200.0 # 0.5 * 20m/s x 20s = 200m
            self.assertAlmostEqual(expected_dist_m, fastsim.cycle.trapz_step_distances(const_spd_cyc).sum())
            self.assertNotEqual(expected_dist_m, const_spd_cyc.dist_m.sum())
        if RUST_AVAILABLE and USE_RUST:
            # total distance
            d_expected = 900.0
            d_v1 = np.array(self.ru_trapz.dist_m).sum()
            d_v2 = fastsim.cycle.trapz_step_distances(self.ru_trapz).sum()
            self.assertAlmostEqual(d_expected, d_v1)
            self.assertAlmostEqual(d_expected, d_v2)
            # distance traveled between 0 s and 10 s
            d_expected = 100.0 # 0.5 * (0s - 10s) * 20m/s = 100m
            d_v1 = np.array(self.ru_trapz.dist_m)[:11].sum()
            d_v2 = fastsim.cycle.trapz_step_start_distance(self.ru_trapz, 11)
            # TODO: is there a way to get the distance from 0 to 10s using existing cycDistMeters system?
            self.assertNotEqual(d_expected, d_v1)
            self.assertAlmostEqual(d_expected, d_v2)
            # distance traveled between 10 s and 45 s
            d_expected = 700.0 # (45s - 10s) * 20m/s = 700m
            d_v1 = np.array(self.ru_trapz.dist_m)[11:46].sum()
            d_v2 = fastsim.cycle.trapz_distance_over_range(self.ru_trapz, 11, 46)
            self.assertAlmostEqual(d_expected, d_v1)
            self.assertAlmostEqual(d_expected, d_v2)
            # distance traveled between 45 s and 55 s
            d_expected = 100.0 # 0.5 * (45s - 55s) * 20m/s = 100m
            d_v1 = np.array(self.trapz.dist_m)[45:56].sum()
            d_v2 = fastsim.cycle.trapz_distance_over_range(self.trapz, 46, 56)
            # TODO: is there a way to get the distance from 45 to 55s using existing cycDistMeters system?
            self.assertNotEqual(d_expected, d_v1)
            self.assertAlmostEqual(d_expected, d_v2)
            # TRIANGLE RAMP SPEED CYCLE
            const_spd_cyc = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 20.0],
                        [0.0, 20.0]
                    ),
                    new_dt=1.0
                )
            ).to_rust()
            expected_dist_m = 200.0 # 0.5 * 20m/s x 20s = 200m
            self.assertAlmostEqual(expected_dist_m, fastsim.cycle.trapz_step_distances(const_spd_cyc).sum())
            self.assertNotEqual(expected_dist_m, np.array(const_spd_cyc.dist_m).sum())

    def test_brake_trajectory(self):
        ""
        if USE_PYTHON:
            trapz = self.trapz.copy()
            brake_accel_m__s2 = -2.0
            idx = 30
            dt = 1.0
            v0 = trapz.mps[idx]
            # distance required to stop (m)
            expected_dts_m = 0.5 * v0 * v0 / abs(brake_accel_m__s2)
            tts_s = -v0 / brake_accel_m__s2
            n = int(np.ceil(tts_s / dt))
            trapz.modify_with_braking_trajectory(brake_accel_m__s2, idx+1)
            self.assertAlmostEqual(v0, trapz.mps[idx])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*dt, trapz.mps[idx+1])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*2*dt, trapz.mps[idx+2])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*3*dt, trapz.mps[idx+3])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*4*dt, trapz.mps[idx+4])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*5*dt, trapz.mps[idx+5])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*6*dt, trapz.mps[idx+6])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*7*dt, trapz.mps[idx+7])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*8*dt, trapz.mps[idx+8])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*9*dt, trapz.mps[idx+9])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*10*dt, trapz.mps[idx+10])
            self.assertEqual(10, n)
            self.assertAlmostEqual(20.0, trapz.mps[idx+11])
            dts_m = fastsim.cycle.trapz_distance_over_range(trapz, idx+1, idx+n+1)
            self.assertAlmostEqual(expected_dts_m, dts_m)
            # Now try with a brake deceleration that doesn't devide evenly by time-steps
            trapz = self.trapz.copy()
            brake_accel_m__s2 = -1.75
            idx = 30
            dt = 1.0
            v0 = trapz.mps[idx]
            # distance required to stop (m)
            expected_dts_m = 0.5 * v0 * v0 / abs(brake_accel_m__s2)
            tts_s = -v0 / brake_accel_m__s2
            n = int(np.round(tts_s / dt))
            trapz.modify_with_braking_trajectory(brake_accel_m__s2, idx+1)
            self.assertAlmostEqual(v0, trapz.mps[idx])
            self.assertEqual(11, n)
            dts_m = fastsim.cycle.trapz_distance_over_range(trapz, idx+1, idx+n+1)
            self.assertAlmostEqual(expected_dts_m, dts_m)
        if RUST_AVAILABLE and USE_RUST:
            trapz = self.ru_trapz.copy()
            brake_accel_m__s2 = -2.0
            idx = 30
            dt = 1.0
            v0 = trapz.mps[idx]
            # distance required to stop (m)
            expected_dts_m = 0.5 * v0 * v0 / abs(brake_accel_m__s2)
            tts_s = -v0 / brake_accel_m__s2
            n = int(np.ceil(tts_s / dt))
            trapz.modify_with_braking_trajectory(brake_accel_m__s2, idx+1)
            self.assertAlmostEqual(v0, trapz.mps[idx])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*dt, trapz.mps[idx+1])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*2*dt, trapz.mps[idx+2])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*3*dt, trapz.mps[idx+3])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*4*dt, trapz.mps[idx+4])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*5*dt, trapz.mps[idx+5])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*6*dt, trapz.mps[idx+6])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*7*dt, trapz.mps[idx+7])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*8*dt, trapz.mps[idx+8])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*9*dt, trapz.mps[idx+9])
            self.assertAlmostEqual(v0 + brake_accel_m__s2*10*dt, trapz.mps[idx+10])
            self.assertEqual(10, n)
            self.assertAlmostEqual(20.0, trapz.mps[idx+11])
            dts_m = fastsim.cycle.trapz_distance_over_range(trapz, idx+1, idx+n+1)
            self.assertAlmostEqual(expected_dts_m, dts_m)
            # Now try with a brake deceleration that doesn't devide evenly by time-steps
            trapz = self.trapz.copy()
            brake_accel_m__s2 = -1.75
            idx = 30
            dt = 1.0
            v0 = trapz.mps[idx]
            # distance required to stop (m)
            expected_dts_m = 0.5 * v0 * v0 / abs(brake_accel_m__s2)
            tts_s = -v0 / brake_accel_m__s2
            n = int(np.round(tts_s / dt))
            trapz.modify_with_braking_trajectory(brake_accel_m__s2, idx+1)
            self.assertAlmostEqual(v0, trapz.mps[idx])
            self.assertEqual(11, n)
            dts_m = fastsim.cycle.trapz_distance_over_range(trapz, idx+1, idx+n+1)
            self.assertAlmostEqual(expected_dts_m, dts_m)
    
    def test_logic_to_enter_eco_approach_automatically(self):
        "Test that we can auto-enter eco-approach"
        if USE_PYTHON:
            trapz = self.trapz.copy()
            veh = fastsim.vehicle.Vehicle.from_vehdb(5)
            sd = fastsim.simdrive.SimDrive(trapz, veh)
            sd.sim_params.coast_allow = True
            sd.sim_params.coast_start_speed_m_per_s = -1
            sd.sim_params.coast_brake_start_speed_m_per_s = 4.0
            sd.sim_drive()
            self.assertTrue(sd.impose_coast.any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="Logic to Enter Eco-Approach Automatically (no 1)",
                    save_file='junk-test-logic-to-enter-eco-approach-automatically-1.png')
            trapz2 = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 200.0, 210.0, 300.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                    ),
                    new_dt=1.0
                )
            )
            veh = fastsim.vehicle.Vehicle.from_vehdb(5)
            sd = fastsim.simdrive.SimDrive(trapz2, veh)
            sd.sim_params.coast_allow = True
            sd.sim_params.coast_start_speed_m_per_s = -1
            sd.sim_params.coast_brake_start_speed_m_per_s = 4.0
            sd.sim_drive()
            self.assertTrue(sd.impose_coast.any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="Logic to Enter Eco-Approach Automatically (no 2)",
                    save_file='junk-test-logic-to-enter-eco-approach-automatically-2.png')
                make_dvdd_plot(
                    sd.cyc,
                    use_mph=False,
                    save_file='junk-test-logic-to-enter-eco-approach-automatically-3-dvdd.png',
                    coast_to_break_speed_m__s=11.0
                )
        if RUST_AVAILABLE and USE_RUST:
            trapz = self.ru_trapz.copy()
            veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
            sd = fastsim.simdrive.RustSimDrive(trapz, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                coast_allow=True,
                coast_start_speed_m_per_s=-1,
                coast_brake_start_speed_m_per_s=4.0
            )
            sd.sim_drive()
            self.assertTrue(np.array(sd.impose_coast).any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="Logic to Enter Eco-Approach Automatically (no 1)",
                    save_file='junk-test-logic-to-enter-eco-approach-automatically-1-rust.png')
            trapz2 = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 200.0, 210.0, 300.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                    ),
                    new_dt=1.0
                )
            ).to_rust()
            veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
            sd = fastsim.simdrive.RustSimDrive(trapz2, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                coast_allow=True,
                coast_start_speed_m_per_s=-1,
                coast_brake_start_speed_m_per_s=4.0
            )
            sd.sim_drive()
            self.assertTrue(np.array(sd.impose_coast).any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="Logic to Enter Eco-Approach Automatically (no 2)",
                    save_file='junk-test-logic-to-enter-eco-approach-automatically-2-rust.png')
                make_dvdd_plot(
                    sd.cyc,
                    use_mph=False,
                    save_file='junk-test-logic-to-enter-eco-approach-automatically-3-dvdd-rust.png',
                    coast_to_break_speed_m__s=11.0
                )

    def test_that_coasting_works_going_uphill(self):
        "Test coasting logic while hill climbing"
        if USE_PYTHON:
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 100.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [0.01, 0.01, 0.01, 0.01, 0.01],
                    ),
                    new_dt=1.0,
                    hold_keys={'grade'},
                )
            )
            veh = fastsim.vehicle.Vehicle.from_vehdb(5)
            sd = fastsim.simdrive.SimDrive(trapz, veh)
            sd.sim_params.coast_allow = True
            sd.sim_params.coast_start_speed_m_per_s = -1
            sd.sim_params.coast_brake_start_speed_m_per_s = 4.0
            sd.sim_drive()
            self.assertTrue(sd.impose_coast.any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                vavgs = np.linspace(5.0, 40.0, endpoint=True)
                grade = 0.01
                def dvdd(vavg, grade):
                    atan_grade = float(np.arctan(grade))
                    g = sd.props.a_grav_mps2
                    M = veh.veh_kg
                    rho_CDFA = sd.props.air_density_kg_per_m3 * veh.frontal_area_m2 * veh.drag_coef
                    return (
                        (g/vavg) * (np.sin(atan_grade) + veh.wheel_rr_coef * np.cos(atan_grade))
                        + (0.5 * rho_CDFA * (1.0/M) * vavg)
                    )
                ks = [dvdd(vavg, grade) for vavg in vavgs]
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="Test That Coasting Works Going Uphill",
                    save_file='junk-test_that_coasting_works_going_uphill-trace.png')
                make_dvdd_plot(
                    sd.cyc,
                    use_mph=False,
                    save_file='junk-test_that_coasting_works_going_uphill-dvdd.png',
                    coast_to_break_speed_m__s=5.0,
                    additional_xs=vavgs,
                    additional_ys=ks
                )
        if RUST_AVAILABLE and USE_RUST:
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 100.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [0.01, 0.01, 0.01, 0.01, 0.01],
                    ),
                    new_dt=1.0,
                    hold_keys={'grade'},
                )
            ).to_rust()
            veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
            sd = fastsim.simdrive.RustSimDrive(trapz, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                coast_allow=True,
                coast_start_speed_m_per_s=-1,
                coast_brake_start_speed_m_per_s=4.0
            )
            sd.sim_drive()
            self.assertTrue(np.array(sd.impose_coast).any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                vavgs = np.linspace(5.0, 40.0, endpoint=True)
                grade = 0.01
                def dvdd(vavg, grade):
                    atan_grade = float(np.arctan(grade))
                    g = sd.props.a_grav_mps2
                    M = veh.veh_kg
                    rho_CDFA = sd.props.air_density_kg_per_m3 * veh.frontal_area_m2 * veh.drag_coef
                    return (
                        (g/vavg) * (np.sin(atan_grade) + veh.wheel_rr_coef * np.cos(atan_grade))
                        + (0.5 * rho_CDFA * (1.0/M) * vavg)
                    )
                ks = [dvdd(vavg, grade) for vavg in vavgs]
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="Test That Coasting Works Going Uphill",
                    save_file='junk-test_that_coasting_works_going_uphill-trace-rust.png')
                make_dvdd_plot(
                    sd.cyc,
                    use_mph=False,
                    save_file='junk-test_that_coasting_works_going_uphill-dvdd-rust.png',
                    coast_to_break_speed_m__s=5.0,
                    additional_xs=vavgs,
                    additional_ys=ks
                )

    def test_that_coasting_logic_works_going_uphill(self):
        "When going uphill, we want to ensure we can still hit our coasting target"
        if USE_PYTHON:
            grade = 0.01
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 100.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [grade]*5,
                    ),
                    new_dt=1.0,
                    hold_keys_next={'grade'},
                )
            )
            veh = fastsim.vehicle.Vehicle.from_vehdb(5)
            sd = fastsim.simdrive.SimDrive(trapz, veh)
            sd.sim_params.coast_allow = True
            sd.sim_params.coast_start_speed_m_per_s = -1
            sd.sim_params.coast_brake_start_speed_m_per_s = 4.0
            sd.sim_params.coast_brake_accel_m_per_s2 = -2.0
            sd.sim_drive()
            self.assertTrue(sd.impose_coast.any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="That Coasting Logic Works Going Uphill (Veh 5)",
                    do_show=False,
                    save_file='junk-test_that_coasting_logic_works_going_uphill-trace-vehicle-5.png')
            # assert we have grade set correctly
            self.assertTrue((sd.cyc0.grade == grade).all())
            self.assertTrue((np.abs(sd.cyc.grade - grade) < 1e-6).all())
            self.assertTrue(
                np.abs(fastsim.cycle.trapz_step_distances(sd.cyc0).sum() - fastsim.cycle.trapz_step_distances(sd.cyc).sum()) < 1.0)
            # test with a different vehicle and grade
            grade = 0.02
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 100.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [grade]*5,
                    ),
                    new_dt=1.0,
                    hold_keys_next={'grade'},
                )
            )
            veh = fastsim.vehicle.Vehicle.from_vehdb(1)
            sd = fastsim.simdrive.SimDrive(trapz, veh)
            sd.sim_params.coast_allow = True
            sd.sim_params.coast_start_speed_m_per_s = -1
            sd.sim_params.coast_brake_start_speed_m_per_s = 7.5
            sd.sim_params.coast_brake_accel_m_per_s2 = -2.5
            sd.sim_drive()
            self.assertTrue(sd.impose_coast.any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="That Coasting Logic Works Going Uphill (Veh 1)",
                    do_show=False,
                    save_file='junk-test_that_coasting_logic_works_going_uphill-trace-vehicle-1.png')
            # TODO: should we use sd.cyc.dist_m or fastsim.cycle.trapz_step_distances().sum() for sd.cyc below?
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(sd.cyc0).sum(),
                sd.cyc.dist_m.sum(),
                places=0)
        if RUST_AVAILABLE and USE_RUST:
            grade = 0.01
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 100.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [grade]*5,
                    ),
                    new_dt=1.0,
                    hold_keys_next={'grade'},
                )
            ).to_rust()
            veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
            sd = fastsim.simdrive.RustSimDrive(trapz, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                coast_allow=True,
                coast_start_speed_m_per_s=-1,
                coast_brake_start_speed_m_per_s=4.0,
                coast_brake_accel_m_per_s2=-2.0,
            )
            assert sd.sim_params.coast_allow
            assert sd.sim_params.coast_start_speed_m_per_s == -1
            assert sd.sim_params.coast_brake_start_speed_m_per_s == 4.0
            assert sd.sim_params.coast_brake_accel_m_per_s2 == -2.0
            sd.sim_drive()
            self.assertTrue(np.array(sd.impose_coast).any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="That Coasting Logic Works Going Uphill (Veh 5)",
                    do_show=False,
                    save_file='junk-test_that_coasting_logic_works_going_uphill-trace-vehicle-5-rust.png')
            # assert we have grade set correctly
            self.assertTrue((np.array(sd.cyc0.grade) == grade).all())
            self.assertTrue((np.abs(np.array(sd.cyc.grade) - grade) < 1e-6).all())
            self.assertTrue(
                np.abs(fastsim.cycle.trapz_step_distances(sd.cyc0).sum() - fastsim.cycle.trapz_step_distances(sd.cyc).sum()) < 1.0)
            # test with a different vehicle and grade
            grade = 0.02
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 100.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [grade]*5,
                    ),
                    new_dt=1.0,
                    hold_keys_next={'grade'},
                )
            ).to_rust()
            veh = fastsim.vehicle.Vehicle.from_vehdb(1).to_rust()
            sd = fastsim.simdrive.RustSimDrive(trapz, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                coast_allow=True,
                coast_start_speed_m_per_s=-1,
                coast_brake_start_speed_m_per_s=7.5,
                coast_brake_accel_m_per_s2=-2.5,
            )
            assert sd.sim_params.coast_allow
            assert sd.sim_params.coast_start_speed_m_per_s == -1
            assert sd.sim_params.coast_brake_start_speed_m_per_s == 7.5
            assert sd.sim_params.coast_brake_accel_m_per_s2 == -2.5
            sd.sim_drive()
            self.assertTrue(np.array(sd.impose_coast).any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="That Coasting Logic Works Going Uphill (Veh 1)",
                    do_show=False,
                    save_file='junk-test_that_coasting_logic_works_going_uphill-trace-vehicle-1-rust.png')
            # TODO: should we use sd.cyc.dist_m or fastsim.cycle.trapz_step_distances().sum() for sd.cyc below?
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(sd.cyc0).sum(),
                np.array(sd.cyc.dist_m).sum(),
                places=0)

    def test_that_coasting_logic_works_going_downhill(self):
        "When going downhill, ensure we can still hit our coasting target"
        if USE_PYTHON:
            grade = -0.0025
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 200.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [grade]*5,
                    ),
                    new_dt=1.0,
                    hold_keys={'grade'},
                )
            )
            veh = fastsim.vehicle.Vehicle.from_vehdb(5)
            sd = fastsim.simdrive.SimDrive(trapz, veh)
            sd.sim_params.coast_allow = True
            sd.sim_params.coast_start_speed_m_per_s = -1
            sd.sim_params.coast_brake_start_speed_m_per_s = 4.0
            sd.sim_params.coast_brake_accel_m_per_s2 = -2.0
            sd.sim_drive()
            self.assertTrue(sd.impose_coast.any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="That Coasting Logic Works Going Downhill (Veh 5)",
                    do_show=False,
                    save_file='junk-test_that_coasting_logic_works_going_downhill-trace-vehicle-5.png',
                    coast_brake_start_speed_m_per_s=sd.sim_params.coast_brake_start_speed_m_per_s)
            # assert we have grade set correctly
            self.assertTrue((sd.cyc0.grade == grade).all())
            self.assertTrue((np.abs(sd.cyc.grade - grade) < 1e-6).all())
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(sd.cyc).sum())
            # test with a different vehicle and grade
            grade = -0.005
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 200.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [grade]*5,
                    ),
                    new_dt=1.0,
                    hold_keys={'grade'},
                )
            )
            veh = fastsim.vehicle.Vehicle.from_vehdb(1)
            sd = fastsim.simdrive.SimDrive(trapz, veh)
            sd.sim_params.coast_allow = True
            sd.sim_params.coast_start_speed_m_per_s = -1
            sd.sim_params.coast_brake_start_speed_m_per_s = 7.5
            sd.sim_params.coast_brake_accel_m_per_s2 = -2.5
            sd.sim_drive()
            self.assertTrue(sd.impose_coast.any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="That Coasting Logic Works Going Downhill (Veh 1)",
                    do_show=False,
                    save_file='junk-test_that_coasting_logic_works_going_downhill-trace-vehicle-1.png',
                    coast_brake_start_speed_m_per_s=sd.sim_params.coast_brake_start_speed_m_per_s)
            # TODO: should we use sd.cyc.dist_m or fastsim.cycle.trapz_step_distances().sum() for sd.cyc below?
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(sd.cyc0).sum(),
                sd.cyc.dist_m.sum())
        if RUST_AVAILABLE and USE_RUST:
            grade = -0.0025
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 200.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [grade]*5,
                    ),
                    new_dt=1.0,
                    hold_keys={'grade'},
                )
            ).to_rust()
            veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
            sd = fastsim.simdrive.RustSimDrive(trapz, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                coast_allow=True,
                coast_start_speed_m_per_s=-1,
                coast_brake_start_speed_m_per_s=4.0,
                coast_brake_accel_m_per_s2=-2.0,
            )
            sd.sim_drive()
            self.assertTrue(np.array(sd.impose_coast).any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="That Coasting Logic Works Going Downhill (Veh 5)",
                    do_show=False,
                    save_file='junk-test_that_coasting_logic_works_going_downhill-trace-vehicle-5-rust.png',
                    coast_brake_start_speed_m_per_s=sd.sim_params.coast_brake_start_speed_m_per_s)
            # assert we have grade set correctly
            self.assertTrue((np.array(sd.cyc0.grade) == grade).all())
            self.assertTrue((np.abs(np.array(sd.cyc.grade) - grade) < 1e-6).all())
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(sd.cyc).sum())
            # test with a different vehicle and grade
            grade = -0.005
            trapz = fastsim.cycle.Cycle.from_dict(
                fastsim.cycle.resample(
                    fastsim.cycle.make_cycle(
                        [0.0, 10.0, 45.0, 55.0, 200.0],
                        [0.0, 20.0, 20.0, 0.0, 0.0],
                        [grade]*5,
                    ),
                    new_dt=1.0,
                    hold_keys={'grade'},
                )
            ).to_rust()
            veh = fastsim.vehicle.Vehicle.from_vehdb(1).to_rust()
            sd = fastsim.simdrive.RustSimDrive(trapz, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                coast_allow=True,
                coast_start_speed_m_per_s=-1,
                coast_brake_start_speed_m_per_s=7.5,
                coast_brake_accel_m_per_s2=-2.5,
            )
            sd.sim_drive()
            self.assertTrue(np.array(sd.impose_coast).any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="That Coasting Logic Works Going Downhill (Veh 1)",
                    do_show=False,
                    save_file='junk-test_that_coasting_logic_works_going_downhill-trace-vehicle-1-rust.png',
                    coast_brake_start_speed_m_per_s=sd.sim_params.coast_brake_start_speed_m_per_s)
            # TODO: should we use sd.cyc.dist_m or fastsim.cycle.trapz_step_distances().sum() for sd.cyc below?
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(sd.cyc0).sum(),
                np.array(sd.cyc.dist_m).sum())

    def test_that_coasting_works_with_multiple_stops_and_grades(self):
        "Ensure coasting hits distance target with multiple stops and both uphill/downhill"
        if USE_PYTHON:
            grade1 = -0.005
            grade2 = 0.005
            c1 = fastsim.cycle.resample(
                fastsim.cycle.make_cycle(
                    [0.0, 10.0, 45.0, 55.0, 200.0],
                    [0.0, 20.0, 20.0, 0.0, 0.0],
                    [grade1]*5,
                ),
                new_dt=1.0,
                hold_keys_next={'grade'},
            )
            c2 = fastsim.cycle.resample(
                fastsim.cycle.make_cycle(
                    [0.0, 10.0, 45.0, 55.0, 200.0],
                    [0.0, 20.0, 20.0, 0.0, 0.0],
                    [grade2]*5,
                ),
                new_dt=1.0,
                hold_keys_next={'grade'},
            )
            veh = fastsim.vehicle.Vehicle.from_vehdb(5)
            cyc = fastsim.cycle.Cycle.from_dict(fastsim.cycle.concat([c1, c2]))
            sd = fastsim.simdrive.SimDrive(cyc, veh)
            sd.sim_params.coast_allow = True
            sd.sim_params.coast_start_speed_m_per_s = -1
            sd.sim_params.coast_brake_start_speed_m_per_s = 4.0
            sd.sim_params.coast_brake_accel_m_per_s2 = -2.0
            sd.sim_drive()
            self.assertTrue(sd.impose_coast.any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="Coasting With Multiple Stops and Grades (Veh 5)",
                    do_show=False,
                    save_file='junk-test_that_coasting_works_with_multiple_stops_and_grades-veh5.png',
                    coast_brake_start_speed_m_per_s=sd.sim_params.coast_brake_start_speed_m_per_s)
            # assert we have grade set correctly
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(sd.cyc).sum())
        if RUST_AVAILABLE and USE_RUST:
            grade1 = -0.005
            grade2 = 0.005
            c1 = fastsim.cycle.resample(
                fastsim.cycle.make_cycle(
                    [0.0, 10.0, 45.0, 55.0, 200.0],
                    [0.0, 20.0, 20.0, 0.0, 0.0],
                    [grade1]*5,
                ),
                new_dt=1.0,
                hold_keys_next={'grade'},
            )
            c2 = fastsim.cycle.resample(
                fastsim.cycle.make_cycle(
                    [0.0, 10.0, 45.0, 55.0, 200.0],
                    [0.0, 20.0, 20.0, 0.0, 0.0],
                    [grade2]*5,
                ),
                new_dt=1.0,
                hold_keys_next={'grade'},
            )
            veh = fastsim.vehicle.Vehicle.from_vehdb(5).to_rust()
            cyc = fastsim.cycle.Cycle.from_dict(fastsim.cycle.concat([c1, c2])).to_rust()
            sd = fastsim.simdrive.RustSimDrive(cyc, veh)
            sd.sim_params = set_nested_values(sd.sim_params,
                coast_allow=True,
                coast_start_speed_m_per_s=-1,
                coast_brake_start_speed_m_per_s=4.0,
                coast_brake_accel_m_per_s2=-2.0,
            )
            assert sd.sim_params.coast_allow
            assert sd.sim_params.coast_start_speed_m_per_s == -1
            assert sd.sim_params.coast_brake_start_speed_m_per_s == 4.0
            assert sd.sim_params.coast_brake_accel_m_per_s2 == -2.0
            sd.sim_drive()
            self.assertTrue(np.array(sd.impose_coast).any(), msg="Coast should initiate automatically")
            if DO_PLOTS:
                make_coasting_plot(
                    sd.cyc0,
                    sd.cyc,
                    use_mph=False,
                    title="Coasting With Multiple Stops and Grades (Veh 5)",
                    do_show=False,
                    save_file='junk-test_that_coasting_works_with_multiple_stops_and_grades-veh5-rust.png',
                    coast_brake_start_speed_m_per_s=sd.sim_params.coast_brake_start_speed_m_per_s)
            # assert we have grade set correctly
            self.assertAlmostEqual(
                fastsim.cycle.trapz_step_distances(sd.cyc0).sum(),
                fastsim.cycle.trapz_step_distances(sd.cyc).sum())

if __name__ == '__main__':
    unittest.main()

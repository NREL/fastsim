use crate::imports::*;

pub struct StepInfo {
    pub dt: si::Time,
    pub speed_prev: si::Velocity,
    pub cyc_speed: si::Velocity,
    pub grade_curr: si::Ratio,
    pub air_density: si::MassDensity,
    pub mass: si::Mass,
    pub drag_coef: si::Ratio,
    pub frontal_area: si::Area,
    pub wheel_inertia: si::MomentOfInertia,
    pub num_wheels: u8,
    pub wheel_radius: si::Length,
    pub wheel_rr_coef: si::Ratio,
    pub pwr_prop_fwd_max: si::Power,
}

impl StepInfo {
    pub fn solve_for_speed(
        &self,
        ach_speed_max_iter: u32,
        ach_speed_tol: si::Ratio,
        ach_speed_solver_gain: f64,
    ) -> si::Velocity {
        let min_speed = 1e-3 * uc::MPS;
        let speed_prev = self.speed_prev.max(min_speed);
        let drag3 = 1.0 / 16.0 * self.air_density * self.drag_coef * self.frontal_area;
        let accel2 = 0.5 * self.mass / self.dt;
        let drag2 = 3.0 / 16.0 * self.air_density * self.drag_coef * self.frontal_area * speed_prev;
        let wheel2 = 0.5 * self.wheel_inertia * self.num_wheels as f64
            / (self.dt * self.wheel_radius.powi(typenum::P2::new()));
        let drag1 = 3.0 / 16.0
            * self.air_density
            * self.drag_coef
            * self.frontal_area
            * speed_prev.powi(typenum::P2::new());
        let roll1 =
            0.5 * self.mass * uc::ACC_GRAV * self.wheel_rr_coef * self.grade_curr.atan().cos();
        let ascent1 = 0.5 * uc::ACC_GRAV * self.grade_curr.atan().sin() * self.mass;
        let accel0 = -0.5 * self.mass * speed_prev.powi(typenum::P2::new()) / self.dt;
        let drag0 = 1.0 / 16.0
            * self.air_density
            * self.drag_coef
            * self.frontal_area
            * speed_prev.powi(typenum::P3::new());
        let roll0 = 0.5
            * self.mass
            * uc::ACC_GRAV
            * self.wheel_rr_coef
            * self.grade_curr.atan().cos()
            * speed_prev;
        let ascent0 = 0.5 * uc::ACC_GRAV * self.grade_curr.atan().sin() * self.mass * speed_prev;
        let wheel0 = -0.5
            * self.wheel_inertia
            * self.num_wheels as f64
            * speed_prev.powi(typenum::P2::new())
            / (self.dt * self.wheel_radius.powi(typenum::P2::new()));

        let t3 = drag3;
        let t2 = accel2 + drag2 + wheel2;
        let t1 = drag1 + roll1 + ascent1;
        let t0 = accel0 + drag0 + roll0 + ascent0 + wheel0;

        // initial guess
        let speed_guess = if self.speed_prev == si::Velocity::ZERO {
            // (1e-3 * uc::MPS).max(0.5 * (self.speed_prev + self.cyc_speed))
            (min_speed).max(self.speed_prev)
        } else {
            (min_speed).max(self.speed_prev)
        };
        // stop criteria
        let max_iter = ach_speed_max_iter;
        let xtol = ach_speed_tol;
        // solver gain
        let g = ach_speed_solver_gain;
        let pwr_err_fn = |speed_guess: si::Velocity| -> si::Power {
            t3 * speed_guess.powi(typenum::P3::new())
                + t2 * speed_guess.powi(typenum::P2::new())
                + t1 * speed_guess
                + t0
                - self.pwr_prop_fwd_max
        };
        let pwr_err_per_speed_guess_fn = |speed_guess: si::Velocity| {
            3.0 * t3 * speed_guess.powi(typenum::P2::new()) + 2.0 * t2 * speed_guess + t1
        };
        let pwr_err = pwr_err_fn(speed_guess);
        if almost_eq_uom(&pwr_err, &(0. * uc::W), Some(1e-6)) {
            return self.cyc_speed;
        }
        let pwr_err_per_speed_guess = pwr_err_per_speed_guess_fn(speed_guess);
        let new_speed_guess = pwr_err - speed_guess * pwr_err_per_speed_guess;
        let mut speed_guesses = vec![speed_guess];
        let mut pwr_errs = vec![pwr_err];
        let mut d_pwr_err_per_d_speed_guesses = vec![pwr_err_per_speed_guess];
        let mut new_speed_guesses = vec![new_speed_guess];
        // speed achieved iteration counter
        let mut spd_ach_iter_counter = 1;
        let mut converged = false;
        let mut speed_ach: si::Velocity = Default::default();
        while spd_ach_iter_counter < max_iter && !converged {
            let speed_guess = *speed_guesses.iter().last().unwrap() * (1.0 - g)
                - g * *new_speed_guesses.iter().last().unwrap()
                    / d_pwr_err_per_d_speed_guesses[speed_guesses.len() - 1];
            let pwr_err = pwr_err_fn(speed_guess);
            let pwr_err_per_speed_guess = pwr_err_per_speed_guess_fn(speed_guess);
            let new_speed_guess = pwr_err - speed_guess * pwr_err_per_speed_guess;
            speed_guesses.push(speed_guess);
            pwr_errs.push(pwr_err);
            d_pwr_err_per_d_speed_guesses.push(pwr_err_per_speed_guess);
            new_speed_guesses.push(new_speed_guess);
            // is the fractional change between previous and current speed guess smaller than `xtol`
            converged = ((*speed_guesses.iter().last().unwrap()
                - speed_guesses[speed_guesses.len() - 2])
                / speed_guesses[speed_guesses.len() - 2])
                .abs()
                < xtol
                && almost_le_uom(&pwr_err, &si::Power::ZERO, None);
            spd_ach_iter_counter += 1;

            // TODO: verify that assuming `speed_guesses.iter().last()` is the correct solution
            speed_ach = speed_guesses.last().unwrap().max(0.0 * uc::MPS);
        }
        speed_ach
    }
}

#[cfg(test)]
mod tests {
    use crate::imports::*;

    use super::StepInfo;

    #[test]
    pub fn test_solver_works_from_0_speed() {
        let step_info = StepInfo {
            speed_prev: si::Velocity::ZERO,
            dt: 1.0 * uc::S,
            cyc_speed: 8.0 * uc::MPS,
            grade_curr: 0.0 * uc::R,
            air_density: 1.2 * uc::KGPM3,
            mass: 1644.2724500334996 * uc::KG,
            drag_coef: 0.393 * uc::R,
            frontal_area: 2.12 * uc::M2,
            wheel_inertia: 0.82 * uc::KGM2,
            num_wheels: 4,
            wheel_radius: 0.326 * uc::M,
            wheel_rr_coef: 0.007 * uc::R,
            pwr_prop_fwd_max: 21750.0 * uc::W,
        };
        let ach_speed_max_iter = 30;
        let ach_speed_tol = 1.0e-6 * uc::R;
        let ach_speed_solver_gain = 0.9;
        let speed_ach =
            step_info.solve_for_speed(ach_speed_max_iter, ach_speed_tol, ach_speed_solver_gain);
        assert!(speed_ach > si::Velocity::ZERO);
    }
}

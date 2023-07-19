use super::super::link::*;
use super::link_point::*;
use super::path_res_coeff::*;
use super::speed_point::*;
use super::train_params::*;
use crate::imports::*;

/// Contains all of the train path parameters in vector form
/// e.g. -  link points, elevations, speed points, and TrainParams
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct PathTpc {
    #[api(skip_set)]
    link_points: Vec<LinkPoint>,
    #[api(skip_set)]
    grades: Vec<PathResCoeff>,
    #[api(skip_set)]
    curves: Vec<PathResCoeff>,
    #[api(skip_set)]
    speed_points: Vec<SpeedPoint>,
    #[api(skip_set)]
    cat_power_limits: Vec<CatPowerLimit>,
    #[api(skip_set)]
    train_params: TrainParams,
    #[api(skip_set)]
    is_finished: bool,
}

impl PathTpc {
    pub fn link_points(&self) -> &[LinkPoint] {
        &self.link_points
    }
    pub fn link_idx_last(&self) -> Option<&LinkIdx> {
        if self.link_points.len() >= 2 {
            Some(&self.link_points[self.link_points.len() - 2].link_idx)
        } else {
            None
        }
    }
    pub fn grades(&self) -> &[PathResCoeff] {
        &self.grades
    }
    pub fn curves(&self) -> &[PathResCoeff] {
        &self.curves
    }
    pub fn speed_points(&self) -> &[SpeedPoint] {
        &self.speed_points
    }
    pub fn cat_power_limits(&self) -> &[CatPowerLimit] {
        &self.cat_power_limits
    }
    pub fn offset_begin(&self) -> si::Length {
        self.link_points.first().unwrap().offset
    }
    pub fn offset_end(&self) -> si::Length {
        self.link_points.last().unwrap().offset
    }
    pub fn is_finished(&self) -> bool {
        self.is_finished
    }

    pub fn new(train_params: TrainParams) -> Self {
        Self {
            link_points: vec![LinkPoint::default()],
            grades: vec![PathResCoeff::default()],
            curves: vec![PathResCoeff::default()],
            speed_points: vec![SpeedPoint {
                offset: si::Length::ZERO,
                speed_limit: train_params.speed_max,
            }],
            cat_power_limits: vec![],
            train_params,
            is_finished: false,
        }
    }

    pub fn extend(&mut self, network: &[Link], link_path: &[LinkIdx]) -> anyhow::Result<()> {
        ensure!(
            !self.link_points.is_empty(),
            "Error: `link_points` is empty."
        );
        ensure!(!self.grades.is_empty(), "Error: `grades` is empty.");
        ensure!(!self.curves.is_empty(), "Error: `curves` is empty.");
        ensure!(
            !self.speed_points.is_empty(),
            "Error: `speed_points` is empty."
        );

        // Set initial elevation when first link is added to path
        if self.grades.len() == 1 && !link_path.is_empty() {
            let elevs_add = &network[link_path.first().unwrap().idx()].elevs;
            if !elevs_add.is_empty() {
                self.grades.last_mut().unwrap().res_net = elevs_add.first().unwrap().elev;
            }
        }

        // Extend link points
        self.link_points.reserve(link_path.len());
        let mut link_point_sum = LinkPoint::default();
        for link_idx in link_path {
            ensure!(link_idx.is_real(), "Error: `link_idx` is not real.");
            let link = &network[link_idx.idx()];
            let offset_base = self.link_points.last().unwrap().offset;

            // Verify that the path to be added is continuous
            if self.link_points.len() >= 2 {
                let link_idx_prev = self.link_points[self.link_points.len() - 2].link_idx;
                ensure!(link_idx_prev.is_real(), "link_idx_prev is not real");
                ensure!(
                    link.idx_prev == link_idx_prev || link.idx_prev_alt == link_idx_prev,
                    "link.idx_curr: {:?} is not contiguous with path!
                    link_points: {:?}",
                    link.idx_curr,
                    self.link_points,
                );
            }

            // Add speeds
            Self::add_speeds(
                &mut self.speed_points,
                &self.train_params,
                &link.speed_sets,
                offset_base,
            );

            // Update link point
            let link_point_add = self.link_points.last_mut().unwrap();
            link_point_add.link_idx = link.idx_curr;
            link_point_add.grade_count = link.elevs.len().max(2) - 1;
            link_point_add.curve_count = link.headings.len().max(2) - 1;
            link_point_add.cat_power_count = link.cat_power_limits.len();

            link_point_sum.add_counts(link_point_add);

            // Add dummy link point
            self.link_points.push(LinkPoint {
                offset: link.length + offset_base,
                ..Default::default()
            });
        }

        // Extend other reservable elements
        self.grades.reserve(link_point_sum.grade_count);
        self.curves.reserve(link_point_sum.curve_count);
        self.cat_power_limits
            .reserve(link_point_sum.cat_power_count);
        for link_idx in link_path {
            let link = &network[link_idx.idx()];
            // Starting/ending offset of current/previous link
            let offset_base = self.grades.last().unwrap().offset;

            // Extend elevs
            if link.elevs.is_empty() {
                self.grades.push(PathResCoeff {
                    offset: offset_base + link.length,
                    res_net: self.grades.last().unwrap().res_net,
                    ..Default::default()
                });
            } else {
                let mut res_net_prev = self.grades.last().unwrap().res_net;
                for (prev, curr) in link.elevs.windows(2).map(|x| (&x[0], &x[1])) {
                    let res_coeff = (curr.elev - prev.elev) / (curr.offset - prev.offset);
                    let res_net = res_net_prev + curr.elev - prev.elev;

                    self.grades.last_mut().unwrap().res_coeff = res_coeff;
                    self.grades.push(PathResCoeff {
                        offset: offset_base + curr.offset,
                        res_net,
                        ..Default::default()
                    });
                    res_net_prev = res_net;
                }
            }

            // Extend curves
            if link.headings.is_empty() {
                self.curves.push(PathResCoeff {
                    offset: offset_base + link.length,
                    res_net: self.curves.last().unwrap().res_net,
                    ..Default::default()
                });
            } else {
                let mut res_net_prev = self.curves.last().unwrap().res_net;
                for (prev, curr) in link.headings.windows(2).map(|x| (&x[0], &x[1])) {
                    let length = curr.offset - prev.offset;
                    let curvature = (-uc::REV / 2.0
                        + (curr.heading - prev.heading + uc::REV / 2.0) % uc::REV)
                        .abs()
                        / length;
                    let one_degree = uc::DEG / (uc::FT * 100.0);

                    let res_coeff = if curvature < one_degree {
                        self.train_params.curve_coeff_0 * curvature
                    } else {
                        self.train_params.curve_coeff_0 * one_degree
                            + self.train_params.curve_coeff_1 * (curvature - one_degree)
                            + self.train_params.curve_coeff_2
                                * (curvature - one_degree)
                                * (curvature - one_degree)
                                / uc::RADPM
                    } / uc::RADPM;
                    let res_net = res_net_prev + res_coeff * length;

                    self.curves.last_mut().unwrap().res_coeff = res_coeff;
                    self.curves.push(PathResCoeff {
                        offset: offset_base + curr.offset,
                        res_net,
                        ..Default::default()
                    });
                    res_net_prev = res_net;
                }
            }

            // Extend cat power limits
            for cpl in &link.cat_power_limits {
                self.cat_power_limits.push(CatPowerLimit {
                    offset_start: offset_base + cpl.offset_start,
                    offset_end: offset_base + cpl.offset_end,
                    power_limit: cpl.power_limit,
                    district_id: cpl.district_id.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn clear(&mut self, offset_back: si::Length) -> anyhow::Result<LinkPoint> {
        ensure!(
            self.link_points.first().unwrap().offset <= offset_back,
            "Error: first link point offset not greater than offset back"
        );
        ensure!(
            offset_back <= self.link_points.last().unwrap().offset,
            "Error: `offset_back` greater than first link point offset"
        );

        // Find last link point before offset back
        let mut link_point_del = LinkPoint::default();
        let mut link_point_idx = 0usize;
        while self.link_points[link_point_idx + 1].offset < offset_back {
            link_point_del.add_counts(&self.link_points[link_point_idx]);
            link_point_idx += 1;
        }
        let link_point_idx = link_point_idx;

        // If at least one link must be deleted
        if link_point_idx > 0 {
            let mut speed_count = 0;
            while self.speed_points[speed_count].offset < self.link_points[link_point_idx].offset {
                speed_count += 1;
            }

            self.link_points.drain(..link_point_idx);
            self.grades.drain(..link_point_del.grade_count);
            self.curves.drain(..link_point_del.curve_count);
            self.cat_power_limits
                .drain(..link_point_del.cat_power_count);

            self.speed_points.drain(..speed_count);
            self.speed_points.first_mut().unwrap().offset =
                self.link_points.first_mut().unwrap().offset;
        }

        //Return the new base link point to shift indices appropriately
        Ok(link_point_del)
    }

    pub fn finish(&mut self) {
        self.grades.push(PathResCoeff {
            offset: uc::M * f64::INFINITY,
            res_net: self.grades.last().unwrap().res_net,
            ..Default::default()
        });
        self.curves.push(PathResCoeff {
            offset: uc::M * f64::INFINITY,
            res_net: self.curves.last().unwrap().res_net,
            ..Default::default()
        });
        self.is_finished = true;
    }

    pub fn recalc_speeds(&mut self, links: &[Link]) {
        self.speed_points.clear();
        self.speed_points.push(SpeedPoint {
            offset: self.link_points.first().unwrap().offset,
            speed_limit: self.train_params.speed_max,
        });
        for link_point in &mut self.link_points {
            Self::add_speeds(
                &mut self.speed_points,
                &self.train_params,
                &links[link_point.link_idx.idx()].speed_sets,
                link_point.offset,
            );
        }
    }

    pub fn reindex(&mut self, link_idxs: &[LinkIdx]) -> anyhow::Result<()> {
        let idx_end = self.link_points.len() - 1;
        for link_point in &mut self.link_points[..idx_end] {
            link_point.link_idx = link_idxs[link_point.link_idx.idx()];
            ensure!(link_point.link_idx.is_real(), "Error: link idx is not real");
        }
        Ok(())
    }

    fn add_speeds(
        speed_points: &mut Vec<SpeedPoint>,
        train_params: &TrainParams,
        speed_sets: &[SpeedSet],
        offset_base: si::Length,
    ) {
        for speed_set in speed_sets {
            if train_params.speed_set_applies(speed_set) {
                speed_points.reserve(speed_set.speed_limits.len() * 2);
                let length_add = if speed_set.is_head_end {
                    si::Length::ZERO
                } else {
                    train_params.length
                };
                for speed_limit in &speed_set.speed_limits {
                    // If the speed limit will actually apply a restriction
                    // Note that this comparison is valid since speed max must be positive
                    if speed_limit.speed < train_params.speed_max {
                        speed_points.insert_speed(&SpeedLimit {
                            offset_start: speed_limit.offset_start + offset_base,
                            offset_end: speed_limit.offset_end + offset_base + length_add,
                            speed: speed_limit.speed,
                        })
                    }
                }
            }
        }
    }
}

impl Default for PathTpc {
    fn default() -> Self {
        Self::new(TrainParams::valid())
    }
}

//TODO: Ask how to handle failed call to extend
impl Valid for PathTpc {
    fn valid() -> Self {
        let mut path_tpc = Self::default();
        path_tpc
            .extend(&Vec::<Link>::valid(), &[LinkIdx::valid()])
            .unwrap_or_default();
        path_tpc.finish();
        path_tpc
    }
}

impl ObjState for PathTpc {
    fn is_fake(&self) -> bool {
        self.link_points.len() <= 1
    }
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if self.is_fake() {
            validate_field_fake(&mut errors, &self.link_points, "Link points");
            validate_field_fake(&mut errors, &self.grades, "Grades");
            validate_field_fake(&mut errors, &self.curves, "Curves");
            validate_field_fake(&mut errors, &self.speed_points, "Speed points");
            validate_field_fake(&mut errors, &self.train_params, "Train params");
        } else {
            validate_field_real(&mut errors, &self.link_points, "Link points");
            validate_field_real(&mut errors, &self.grades, "Grades");
            validate_field_real(&mut errors, &self.curves, "Curves");
            validate_field_real(&mut errors, &self.speed_points, "Speed points");
            validate_field_real(&mut errors, &self.train_params, "Train params");
            early_err!(errors, "Path TPC");

            let mut link_point_sum = LinkPoint::default();
            println!("{:?}, {:?}", self.link_points, self.speed_points);
            for link_point in &self.link_points {
                let mut errors_combo = ValidationErrors::new();
                if link_point.offset != self.grades[link_point_sum.grade_count].offset {
                    errors_combo.push(anyhow!(
                        "Link point offset = {:?} at grade index = {:?} does not equal offset for grade = {:?}!",
                        link_point.offset,
                        link_point_sum.grade_count,
                        self.grades[link_point_sum.grade_count]
                    ));
                }
                if link_point.offset != self.curves[link_point_sum.curve_count].offset {
                    errors_combo.push(anyhow!(
                        "Link point offset = {:?} at curve index = {:?} does not equal offset for curve = {:?}!",
                        link_point.offset,
                        link_point_sum.curve_count,
                        self.curves[link_point_sum.curve_count]
                    ));
                }

                link_point_sum.add_counts(link_point);

                if link_point_sum.grade_count >= self.grades.len() {
                    errors_combo.push(anyhow!(
                        "Grade index = {:?} is too large (total grade count = {:?})!",
                        link_point_sum.grade_count,
                        self.grades.len()
                    ))
                }
                if link_point_sum.curve_count >= self.curves.len() {
                    errors_combo.push(anyhow!(
                        "Curve index = {:?} is too large (total curve count = {:?})!",
                        link_point_sum.curve_count,
                        self.curves.len()
                    ))
                }
                if link_point_sum.cat_power_count > self.cat_power_limits.len() {
                    errors_combo.push(anyhow!(
                        "Cat power index = {:?} is too large (total cat power count = {:?})!",
                        link_point_sum.cat_power_count,
                        self.cat_power_limits.len()
                    ))
                }
                if !errors_combo.is_empty() {
                    errors_combo.add_context(anyhow!(
                        "Link point = {:?} out of range for other path objects!",
                        link_point
                    ));
                    errors.append(&mut errors_combo);
                    continue;
                }

                if !errors_combo.is_empty() {
                    errors_combo.add_context(anyhow!(
                        "Link point = {:?} does not match other path objects!",
                        link_point
                    ));
                    errors.append(&mut errors_combo);
                }
            }
        }

        errors.make_err()
    }
}

#[cfg(test)]
mod test_path_tpc {
    use super::*;
    use crate::testing::*;

    impl Cases for PathTpc {
        // TODO: Fix the validation function to allow this state
        // fn fake_cases() -> Vec<Self> {
        //     vec![PathTpc::default()]
        // }
        fn real_cases() -> Vec<Self> {
            vec![PathTpc::valid()]
        }
    }
    check_cases!(PathTpc);
}

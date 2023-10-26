use crate::imports::*;

#[pyo3_api(
    #[classmethod]
    #[pyo3(name = "from_csv_file")]
    fn from_csv_file_py(_cls: &PyType, pathstr: String) -> anyhow::Result<Self> {
        Self::from_csv_file(&pathstr)
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    #[setter]
    fn set_grade(&mut self, grade: Vec<f64>) {
        self.grade = Some(grade.iter().map(|x| *x * uc::R).collect());
    }
)]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
/// Container
pub struct Cycle {
    #[serde(skip_serializing_if = "Option::is_none")]
    // TODO: either write or automate generation of getter and setter for this
    // TODO: put the above TODO in github issue for all fields with `Option<...>` type
    #[api(skip_get, skip_set)]
    /// inital elevation
    pub init_elev: Option<si::Length>,
    /// simulation time
    #[serde(rename = "time_seconds")]
    pub time: Vec<si::Time>,
    /// prescribed speed
    #[serde(rename = "speed_mps")]
    pub speed: Vec<si::Velocity>,
    // TODO: consider trapezoidal integration scheme
    /// calculated prescribed distance based on RHS integral of time and speed
    pub dist: Vec<si::Length>,
    /// road grade
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub grade: Option<Vec<si::Ratio>>,
    // TODO: consider trapezoidal integration scheme
    /// calculated prescribed elevation based on RHS integral distance and grade
    pub elev: Vec<si::Length>,
    /// road charging/discharing capacity
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub pwr_max_chrg: Option<Vec<si::Power>>,
}

const ELEV_DEF_FT: f64 = 400.;
/// Returns default elevation
pub fn get_elev_def() -> si::Length {
    ELEV_DEF_FT * uc::FT
}

impl SerdeAPI for Cycle {
    /// Sets `self.dist` and `self.elev`
    ///
    /// Assumptions
    /// - if `init_elev.is_none()`, then defaults to
    fn init(&mut self) -> anyhow::Result<()> {
        assert_eq!(self.time.len(), self.speed.len());
        match &mut self.grade {
            Some(grade) => assert_eq!(grade.len(), self.len()),
            None => self.grade = Some(vec![0. * uc::R; self.len()]),
        }
        match &self.pwr_max_chrg {
            Some(pwr_max_chrg) => assert_eq!(pwr_max_chrg.len(), self.len()),
            None => {}
        }

        // calculate distance from RHS integral of speed and time
        self.dist = self
            .time
            .iter()
            .zip(&self.speed)
            .scan(0. * uc::M, |dist, (time, speed)| {
                *dist += *time * *speed;
                Some(*dist)
            })
            .collect();

        // calculate elevation from RHS integral of grade and distance
        match &self.grade {
            Some(grade) => {
                self.init_elev = self.init_elev.or(Some(get_elev_def()));
                self.elev = grade
                    .iter()
                    .zip(&self.dist)
                    .scan(
                        // already guaranteed to be `Some`
                        self.init_elev.unwrap(),
                        |elev, (grade, dist)| {
                            // TODO: Kyle, check this
                            *elev += *dist * *grade;
                            Some(*elev)
                        },
                    )
                    .collect();
            }
            None => {}
        };

        Ok(())
    }

    fn from_file(filename: &str) -> Result<Self, anyhow::Error> {
        // check if the extension is csv, and if it is, then call Self::from_csv_file
        let pathbuf = PathBuf::from(filename);
        let file = File::open(filename)?;
        let extension = pathbuf.extension().unwrap().to_str().unwrap();
        let mut cyc = match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            "csv" => Ok(Self::from_csv_file(filename)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        };

        match &mut cyc {
            Ok(cyc) => cyc.init()?,
            Err(_) => (),
        }

        cyc
    }
}

impl Cycle {
    /// rust-internal time steps at i
    pub fn dt_at_i(&self, i: usize) -> si::Time {
        self.time[i] - self.time[i - 1]
    }

    pub fn dt(&self, i: usize) -> si::Time {
        self.time[i] - self.time[i - 1]
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, element: CycleElement) -> anyhow::Result<()> {
        self.time.push(element.time);
        self.speed.push(element.speed);
        match (&mut self.grade, element.grade) {
            (Some(grade_mut), Some(grade)) => grade_mut.push(grade),
            (None, Some(_)) => {
                bail!("Element and Cycle `grade` fields must both be `Some` or `None`")
            }
            (Some(_), None) => {
                bail!("Element and Cycle `grade` fields must both be `Some` or `None`")
            }
            _ => {}
        }
        match (&mut self.pwr_max_chrg, element.pwr_max_charge) {
            (Some(pwr_max_chrg_mut), Some(pwr_max_chrg)) => pwr_max_chrg_mut.push(pwr_max_chrg),
            (None, Some(_)) => {
                bail!("Element and Cycle `pwr_max_chrg` fields must both be `Some` or `None`")
            }
            (Some(_), None) => {
                bail!("Element and Cycle `pwr_max_chrg` fields must both be `Some` or `None`")
            }
            _ => {}
        }
        self.speed.push(element.speed);
        Ok(())
    }

    pub fn extend(&mut self, vec: Vec<CycleElement>) -> anyhow::Result<()> {
        self.time.extend(vec.iter().map(|x| x.time).clone());
        todo!();
        // self.time.extend(vec.iter().map(|x| x.time).clone());
        // match (&mut self.grade, vec.grade) {
        //     (Some(grade_mut), Some(grade)) => grade_mut.push(grade),
        //     (None, Some(_)) => {
        //         bail!("Element and Cycle `grade` fields must both be `Some` or `None`")
        //     }
        //     (Some(_), None) => {
        //         bail!("Element and Cycle `grade` fields must both be `Some` or `None`")
        //     }
        //     _ => {}
        // }
        // match (&mut self.pwr_max_chrg, vec.pwr_max_charge) {
        //     (Some(pwr_max_chrg_mut), Some(pwr_max_chrg)) => pwr_max_chrg_mut.push(pwr_max_chrg),
        //     (None, Some(_)) => {
        //         bail!("Element and Cycle `pwr_max_chrg` fields must both be `Some` or `None`")
        //     }
        //     (Some(_), None) => {
        //         bail!("Element and Cycle `pwr_max_chrg` fields must both be `Some` or `None`")
        //     }
        //     _ => {}
        // }
        // self.speed.push(vec.speed);
        Ok(())
    }

    pub fn trim(&mut self, start_idx: Option<usize>, end_idx: Option<usize>) -> anyhow::Result<()> {
        let start_idx = start_idx.unwrap_or_default();
        let end_idx = end_idx.unwrap_or(self.len());
        ensure!(end_idx <= self.len(), format_dbg!(end_idx <= self.len()));

        self.time = self.time[start_idx..end_idx].to_vec();
        self.speed = self.speed[start_idx..end_idx].to_vec();
        Ok(())
    }

    /// Load cycle from csv file
    ///
    /// TODO: move this into `from_file`
    pub fn from_csv_file(pathstr: &str) -> Result<Self, anyhow::Error> {
        let pathbuf = PathBuf::from(&pathstr);

        // create empty cycle to be populated
        let mut pt = Self::default();

        let file = File::open(pathbuf)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        for result in rdr.deserialize() {
            let pt_elem: CycleElement = result?;
            pt.push(pt_elem)?;
        }
        pt.init()?;
        if pt.is_empty() {
            bail!("Invalid Cycle file; Cycle is empty")
        } else {
            Ok(pt)
        }
    }
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, SerdeAPI, Clone)]
#[pyo3_api()]
/// Element of `Cycle`.  Used for vec-like operations.
pub struct CycleElement {
    /// simulation time \[s\]
    #[serde(rename = "time_seconds")]
    time: si::Time,
    /// simulation power \[W\]
    #[serde(rename = "speed_mps")]
    speed: si::Velocity,
    /// road grade
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub grade: Option<si::Ratio>,
    /// road charging/discharing capacity
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub pwr_max_charge: Option<si::Power>,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mock_cyc_len_2() -> Cycle {
        let mut cyc = Cycle {
            init_elev: None,
            time: (0..=2).map(|x| (x as f64) * uc::S).collect(),
            speed: (0..=2).map(|x| (x as f64) * uc::MPS).collect(),
            dist: vec![],
            grade: Some((0..=2).map(|x| (x as f64 * uc::R) / 100.).collect()),
            elev: vec![],
            pwr_max_chrg: None,
        };
        cyc.init().unwrap();
        cyc
    }

    #[test]
    fn test_init() {
        let cyc = mock_cyc_len_2();
        assert_eq!(
            cyc.dist,
            [0., 1., 5.] // meters
                .iter()
                .map(|x| *x * uc::M)
                .collect::<Vec<si::Length>>()
        );
        assert_eq!(
            cyc.elev,
            [121.92, 121.93, 122.03] // meters
                .iter()
                .map(|x| *x * uc::M)
                .collect::<Vec<si::Length>>()
        );
    }
}

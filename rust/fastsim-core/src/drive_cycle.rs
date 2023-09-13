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
        self.grade = Some(grade.iter().map(|x| *x * uc::R).collect::<Vec<si::Ratio>>());
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
    /// prescribed distance
    pub dist: Vec<si::Length>,
    /// road grade
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub grade: Option<Vec<si::Ratio>>,
    /// prescribed elevation
    pub elev: Vec<si::Length>,
    /// road charging/discharing capacity
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub pwr_max_charge: Option<Vec<si::Power>>,
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
        match &self.grade {
            Some(grade) => {
                self.elev = {
                    self.init_elev = self.init_elev.or(Some(get_elev_def()));
                    let mut elev = vec![self.init_elev.unwrap(); grade.len()];
                    elev.iter_mut()
                        .zip(grade)
                        .map(|(elev, grade)| (*elev, *grade))
                        .zip(&self.dist)
                        .fold(self.init_elev.unwrap(), |acc, ((_, grade), dist)| {
                            acc + grade * *dist
                        });

                    elev
                }
            }
            None => {}
        };

        Ok(())
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
        self.speed.push(element.speed);
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
            pt.push(pt_elem);
        }
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

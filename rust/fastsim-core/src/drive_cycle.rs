use crate::imports::*;

#[pyo3_api(
    #[new]
    fn __new__(
        time_seconds: Vec<f64>,
        pwr_watts: Vec<f64>,
        engine_on: Vec<Option<bool>>,
    ) -> PyResult<Self> {
        Ok(Self::new(time_seconds, pwr_watts, engine_on))
    }

    #[classmethod]
    #[pyo3(name = "from_csv_file")]
    fn from_csv_file_py(_cls: &PyType, pathstr: String) -> anyhow::Result<Self> {
        Self::from_csv_file(&pathstr)
    }

    fn __len__(&self) -> usize {
        self.len()
    }
)]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI)]
/// Container
pub struct Cycle {
    /// simulation time \[s\]
    #[serde(rename = "time_seconds")]
    pub time: Vec<si::Time>,
    /// simulation power \[W\]
    #[serde(rename = "pwr_watts")]
    pub pwr: Vec<si::Power>,
    /// Whether engine is on
    pub engine_on: Vec<Option<bool>>,
}

impl Cycle {
    pub fn new(time_s: Vec<f64>, pwr_watts: Vec<f64>, engine_on: Vec<Option<bool>>) -> Self {
        Self {
            time: time_s.iter().map(|x| uc::S * (*x)).collect(),
            pwr: pwr_watts.iter().map(|x| uc::W * (*x)).collect(),
            engine_on,
        }
    }

    pub fn empty() -> Self {
        Self {
            time: Vec::new(),
            pwr: Vec::new(),
            engine_on: Vec::new(),
        }
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

    pub fn push(&mut self, pt_element: CycleElement) {
        self.time.push(pt_element.time);
        self.pwr.push(pt_element.pwr);
        self.engine_on.push(pt_element.engine_on);
    }

    pub fn trim(&mut self, start_idx: Option<usize>, end_idx: Option<usize>) -> anyhow::Result<()> {
        let start_idx = start_idx.unwrap_or_default();
        let end_idx = end_idx.unwrap_or(self.len());
        ensure!(end_idx <= self.len(), format_dbg!(end_idx <= self.len()));

        self.time = self.time[start_idx..end_idx].to_vec();
        self.pwr = self.pwr[start_idx..end_idx].to_vec();
        self.engine_on = self.engine_on[start_idx..end_idx].to_vec();
        Ok(())
    }

    /// Load cycle from csv file
    pub fn from_csv_file(pathstr: &str) -> Result<Self, anyhow::Error> {
        let pathbuf = PathBuf::from(&pathstr);

        // create empty cycle to be populated
        let mut pt = Self::empty();

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

impl Default for Cycle {
    fn default() -> Self {
        let pwr_max_watts = 1.5e6;
        let pwr_watts_ramp: Vec<f64> = Vec::linspace(0., pwr_max_watts, 300);
        let mut pwr_watts = pwr_watts_ramp.clone();
        pwr_watts.append(&mut vec![pwr_max_watts; 100]);
        pwr_watts.append(&mut pwr_watts_ramp.iter().rev().copied().collect());
        let time_s: Vec<f64> = (0..pwr_watts.len()).map(|x| x as f64).collect();
        let time_len = time_s.len();
        Self::new(time_s, pwr_watts, vec![Some(true); time_len])
    }
}

/// Element of `Cycle`.  Used for vec-like operations.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub struct CycleElement {
    /// simulation time \[s\]
    #[serde(rename = "time_seconds")]
    time: si::Time,
    /// simulation power \[W\]
    #[serde(rename = "pwr_watts")]
    pwr: si::Power,
    /// Whether engine is on
    engine_on: Option<bool>,
}

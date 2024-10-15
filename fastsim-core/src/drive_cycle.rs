use crate::imports::*;
use fastsim_2::cycle::RustCycle as Cycle2;

#[fastsim_api(
    fn __len__(&self) -> usize {
        self.len()
    }

    #[setter("__grade")]
    fn set_grade_py(&mut self, grade: Vec<f64>) {
        self.grade = grade.iter().map(|x| *x * uc::R).collect();
    }

    #[getter]
    fn get_grade_py(&self) -> Vec<f64> {
        self.grade.iter().map(|x| x.get::<si::ratio>()).collect()
    }
)]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
/// Container
pub struct Cycle {
    /// Name of cycle (can be left empty)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    // TODO: either write or automate generation of getter and setter for this
    // TODO: put the above TODO in github issue for all fields with `Option<...>` type
    #[api(skip_get, skip_set)]
    /// inital elevation
    pub init_elev: Option<si::Length>,
    /// simulation time
    pub time: Vec<si::Time>,
    /// prescribed speed
    #[serde(alias = "speed_mps")]
    pub speed: Vec<si::Velocity>,
    // TODO: consider trapezoidal integration scheme
    /// calculated prescribed distance based on RHS integral of time and speed
    pub dist: Vec<si::Length>,
    /// road grade (expressed as a decimal, not percent)
    #[api(skip_get, skip_set)]
    pub grade: Vec<si::Ratio>,
    // TODO: consider trapezoidal integration scheme
    /// calculated prescribed elevation based on RHS integral distance and grade
    pub elev: Vec<si::Length>,
    /// road charging/discharing capacity
    #[api(skip_get, skip_set)]
    pub pwr_max_chrg: Vec<si::Power>,
    /// grade interpolator
    #[api(skip_get, skip_set)]
    pub grade_interp: Option<Interpolator>,
}

lazy_static! {
    static ref ELEV_DEFAULT: si::Length = 400. * uc::FT;
}

impl Init for Cycle {
    /// Sets `self.dist` and `self.elev`
    ///
    /// Assumptions
    /// - if `init_elev.is_none()`, then defaults to
    fn init(&mut self) -> anyhow::Result<()> {
        ensure!(self.time.len() == self.speed.len());
        ensure!(self.grade.len() == self.len());
        // TODO: figure out if this should be uncommented -- probably need to use a `match`
        // somewhere to fix this ensure!(self.pwr_max_chrg.len() == self.len());

        // calculate distance from RHS integral of speed and time
        self.dist = {
            self.time
                .diff()
                .iter()
                .zip(&self.speed)
                .scan(0. * uc::M, |dist, (dt, speed)| {
                    *dist += *dt * *speed;
                    Some(*dist)
                })
                .collect()
        };

        // calculate elevation from RHS integral of grade and distance
        self.init_elev = self.init_elev.or_else(|| Some(*ELEV_DEFAULT));
        self.elev = self
            .grade
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
        // println!("{:?}", self.dist);
        self.grade_interp = Some(Interpolator::Interp1D(Interp1D::new(
            self.dist.iter().map(|x| x.get::<si::meter>()).collect(),
            self.grade.iter().map(|y| y.get::<si::ratio>()).collect(),
            Strategy::Linear,
            Extrapolate::Error,
        )?));

        Ok(())
    }
}

impl SerdeAPI for Cycle {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &[
        #[cfg(feature = "bincode")]
        "bin",
        #[cfg(feature = "csv")]
        "csv",
        #[cfg(feature = "json")]
        "json",
        #[cfg(feature = "toml")]
        "toml",
        #[cfg(feature = "yaml")]
        "yaml",
    ];
    const ACCEPTED_STR_FORMATS: &'static [&'static str] = &[
        #[cfg(feature = "csv")]
        "csv",
        #[cfg(feature = "json")]
        "json",
        #[cfg(feature = "toml")]
        "toml",
        #[cfg(feature = "yaml")]
        "yaml",
    ];
    #[cfg(feature = "resources")]
    const RESOURCE_PREFIX: &'static str = "cycles";

    /// Write (serialize) an object into anything that implements [`std::io::Write`]
    ///
    /// # Arguments:
    ///
    /// * `wtr` - The writer into which to write object data
    /// * `format` - The target format, any of those listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`)
    ///
    fn to_writer<W: std::io::Write>(&self, mut wtr: W, format: &str) -> anyhow::Result<()> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            #[cfg(feature = "bincode")]
            "bin" | "bincode" => bincode::serialize_into(wtr, self)?,
            #[cfg(feature = "csv")]
            "csv" => {
                let mut wtr = csv::Writer::from_writer(wtr);
                for i in 0..self.len() {
                    wtr.serialize(CycleElement {
                        time: self.time[i],
                        speed: self.speed[i],
                        grade: Some(self.grade[i]),
                        pwr_max_charge: Some(self.pwr_max_chrg[i]),
                    })?;
                }
                wtr.flush()?
            }
            #[cfg(feature = "json")]
            "json" => serde_json::to_writer(wtr, self)?,
            #[cfg(feature = "toml")]
            "toml" => {
                let toml_string = self.to_toml()?;
                wtr.write_all(toml_string.as_bytes())?;
            }
            #[cfg(feature = "yaml")]
            "yaml" | "yml" => serde_yaml::to_writer(wtr, self)?,
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        }
        Ok(())
    }

    /// Deserialize an object from anything that implements [`std::io::Read`]
    ///
    /// # Arguments:
    ///
    /// * `rdr` - The reader from which to read object data
    /// * `format` - The source format, any of those listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`)
    ///
    fn from_reader<R: std::io::Read>(
        rdr: &mut R,
        format: &str,
        skip_init: bool,
    ) -> anyhow::Result<Self> {
        let mut deserialized: Self = match format.trim_start_matches('.').to_lowercase().as_str() {
            #[cfg(feature = "bincode")]
            "bin" | "bincode" => bincode::deserialize_from(rdr)?,
            #[cfg(feature = "csv")]
            "csv" => {
                // Create empty cycle to be populated
                let mut cyc = Self::default();
                let mut rdr = csv::Reader::from_reader(rdr);
                for result in rdr.deserialize() {
                    cyc.push(result?)?;
                }
                cyc
            }
            #[cfg(feature = "json")]
            "json" => serde_json::from_reader(rdr)?,
            #[cfg(feature = "toml")]
            "toml" => {
                let mut buf = String::new();
                rdr.read_to_string(&mut buf)?;
                Self::from_toml(buf, skip_init)?
            }
            #[cfg(feature = "yaml")]
            "yaml" | "yml" => serde_yaml::from_reader(rdr)?,
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        };
        if !skip_init {
            deserialized.init()?;
        }
        Ok(deserialized)
    }

    /// Write (serialize) an object into a string
    ///
    /// # Arguments:
    ///
    /// * `format` - The target format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
    ///
    fn to_str(&self, format: &str) -> anyhow::Result<String> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            #[cfg(feature = "csv")]
            "csv" => self.to_csv(),
            #[cfg(feature = "json")]
            "json" => self.to_json(),
            #[cfg(feature = "toml")]
            "toml" => self.to_toml(),
            #[cfg(feature = "yaml")]
            "yaml" | "yml" => self.to_yaml(),
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_STR_FORMATS
            ),
        }
    }

    /// Read (deserialize) an object from a string
    ///
    /// # Arguments:
    ///
    /// * `contents` - The string containing the object data
    /// * `format` - The source format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
    ///
    fn from_str<S: AsRef<str>>(contents: S, format: &str, skip_init: bool) -> anyhow::Result<Self> {
        Ok(
            match format.trim_start_matches('.').to_lowercase().as_str() {
                #[cfg(feature = "csv")]
                "csv" => Self::from_csv(contents, skip_init)?,
                #[cfg(feature = "json")]
                "json" => Self::from_json(contents, skip_init)?,
                #[cfg(feature = "toml")]
                "toml" => Self::from_toml(contents, skip_init)?,
                #[cfg(feature = "yaml")]
                "yaml" | "yml" => Self::from_yaml(contents, skip_init)?,
                _ => bail!(
                    "Unsupported format {format:?}, must be one of {:?}",
                    Self::ACCEPTED_STR_FORMATS
                ),
            },
        )
    }
}

impl Cycle {
    /// rust-internal time steps at i
    pub fn dt_at_i(&self, i: usize) -> anyhow::Result<si::Time> {
        Ok(*self.time.get(i).with_context(|| format_dbg!())?
            - *self.time.get(i - 1).with_context(|| format_dbg!())?)
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, element: CycleElement) -> anyhow::Result<()> {
        // TODO: maybe automate generation of this function as derive macro
        // TODO: maybe automate `ensure!` that all vec fields are same length before returning result
        // TODO: make sure all fields are being updated as appropriate
        self.time.push(element.time);
        self.speed.push(element.speed);
        match element.grade {
            Some(grade) => self.grade.push(grade),
            _ => self.grade.push(0.0 * uc::R),
        }
        match element.pwr_max_charge {
            Some(pwr_max_chrg) => self.pwr_max_chrg.push(pwr_max_chrg),
            _ => self.pwr_max_chrg.push(0.0 * uc::W),
        }
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
        // Ok(())
    }

    pub fn trim(&mut self, start_idx: Option<usize>, end_idx: Option<usize>) -> anyhow::Result<()> {
        let start_idx = start_idx.unwrap_or_default();
        let end_idx = end_idx.unwrap_or_else(|| self.len());
        ensure!(end_idx <= self.len(), format_dbg!(end_idx <= self.len()));

        self.time = self.time[start_idx..end_idx].to_vec();
        self.speed = self.speed[start_idx..end_idx].to_vec();
        Ok(())
    }

    /// Write (serialize) cycle to a CSV string
    #[cfg(feature = "csv")]
    pub fn to_csv(&self) -> anyhow::Result<String> {
        let mut buf = Vec::with_capacity(self.len());
        self.to_writer(&mut buf, "csv")?;
        Ok(String::from_utf8(buf)?)
    }

    /// Read (deserialize) an object from a CSV string
    ///
    /// # Arguments
    ///
    /// * `json_str` - JSON-formatted string to deserialize from
    ///
    #[cfg(feature = "csv")]
    fn from_csv<S: AsRef<str>>(csv_str: S, skip_init: bool) -> anyhow::Result<Self> {
        let mut csv_de = Self::from_reader(&mut csv_str.as_ref().as_bytes(), "csv", skip_init)?;
        if !skip_init {
            csv_de.init()?;
        }
        Ok(csv_de)
    }

    pub fn to_fastsim2(&self) -> anyhow::Result<Cycle2> {
        let cyc2 = Cycle2 {
            name: self.name.clone(),
            time_s: self.time.iter().map(|t| t.get::<si::second>()).collect(),
            mps: self
                .speed
                .iter()
                .map(|s| s.get::<si::meter_per_second>())
                .collect(),
            grade: self.grade.iter().map(|g| g.get::<si::ratio>()).collect(),
            orphaned: false,
            road_type: vec![0.; self.len()].into(),
        };

        Ok(cyc2)
    }
}

#[fastsim_api]
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
/// Element of `Cycle`.  Used for vec-like operations.
pub struct CycleElement {
    /// simulation time \[s\]
    #[serde(alias = "cycSecs")]
    time: si::Time,
    /// simulation power \[W\]
    #[serde(alias = "speed_mps", alias = "cycMps")]
    speed: si::Velocity,
    // TODO: make `fastsim_api` handle Option or write custom getter/setter
    #[api(skip_get, skip_set)]
    /// road grade
    #[serde(skip_serializing_if = "Option::is_none", alias = "cycGrade")]
    pub grade: Option<si::Ratio>,
    // TODO: make `fastsim_api` handle Option or write custom getter/setter
    #[api(skip_get, skip_set)]
    /// road charging/discharing capacity
    pub pwr_max_charge: Option<si::Power>,
}

impl SerdeAPI for CycleElement {}
impl Init for CycleElement {}

#[cfg(test)]
mod tests {
    use super::*;
    fn mock_cyc_len_2() -> Cycle {
        let mut cyc = Cycle {
            name: String::new(),
            init_elev: None,
            time: (0..=2).map(|x| (x as f64) * uc::S).collect(),
            speed: (0..=2).map(|x| (x as f64) * uc::MPS).collect(),
            dist: vec![],
            grade: (0..=2).map(|x| (x as f64 * uc::R) / 100.).collect(),
            elev: vec![],
            pwr_max_chrg: vec![],
            grade_interp: Default::default(),
        };
        cyc.init().unwrap();
        cyc
    }

    #[test]
    fn test_init() {
        let cyc = mock_cyc_len_2();
        assert_eq!(
            cyc.dist,
            [0., 1., 3.] // meters
                .iter()
                .map(|x| *x * uc::M)
                .collect::<Vec<si::Length>>()
        );
        assert_eq!(
            cyc.elev,
            [121.92, 121.93, 121.99000000000001] // meters
                .iter()
                .map(|x| *x * uc::M)
                .collect::<Vec<si::Length>>()
        );
    }
}

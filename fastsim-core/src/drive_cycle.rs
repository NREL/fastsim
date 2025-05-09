use crate::imports::*;
use crate::prelude::*;
#[cfg(feature = "pyo3")]
use crate::resources;
use fastsim_2::cycle::RustCycle as Cycle2;

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
/// Container
pub struct Cycle {
    /// Name of cycle (can be left empty)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// inital elevation
    pub init_elev: Option<si::Length>,
    /// simulation time
    pub time: Vec<si::Time>,
    /// prescribed speed
    #[serde(alias = "speed_mps")]
    pub speed: Vec<si::Velocity>,
    // TODO: consider trapezoidal integration scheme
    /// calculated prescribed distance based on RHS integral of time and speed
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dist: Vec<si::Length>,
    /// road grade (expressed as a decimal, not percent)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grade: Vec<si::Ratio>,
    // TODO: consider trapezoidal integration scheme
    // TODO: @mokeefe, please check out how elevation is handled
    /// calculated prescribed elevation based on RHS integral distance and grade
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub elev: Vec<si::Length>,
    /// road charging/discharing capacity
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pwr_max_chrg: Vec<si::Power>,
    /// ambient air temperature w.r.t. to time (rather than spatial position)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub temp_amb_air: Vec<si::Temperature>,
    /// solar heat load w.r.t. to time (rather than spatial position)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pwr_solar_load: Vec<si::Power>,
    // TODO: add provision for optional time-varying aux load
    /// grade interpolator
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grade_interp: Option<Interpolator>,
    /// elevation interpolator
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elev_interp: Option<Interpolator>,
}

#[named_struct_pyo3_api]
impl Cycle {
    #[pyo3(name = "list_resources")]
    #[staticmethod]
    /// list available cycle resources
    fn list_resources_py() -> Vec<String> {
        resources::list_resources(Self::RESOURCE_PREFIX)
    }

    #[pyo3(name = "len")]
    fn len_py(&self) -> PyResult<usize> {
        Ok(self.len_checked()?)
    }
}

lazy_static! {
    pub static ref ELEV_DEFAULT: si::Length = 400. * uc::FT;
}

impl Init for Cycle {
    /// Sets `self.dist` and `self.elev`
    /// # Assumptions
    /// - if `init_elev.is_none()`, then defaults to [static@ELEV_DEFAULT]
    fn init(&mut self) -> Result<(), Error> {
        let _ = self
            .len_checked()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;

        if !self.temp_amb_air.is_empty() {
            if self.temp_amb_air.len() != self.time.len() {
                return Err(Error::InitError(format_dbg!()));
            }
        } else {
            self.temp_amb_air = vec![*TE_STD_AIR; self.time.len()];
        }

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

        // populate grade if not provided
        if self.grade.is_empty() {
            self.grade = vec![
                si::Ratio::ZERO;
                self.len_checked()
                    .map_err(|err| Error::InitError(format_dbg!(err)))?
            ]
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
        let g0 = self.grade[0];
        if self.grade.iter().all(|&g| g != g0) {
            self.grade_interp = Some(
                Interpolator::new_1d(
                    self.dist.iter().map(|x| x.get::<si::meter>()).collect(),
                    self.grade.iter().map(|y| y.get::<si::ratio>()).collect(),
                    Strategy::Linear,
                    Extrapolate::Error,
                )
                .map_err(ninterp::error::Error::from)?,
            );

            self.elev_interp = Some(
                Interpolator::new_1d(
                    self.dist.iter().map(|x| x.get::<si::meter>()).collect(),
                    self.elev.iter().map(|y| y.get::<si::meter>()).collect(),
                    Strategy::Linear,
                    Extrapolate::Error,
                )
                .map_err(ninterp::error::Error::from)?,
            );
        } else {
            self.grade_interp = Some(Interpolator::Interp0D(g0.get::<si::ratio>()));
            self.elev_interp = Some(Interpolator::Interp0D(
                self.init_elev.unwrap().get::<si::meter>(),
            ));
        }

        Ok(())
    }
}

impl SerdeAPI for Cycle {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &[
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
    fn to_writer<W: std::io::Write>(&self, mut wtr: W, format: &str) -> Result<(), Error> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            #[cfg(feature = "csv")]
            "csv" => {
                let mut wtr = csv::Writer::from_writer(wtr);
                for i in 0..self
                    .len_checked()
                    .map_err(|err| Error::SerdeError(format_dbg!(err)))?
                {
                    wtr.serialize(CycleElement {
                        // unchecked indexing should be ok because of `self.len()`
                        time: self.time[i],
                        speed: self.speed[i],
                        grade: if !self.grade.is_empty() {
                            Some(self.grade[i])
                        } else {
                            None
                        },
                        pwr_max_charge: if !self.pwr_max_chrg.is_empty() {
                            Some(self.pwr_max_chrg[i])
                        } else {
                            None
                        },
                        temp_amb_air: if !self.temp_amb_air.is_empty() {
                            Some(self.temp_amb_air[i])
                        } else {
                            None
                        },
                        pwr_solar_load: if !self.pwr_solar_load.is_empty() {
                            Some(self.pwr_solar_load[i])
                        } else {
                            None
                        },
                    })
                    .map_err(|err| Error::SerdeError(format_dbg!(err)))?;
                }
                wtr.flush()
                    .map_err(|err| Error::SerdeError(format_dbg!(err)))?
            }
            #[cfg(feature = "json")]
            "json" => serde_json::to_writer(wtr, self)
                .map_err(|err| Error::SerdeError(format_dbg!(err)))?,
            #[cfg(feature = "toml")]
            "toml" => {
                let toml_string = self
                    .to_toml()
                    .map_err(|err| Error::SerdeError(format_dbg!(err)))?;
                wtr.write_all(toml_string.as_bytes())
                    .map_err(|err| Error::SerdeError(format_dbg!(err)))?;
            }
            #[cfg(feature = "yaml")]
            "yaml" | "yml" => serde_yaml::to_writer(wtr, self)
                .map_err(|err| Error::SerdeError(format_dbg!(err)))?,
            _ => Err(Error::SerdeError(format!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS,
            )))?,
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
    ) -> Result<Self, Error> {
        let mut deserialized: Self =
            match format.trim_start_matches('.').to_lowercase().as_str() {
                #[cfg(feature = "csv")]
                "csv" => {
                    // Create empty cycle to be populated
                    let mut cyc = Self::default();
                    let mut rdr = csv::Reader::from_reader(rdr);
                    for result in rdr.deserialize() {
                        cyc.push(result.map_err(|err| Error::SerdeError(format_dbg!(err)))?)
                            .map_err(|err| Error::SerdeError(format!("{err}")))?;
                    }
                    cyc
                }
                #[cfg(feature = "json")]
                "json" => serde_json::from_reader(rdr)
                    .map_err(|err| Error::SerdeError(format!("{err}")))?,
                #[cfg(feature = "toml")]
                "toml" => {
                    let mut buf = String::new();
                    rdr.read_to_string(&mut buf)
                        .map_err(|err| Error::SerdeError(format_dbg!(err)))?;
                    Self::from_toml(buf, skip_init)
                        .map_err(|err| Error::SerdeError(format_dbg!(err)))?
                }
                #[cfg(feature = "yaml")]
                "yaml" | "yml" => serde_yaml::from_reader(rdr)
                    .map_err(|err| Error::SerdeError(format_dbg!(err)))?,
                _ => {
                    return Err(Error::SerdeError(format!(
                        "Unsupported format {format:?}, must be one of {:?}",
                        Self::ACCEPTED_BYTE_FORMATS
                    )))
                }
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

    pub fn len_checked(&self) -> anyhow::Result<usize> {
        ensure!(
            self.time.len() == self.speed.len(),
            format!(
                "{}\n`time` and `speed` fields do not have same `len()`",
                format_dbg!()
            )
        );
        ensure!(
            self.dist.is_empty() || self.time.len() == self.dist.len(),
            format!(
                "{}\n`time` and `dist` fields do not have same `len()`",
                format_dbg!()
            )
        );
        ensure!(
            self.grade.is_empty() || self.time.len() == self.grade.len(),
            format!(
                "{}\n`time` and `grade` fields do not have same `len()`",
                format_dbg!()
            )
        );
        ensure!(
            self.elev.is_empty() || self.grade.len() == self.elev.len(),
            format!(
                "{}\n`grade` and `elev` fields do not have same `len()`",
                format_dbg!()
            )
        );
        ensure!(
            self.pwr_max_chrg.is_empty() || self.time.len() == self.pwr_max_chrg.len(),
            format!(
                "{}\n`time` and `pwr_max_chrg` fields do not have same `len()`",
                format_dbg!()
            )
        );
        ensure!(
            self.temp_amb_air.is_empty() || self.time.len() == self.temp_amb_air.len(),
            format!(
                "{}\n`time` and `temp_amb_air` fields do not have same `len()`",
                format_dbg!()
            )
        );
        Ok(self.time.len())
    }

    pub fn is_empty(&self) -> anyhow::Result<bool> {
        Ok(self.len_checked().with_context(|| format_dbg!())? == 0)
    }

    pub fn push(&mut self, element: CycleElement) -> anyhow::Result<()> {
        // TODO: maybe automate generation of this function as derive macro
        // TODO: maybe automate `ensure!` that all vec fields are same length before returning result
        // TODO: make sure all fields are being updated as appropriate
        self.time.push(element.time);
        self.speed.push(element.speed);
        match element.grade {
            Some(grade) => self.grade.push(grade),
            None => self.grade.push(si::Ratio::ZERO),
        }
        match element.pwr_max_charge {
            Some(pwr_max_chrg) => self.pwr_max_chrg.push(pwr_max_chrg),
            None => self.pwr_max_chrg.push(si::Power::ZERO),
        }
        match element.temp_amb_air {
            Some(temp_amb_air) => self.temp_amb_air.push(temp_amb_air),
            None => self.temp_amb_air.push(*TE_STD_AIR),
        }
        match element.pwr_solar_load {
            Some(pwr_solar_load) => self.pwr_solar_load.push(pwr_solar_load),
            None => self.pwr_solar_load.push(si::Power::ZERO),
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
        let len = self.len_checked().with_context(|| format_dbg!())?;
        let end_idx = end_idx.unwrap_or(len);
        ensure!(end_idx <= len, format_dbg!(end_idx <= len));

        self.time = self.time[start_idx..end_idx].to_vec();
        self.speed = self.speed[start_idx..end_idx].to_vec();
        Ok(())
    }

    /// Write (serialize) cycle to a CSV string
    #[cfg(feature = "csv")]
    pub fn to_csv(&self) -> anyhow::Result<String> {
        let mut buf = Vec::with_capacity(self.len_checked().with_context(|| format_dbg!())?);
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
            road_type: vec![0.; self.len_checked().with_context(|| format_dbg!())?].into(),
        };

        Ok(cyc2)
    }
}

impl TryFrom<CycleBuilder> for Cycle {
    type Error = anyhow::Error;
    fn try_from(value: CycleBuilder) -> anyhow::Result<Self, Self::Error> {
        let mut cyc = Self {
            name: value.name,
            init_elev: None,
            time: value.time,
            speed: value.speed,
            dist: Default::default(),
            grade: Default::default(),
            elev: Default::default(),
            pwr_max_chrg: Default::default(),
            temp_amb_air: Default::default(),
            pwr_solar_load: Default::default(),
            grade_interp: None,
            elev_interp: Default::default(),
        };
        cyc.init()?;
        Ok(cyc)
    }
}

#[serde_api]
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[non_exhaustive]
/// Simple cycle to be converted into [Cycle] with appropriate defaults
pub struct CycleBuilder {
    /// Name of cycle (can be left empty)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// simulation time
    pub time: Vec<si::Time>,
    /// prescribed speed
    pub speed: Vec<si::Velocity>,
}

#[serde_api]
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
/// Element of `Cycle`.  Used for vec-like operations.
pub struct CycleElement {
    /// simulation time \[s\]
    #[serde(alias = "cycSecs")]
    pub time: si::Time,
    /// simulation power \[W\]
    #[serde(alias = "speed_mps", alias = "cycMps")]
    pub speed: si::Velocity,
    // `dist` is not included here because it is derived in `Init::init`
    /// road grade
    #[serde(alias = "cycGrade")]
    pub grade: Option<si::Ratio>,
    // `elev` is not included here because it is derived in `Init::init`
    /// road charging/discharing capacity
    pub pwr_max_charge: Option<si::Power>,
    // TODO: make sure all fields in cycle are represented here, as appropriate
    /// ambient air temperature w.r.t. to time (rather than spatial position)
    pub temp_amb_air: Option<si::Temperature>,
    /// solar heat load w.r.t. to time (rather than spatial position)
    pub pwr_solar_load: Option<si::Power>,
}

impl SerdeAPI for CycleElement {}
impl Init for CycleElement {}

#[named_struct_pyo3_api]
impl CycleElement {}

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
            elev_interp: Default::default(),
            temp_amb_air: Default::default(),
            pwr_solar_load: Default::default(),
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

lazy_static! {
    pub static ref CYC_ACCEL: Cycle = Cycle::try_from(CycleBuilder {
        name: String::from("accel test"),
        time: (0..300)
            .map(|t| (t as f64) * uc::S)
            .collect::<Vec<si::Time>>(),
        speed: vec![90.0 * uc::MPH; 300],
    })
    .unwrap();
}

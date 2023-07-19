use super::cat_power::*;
use super::elev::*;
use super::heading::*;
use super::link_idx::*;
use super::speed::*;

use crate::imports::*;

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, SerdeAPI)]
/// An arbitrary unit of single track that does not include turnouts
#[altrios_api]
pub struct Link {
    pub elevs: Vec<Elev>,
    #[serde(default)]
    pub headings: Vec<Heading>,
    pub speed_sets: Vec<SpeedSet>,
    #[serde(default)]
    pub cat_power_limits: Vec<CatPowerLimit>,
    pub length: si::Length,

    pub idx_next: LinkIdx,
    pub idx_next_alt: LinkIdx,
    pub idx_prev: LinkIdx,
    pub idx_prev_alt: LinkIdx,
    pub idx_curr: LinkIdx,
    pub idx_flip: LinkIdx,
    // TODO:  Geordie will add this
    #[serde(default)]
    pub link_idxs_lockout: Vec<LinkIdx>,
}
impl Link {
    fn is_linked_prev(&self, idx: LinkIdx) -> bool {
        self.idx_curr.is_fake() || self.idx_prev == idx || self.idx_prev_alt == idx
    }
    fn is_linked_next(&self, idx: LinkIdx) -> bool {
        self.idx_curr.is_fake() || self.idx_next == idx || self.idx_next_alt == idx
    }
}

impl Valid for Link {
    fn valid() -> Self {
        Self {
            elevs: Vec::<Elev>::valid(),
            headings: Vec::<Heading>::valid(),
            speed_sets: Vec::<SpeedSet>::valid(),
            length: uc::M * 10000.0,
            idx_curr: LinkIdx::valid(),
            ..Self::default()
        }
    }
}

impl ObjState for Link {
    fn is_fake(&self) -> bool {
        self.idx_curr.is_fake()
    }

    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if self.is_fake() {
            validate_field_fake(&mut errors, &self.idx_next, "Link index next");
            validate_field_fake(&mut errors, &self.idx_next_alt, "Link index next alt");
            validate_field_fake(&mut errors, &self.idx_prev, "Link index prev");
            validate_field_fake(&mut errors, &self.idx_prev_alt, "Link index prev alt");
            validate_field_fake(&mut errors, &self.idx_curr, "Link index curr");
            validate_field_fake(&mut errors, &self.idx_flip, "Link index flip");
            si_chk_num_eqz(&mut errors, &self.length, "Link length");
            validate_field_fake(&mut errors, &self.elevs, "Elevations");
            validate_field_fake(&mut errors, &self.headings, "Headings");
            validate_field_fake(&mut errors, &self.speed_sets, "Speed sets");
            // validate cat_power_limits
            if !self.cat_power_limits.is_empty() {
                errors.push(anyhow!(
                    "Catenary power limits = {:?} must be empty!",
                    self.cat_power_limits
                ));
            }
        } else {
            si_chk_num_gtz(&mut errors, &self.length, "Link length");
            validate_field_real(&mut errors, &self.elevs, "Elevations");
            if !self.headings.is_empty() {
                validate_field_real(&mut errors, &self.headings, "Headings");
            }
            validate_field_real(&mut errors, &self.speed_sets, "Speed sets");
            validate_field_real(&mut errors, &self.cat_power_limits, "Catenary power limits");

            early_err!(errors, "Link");

            if self.idx_flip.is_real() {
                for (var, name) in [
                    (self.idx_curr, "curr"),
                    (self.idx_next, "next"),
                    (self.idx_next_alt, "next alt"),
                    (self.idx_prev, "prev"),
                    (self.idx_prev_alt, "prev alt"),
                ] {
                    if var == self.idx_flip {
                        errors.push(anyhow!(
                            "Link index flip = {:?} and link index {} = {:?} must be different!",
                            self.idx_flip,
                            name,
                            var
                        ));
                    }
                }
            }
            if self.idx_next_alt.is_real() && self.idx_next.is_fake() {
                errors.push(anyhow!(
                    "Link index next = {:?} must be real when link index next alt = {:?} is real!",
                    self.idx_next,
                    self.idx_next_alt
                ));
            }
            if self.idx_prev_alt.is_real() && self.idx_prev.is_fake() {
                errors.push(anyhow!(
                    "Link index prev = {:?} must be real when link index prev alt = {:?} is real!",
                    self.idx_prev,
                    self.idx_prev_alt
                ));
            }

            if self.elevs.first().unwrap().offset != si::Length::ZERO {
                errors.push(anyhow!(
                    "First elevation offset = {:?} is invalid, must equal zero!",
                    self.elevs.first().unwrap().offset
                ));
            }
            if self.elevs.last().unwrap().offset != self.length {
                errors.push(anyhow!(
                    "Last elevation offset = {:?} is invalid, must equal length = {:?}!",
                    self.elevs.last().unwrap().offset,
                    self.length
                ));
            }
            if !self.headings.is_empty() {
                if self.headings.first().unwrap().offset != si::Length::ZERO {
                    errors.push(anyhow!(
                        "First heading offset = {:?} is invalid, must equal zero!",
                        self.headings.first().unwrap().offset
                    ));
                }
                if self.headings.last().unwrap().offset != self.length {
                    errors.push(anyhow!(
                        "Last heading offset = {:?} is invalid, must equal length = {:?}!",
                        self.headings.last().unwrap().offset,
                        self.length
                    ));
                }
            }
            if !self.cat_power_limits.is_empty() {
                if self.cat_power_limits.first().unwrap().offset_start < si::Length::ZERO {
                    errors.push(anyhow!(
                        "First cat power limit offset start = {:?} is invalid, must be greater than or equal to zero!",
                        self.cat_power_limits.first().unwrap().offset_start
                    ));
                }
                if self.cat_power_limits.last().unwrap().offset_end > self.length {
                    errors.push(anyhow!(
                        "Last cat power limit offset end = {:?} is invalid, must be less than or equal to length = {:?}!",
                        self.cat_power_limits.last().unwrap().offset_end,
                        self.length
                    ));
                }
            }
        }
        errors.make_err()
    }
}

#[cfg_attr(feature = "pyo3", pyfunction(name = "import_network"))]
pub fn import_network_py(filename: String) -> anyhow::Result<Vec<Link>> {
    let network = Vec::<Link>::from_file(&filename)?;
    network.validate()?;
    Ok(network)
}

impl Valid for Vec<Link> {
    fn valid() -> Self {
        vec![Link::default(), Link::valid()]
    }
}

impl ObjState for [Link] {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if self.len() < 2 {
            errors.push(anyhow!(
                "There must be at least two links (one physical and one dummy)!"
            ));
            return Err(errors);
        }
        validate_slice_fake(&mut errors, &self[..1], "Link");
        validate_slice_real_shift(&mut errors, &self[1..], "Link", 1);
        early_err!(errors, "Links");

        for (idx, link) in self.iter().enumerate().skip(1) {
            // Validate flip and curr
            if link.idx_curr.idx() != idx {
                errors.push(anyhow!(
                    "Link idx {} is not equal to index in vector {}!",
                    link.idx_curr,
                    idx
                ))
            }
            if link.idx_flip == link.idx_curr {
                errors.push(anyhow!(
                    "Normal {} and flipped {} links must be different!",
                    link.idx_curr,
                    link.idx_flip
                ));
            }
            if link.idx_flip.is_real() && self[link.idx_flip.idx()].idx_flip != link.idx_curr {
                errors.push(anyhow!(
                    "Flipped link {} does not properly reference current link {}!",
                    link.idx_flip,
                    link.idx_curr
                ));
            }

            // Validate next
            if link.idx_next.is_real() {
                for (link_next, name) in [
                    (&self[link.idx_next.idx()], "next link"),
                    (&self[link.idx_next_alt.idx()], "next link alt"),
                ] {
                    if !link_next.is_linked_prev(link.idx_curr) {
                        errors.push(anyhow!(
                            "Current link {} with {} {} prev idx {} and prev idx alt {} do not point back!",
                            link.idx_curr,
                            name,
                            link_next.idx_curr,
                            link_next.idx_prev,
                            link_next.idx_prev_alt,
                        ));
                    }
                    if link.idx_next_alt.is_real() && link_next.idx_prev_alt.is_real() {
                        errors.push(anyhow!(
                            "Current link {} and {} {} have coincident switch points!",
                            link.idx_curr,
                            name,
                            link_next.idx_curr,
                        ));
                    }
                }
            } else if link.idx_next_alt.is_real() {
                errors.push(anyhow!(
                    "Next idx alt {} is real when next idx {} is fake!",
                    link.idx_next_alt,
                    link.idx_next,
                ));
            }

            // Validate prev
            if link.idx_prev.is_real() {
                for (link_prev, name) in [
                    (&self[link.idx_prev.idx()], "prev link"),
                    (&self[link.idx_prev_alt.idx()], "prev link alt"),
                ] {
                    if !link_prev.is_linked_next(link.idx_curr) {
                        errors.push(anyhow!(
                            "Current link {} with {} {} next idx {} and next idx alt {} do not point back!",
                            link.idx_curr,
                            name,
                            link_prev.idx_curr,
                            link_prev.idx_next,
                            link_prev.idx_next_alt,
                        ));
                    }
                    if link.idx_prev_alt.is_real() && link_prev.idx_next_alt.is_real() {
                        errors.push(anyhow!(
                            "Current link {} and {} {} have coincident switch points!",
                            link.idx_curr,
                            name,
                            link_prev.idx_curr,
                        ));
                    }
                }
            } else if link.idx_prev_alt.is_real() {
                errors.push(anyhow!(
                    "Prev idx alt {} is real when prev idx {} is fake!",
                    link.idx_prev_alt,
                    link.idx_prev
                ));
            }
        }
        errors.make_err()
    }
}

#[cfg(test)]
mod test_link {
    use super::*;
    use crate::testing::*;

    impl Cases for Link {
        fn real_cases() -> Vec<Self> {
            vec![
                Self::valid(),
                Self {
                    idx_flip: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_next: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_prev: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_next: LinkIdx::new(2),
                    idx_next_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_prev: LinkIdx::new(2),
                    idx_prev_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
            ]
        }
        fn fake_cases() -> Vec<Self> {
            vec![Self::default()]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![
                Self {
                    elevs: Vec::<Elev>::invalid_cases().first().unwrap().to_vec(),
                    ..Self::valid()
                },
                Self {
                    elevs: Vec::<Elev>::new(),
                    ..Self::valid()
                },
                Self {
                    length: si::Length::ZERO,
                    ..Self::valid()
                },
                Self {
                    length: -uc::M,
                    ..Self::valid()
                },
                Self {
                    length: uc::M * f64::NAN,
                    ..Self::valid()
                },
                Self {
                    idx_curr: LinkIdx::default(),
                    ..Self::valid()
                },
                Self {
                    idx_flip: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_next_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_prev_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_next: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_next: LinkIdx::new(3),
                    idx_next_alt: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_prev: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_prev: LinkIdx::new(3),
                    idx_prev_alt: LinkIdx::new(2),
                    ..Self::valid()
                },
            ]
        }
    }

    check_cases!(Link);

    #[test]
    fn check_elevs_start() {
        for mut link in Link::real_cases() {
            link.elevs.first_mut().unwrap().offset -= uc::M;
            link.validate().unwrap_err();
        }
    }

    #[test]
    fn check_elevs_end() {
        for mut link in Link::real_cases() {
            link.elevs.last_mut().unwrap().offset += uc::M;
            link.validate().unwrap_err();
        }
    }
}

#[cfg(test)]
mod test_links {
    use super::*;
    use crate::testing::*;

    impl Cases for Vec<Link> {
        fn real_cases() -> Vec<Self> {
            vec![Self::valid()]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![vec![], Self::valid()[..1].to_vec()]
        }
    }
    //check_cases!(Vec<Link>);
    //check_vec_elems!(Link);

    #[test]
    fn test_to_and_from_file_for_links() {
        let links = Vec::<Link>::valid();
        links.to_file("links_test2.yaml").unwrap();
        assert_eq!(Vec::<Link>::from_file("links_test2.yaml").unwrap(), links);
        std::fs::remove_file("links_test2.yaml").unwrap();
    }
}

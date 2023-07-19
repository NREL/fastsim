use crate::imports::*;
use serde::{de::Visitor, Deserializer, Serializer};
use std::fmt;

#[altrios_api(
    #[new]
    fn __new__(
        idx: u32
    ) -> Self {
        Self::new(
            idx
        )
    }
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, SerdeAPI)]
pub struct LinkIdx {
    #[api(skip_set)]
    idx: u32,
}
pub const LINK_IDX_NA: LinkIdx = LinkIdx { idx: 0 };

impl LinkIdx {
    pub fn new(idx: u32) -> Self {
        Self { idx }
    }
    pub fn idx(&self) -> usize {
        self.idx.idx()
    }
}

impl std::hash::Hash for LinkIdx {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        hasher.write_u32(self.idx);
    }
}
impl nohash_hasher::IsEnabled for LinkIdx {}

impl fmt::Display for LinkIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.idx)
    }
}

impl Serialize for LinkIdx {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.idx)
    }
}

impl<'de> Deserialize<'de> for LinkIdx {
    fn deserialize<D>(deserializer: D) -> Result<LinkIdx, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LinkIdxVisitor;
        impl<'de> Visitor<'de> for LinkIdxVisitor {
            type Value = LinkIdx;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("integer")
            }

            fn visit_u32<E>(self, v: u32) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(LinkIdx::new(v))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if v >= u64::from(u32::MIN) && v <= u64::from(u32::MAX) {
                    Ok(LinkIdx::new(v as u32))
                } else {
                    Err(E::custom(format!("u32 out of range: {v}")))
                }
            }
        }

        deserializer.deserialize_u32(LinkIdxVisitor)
    }
}

impl Valid for LinkIdx {
    fn valid() -> Self {
        Self { idx: 1 }
    }
}

impl ObjState for LinkIdx {
    fn is_fake(&self) -> bool {
        self.idx == 0
    }
}

#[cfg(test)]
mod test_link_idx {
    use super::*;
    use crate::testing::*;

    impl Cases for LinkIdx {
        fn real_cases() -> Vec<Self> {
            vec![Self::valid()]
        }
        fn fake_cases() -> Vec<Self> {
            vec![Self::new(0)]
        }
    }
    check_cases!(LinkIdx);

    #[test]
    fn check_new() {
        assert!(LinkIdx::new(0) == LinkIdx { idx: 0 });
        assert!(LinkIdx::new(1) == LinkIdx { idx: 1 });
        assert!(LinkIdx::new(4294967295) == LinkIdx { idx: 4294967295 });
    }
}

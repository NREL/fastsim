use crate::imports::*;

///Standardizes conversion from smaller than usize types for indexing.
pub trait Idx {
    fn idx(self) -> usize;
}

#[duplicate_item(Self; [u8]; [u16])]
impl Idx for Self {
    fn idx(self) -> usize {
        self.into()
    }
}

impl Idx for u32 {
    fn idx(self) -> usize {
        self.try_into().unwrap()
    }
}

impl Idx for Option<NonZeroU16> {
    fn idx(self) -> usize {
        self.map(u16::from).unwrap_or(0) as usize
    }
}

/// Trait implemented for indexing types, specifically [usize], to assist in
/// converting them into an [Option]\<NonZeroUxxxx\>.
///
/// This is necessary because both the common type conversion trait ([From])
/// and the types involved (e.g. [Option]<[NonZeroU16]> ) are from the
/// standard library.  Rust does not allow for the combination of traits and
/// types if both are from external libraries.
///
/// This trait is default implemented for [usize] to convert into any type that
/// already implements [TryFrom]<[NonZeroUsize]>, which includes most of the
/// NonZeroUxxxx types.
///
/// This approach will error on a loss of precision.  So, if the [usize] value
/// does not fit into the `NonZero` type, then an [Error] variant is returned.
///
/// If the value is `0`, then an [Ok]\([None]) variant is returned.
///
/// Note that the base `NonZero` types already have a similar feature if
/// converting from the same basic type (e.g. [u16] to [Option]<[NonZeroU16]>),
/// where the type is guaranteed to fit, but just might be `0`.  If that is the
/// use case, then just use the `new()` method, which returns the [None] variant
/// if the value is `0`.
///
/// The intended usage is as follows:
/// ```
/// # use std::num::NonZeroU8;
/// # use altrios_core::traits::TryFromIdx;
///
/// let non_zero : usize = 42;
/// let good_val : Option<NonZeroU8> = non_zero.try_from_idx().unwrap();
/// assert!(good_val == NonZeroU8::new(42));
///
/// let zero : usize = 0;
/// let none_val : Option<NonZeroU8> = zero.try_from_idx().unwrap();
/// assert!(none_val == None);
///
/// let too_big : usize = 256;
/// let bad_val : Result<Option<NonZeroU8>, _> = too_big.try_from_idx();
/// assert!(bad_val.is_err());
/// ```
pub trait TryFromIdx<T> {
    type Error;

    fn try_from_idx(&self) -> Result<Option<T>, Self::Error>;
}

impl<T> TryFromIdx<T> for usize
where
    T: TryFrom<NonZeroUsize>,
{
    type Error = <T as TryFrom<NonZeroUsize>>::Error;

    fn try_from_idx(&self) -> Result<Option<T>, Self::Error> {
        NonZeroUsize::new(*self).map_or(
            // If value is a 0-valued usize, then we immediately return an
            // Ok(None).
            Ok(None),
            // Otherwise we attempt to convert it from a NonZeroUsize into a
            // different type with potentially smaller accuracy.
            |val| {
                T::try_from(val)
                    // We wrap a valid result in Some
                    .map(Some)
            },
        )
    }
}

pub trait Linspace {
    fn linspace(start: f64, stop: f64, n_elements: usize) -> Vec<f64> {
        let n_steps = n_elements - 1;
        let step_size = (stop - start) / n_steps as f64;
        let v_norm: Vec<f64> = (0..=n_steps)
            .collect::<Vec<usize>>()
            .iter()
            .map(|x| *x as f64)
            .collect();
        let v = v_norm.iter().map(|x| (x * step_size) + start).collect();
        v
    }
}

impl Linspace for Vec<f64> {}

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    /// runs any initialization steps that may be needed.
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    /// Save current data structure to file. Method adaptively calls serialization methods
    /// dependent on the suffix of the file given as str.
    ///
    /// # Argument:
    ///
    /// * `filename`: a `str` storing the targeted file name. Currently `.json` and `.yaml` suffixes are
    /// supported
    ///
    /// # Returns:
    ///
    /// A Rust Result
    fn to_file(&self, filename: &str) -> Result<(), anyhow::Error> {
        let file = PathBuf::from(filename);
        match file.extension().unwrap().to_str().unwrap() {
            "json" => serde_json::to_writer(&File::create(file)?, self)?,
            "yaml" => serde_yaml::to_writer(&File::create(file)?, self)?,
            _ => serde_json::to_writer(&File::create(file)?, self)?,
        };
        Ok(())
    }

    /// Read from file and return instantiated struct. Method adaptively calls deserialization
    /// methods dependent on the suffix of the file name given as str.
    /// Function returns a dynamic Error Result if it fails.
    ///
    /// # Argument:
    ///
    /// * `filename`: a `str` storing the targeted file name. Currently `.json` and `.yaml` suffixes are
    /// supported
    ///
    /// # Returns:
    ///
    /// A Rust Result wrapping data structure if method is called successfully; otherwise a dynamic
    /// Error.
    fn from_file(filename: &str) -> Result<Self, anyhow::Error>
    where
        Self: std::marker::Sized,
        for<'de> Self: Deserialize<'de>,
    {
        let extension = Path::new(filename)
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or("");

        let file = File::open(filename)?;
        // deserialized file
        let mut file_de: Self = match extension {
            "yaml" => serde_yaml::from_reader(file)?,
            "json" => serde_json::from_reader(file)?,
            _ => bail!("Unsupported file extension {}", extension),
        };
        file_de.init()?;
        Ok(file_de)
    }

    /// json serialization method.
    fn to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    /// json deserialization method.
    fn from_json(json_str: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_json::from_str(json_str)?)
    }

    /// yaml serialization method.
    fn to_yaml(&self) -> String {
        serde_yaml::to_string(&self).unwrap()
    }

    /// yaml deserialization method.
    fn from_yaml(yaml_str: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_yaml::from_str(yaml_str)?)
    }

    /// bincode serialization method.
    fn to_bincode(&self) -> Vec<u8> {
        serialize(&self).unwrap()
    }

    /// bincode deserialization method.
    fn from_bincode(encoded: &[u8]) -> Result<Self, anyhow::Error> {
        Ok(deserialize(encoded)?)
    }
}

impl<T: SerdeAPI> SerdeAPI for Vec<T> {
    fn init(&mut self) -> anyhow::Result<()> {
        for val in self {
            val.init()?
        }
        Ok(())
    }
}

use std::fmt::{Debug, Display};
use std::ops::{Deref, DerefMut};

///Define a better trait bound for all error types!
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ComboError<E: Display> {
    layer: usize,
    error: E,
}

impl<E: Debug + Display> ComboError<E> {
    pub fn new(error: E) -> Self {
        Self { layer: 0, error }
    }
}

impl<E: Debug + Display> Deref for ComboError<E> {
    type Target = E;
    fn deref(&self) -> &E {
        &self.error
    }
}

impl<E: Debug + Display> DerefMut for ComboError<E> {
    fn deref_mut(&mut self) -> &mut E {
        &mut self.error
    }
}

pub struct ComboErrors<E: Debug + Display>(Vec<ComboError<E>>);

impl<E: Debug + Display> ComboErrors<E> {
    #[inline]
    pub fn new() -> Self {
        ComboErrors(vec![])
    }

    #[inline]
    pub fn add_context(&mut self, error_add: E) {
        for error in &mut self.0 {
            error.layer += 1;
        }
        self.0.insert(0, ComboError::<E>::new(error_add));
    }
    #[inline]
    pub fn push(&mut self, error_add: E) {
        self.0.push(ComboError::<E>::new(error_add));
    }

    #[inline]
    pub fn make_err(self) -> Result<(), Self> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(self)
        }
    }
}

impl<E: Debug + Display> Default for ComboErrors<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: Debug + Display> Deref for ComboErrors<E> {
    type Target = Vec<ComboError<E>>;
    fn deref(&self) -> &Vec<ComboError<E>> {
        &self.0
    }
}
impl<E: Debug + Display> DerefMut for ComboErrors<E> {
    fn deref_mut(&mut self) -> &mut Vec<ComboError<E>> {
        &mut self.0
    }
}
impl<E: Debug + Display> std::error::Error for ComboErrors<E> {}

impl<E: Debug + Display> Display for ComboErrors<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bullet = "- ";
        let tab = "  ";
        writeln!(f, "Combo error:")?;
        for error in &self.0 {
            writeln!(f, "{}{}{}", tab.repeat(error.layer), bullet, error.error)?;
        }
        Ok(())
    }
}

impl<E: Debug + Display> Debug for ComboErrors<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bullet = "- ";
        let tab = "  ";
        writeln!(f, "Combo error:")?;
        for error in &self.0 {
            writeln!(f, "{}{}{:?}", tab.repeat(error.layer), bullet, error.error)?;
        }
        Ok(())
    }
}

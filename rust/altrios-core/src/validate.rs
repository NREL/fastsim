use crate::combo_error::ComboErrors;
use crate::imports::*;
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use uom::si::Quantity;
use uom::ConstZero;

pub type ValidationError = anyhow::Error;
pub type ValidationErrors = ComboErrors<ValidationError>;
pub type ValidationResults = Result<(), ValidationErrors>;

///Generate valid default like input for use in other objects
pub trait Valid: Sized + Default {
    fn valid() -> Self {
        Default::default()
    }
}

///Specify when an object is valid, real, and fake
pub trait ObjState {
    fn is_fake(&self) -> bool {
        false
    }
    fn validate(&self) -> ValidationResults {
        Ok(())
    }
}

pub trait ObjStateConst: ObjState {
    fn is_real(&self) -> bool;
    fn is_valid(&self) -> bool;
    fn real(&self) -> Option<&Self>;
}

impl<T: ObjState> ObjStateConst for T {
    fn is_real(&self) -> bool {
        !self.is_fake()
    }
    fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
    fn real(&self) -> Option<&Self> {
        if self.is_fake() {
            return None;
        }
        Some(self)
    }
}

impl<T> ObjState for Vec<T>
where
    [T]: ObjState,
{
    fn is_fake(&self) -> bool {
        (**self).is_fake()
    }
    fn validate(&self) -> ValidationResults {
        (**self).validate()
    }
}

pub fn validate_slice_real<T>(errors: &mut ValidationErrors, slice: &[T], elem_name: &str)
where
    T: ObjState,
{
    validate_slice_real_shift(errors, slice, elem_name, 0)
}

pub fn validate_slice_real_shift<T>(
    errors: &mut ValidationErrors,
    slice: &[T],
    elem_name: &str,
    idx_shift: isize,
) where
    T: ObjState,
{
    for (index, val) in slice.iter().enumerate() {
        if val.is_fake() {
            errors.push(anyhow!(
                "{} at index = {} must be real!",
                elem_name,
                index as isize + idx_shift
            ));
        }
        if let Err(mut errors_add) = val.validate() {
            errors_add.add_context(anyhow!(
                "{} at index = {} must be valid!",
                elem_name,
                index as isize + idx_shift
            ));
            errors.append(&mut errors_add);
        }
    }
}

pub fn validate_slice_fake<T>(errors: &mut ValidationErrors, slice: &[T], elem_name: &str)
where
    T: ObjState,
{
    validate_slice_fake_shift(errors, slice, elem_name, 0)
}

pub fn validate_slice_fake_shift<T>(
    errors: &mut ValidationErrors,
    slice: &[T],
    elem_name: &str,
    idx_shift: isize,
) where
    T: ObjState,
{
    for (index, val) in slice.iter().enumerate() {
        if val.is_real() {
            errors.push(anyhow!(
                "{} at index = {} must be fake!",
                elem_name,
                index as isize + idx_shift
            ));
        }
        if let Err(mut errors_add) = val.validate() {
            errors_add.add_context(anyhow!(
                "{} at index = {} must be valid!",
                elem_name,
                index as isize + idx_shift
            ));
            errors.append(&mut errors_add);
        }
    }
}

pub fn validate_field_fake<T>(errors: &mut ValidationErrors, field_val: &T, field_name: &str)
where
    T: ObjState + Debug,
{
    if !field_val.is_fake() || field_val.is_real() {
        errors.push(anyhow!(
            "{} = {:?} must be fake and not real!",
            field_name,
            field_val
        ));
    }
    if let Err(mut errors_add) = field_val.validate() {
        errors_add.add_context(anyhow!("{} must be valid!", field_name));
        errors.append(&mut errors_add);
    }
}

pub fn validate_field_real<T>(errors: &mut ValidationErrors, field_val: &T, field_name: &str)
where
    T: ObjState + Debug,
{
    if !field_val.is_real() || field_val.is_fake() {
        errors.push(anyhow!(
            "{} = {:?} must be real and not fake!",
            field_name,
            field_val
        ));
    }
    if let Err(mut errors_add) = field_val.validate() {
        errors_add.add_context(anyhow!("{} must be valid!", field_name));
        errors.append(&mut errors_add);
    }
}

pub fn si_chk_num<D, U>(
    errors: &mut ValidationErrors,
    field_val: &Quantity<D, U, f64>,
    field_name: &str,
) where
    D: uom::si::Dimension + ?Sized,
    U: uom::si::Units<f64> + ?Sized,
{
    if field_val.is_nan() {
        errors.push(anyhow!(
            "{} = {:?} must be a number!",
            field_name,
            field_val
        ));
    }
}

pub fn si_chk_num_fin<D, U>(
    errors: &mut ValidationErrors,
    field_val: &uom::si::Quantity<D, U, f64>,
    field_name: &str,
) where
    D: uom::si::Dimension + ?Sized,
    U: uom::si::Units<f64> + ?Sized,
{
    if field_val.is_nan() || field_val.is_infinite() {
        errors.push(anyhow!(
            "{} = {:?} must be a finite number!",
            field_name,
            field_val
        ));
    }
}

/// Check that SI value is greater than or equal to zero
pub fn si_chk_num_gez<T>(errors: &mut ValidationErrors, field_val: &T, field_name: &str)
where
    T: Debug + PartialOrd + ConstZero,
{
    if let None | Some(Ordering::Less) = field_val.partial_cmp(&T::ZERO) {
        errors.push(anyhow!(
            "{} = {:?} must be a positive number!",
            field_name,
            field_val
        ));
    }
}

pub fn si_chk_num_gtz<T>(errors: &mut ValidationErrors, field_val: &T, field_name: &str)
where
    T: Debug + PartialOrd + ConstZero,
{
    if let None | Some(Ordering::Less) | Some(Ordering::Equal) = field_val.partial_cmp(&T::ZERO) {
        errors.push(anyhow!(
            "{} = {:?} must be a number larger than zero!",
            field_name,
            field_val
        ));
    }
}

pub fn si_chk_num_gez_fin<D, U>(
    errors: &mut ValidationErrors,
    field_val: &uom::si::Quantity<D, U, f64>,
    field_name: &str,
) where
    D: uom::si::Dimension + ?Sized,
    U: uom::si::Units<f64> + ?Sized,
{
    if !(*field_val >= uom::si::Quantity::<D, U, f64>::ZERO && field_val.is_finite()) {
        errors.push(anyhow!(
            "{} = {:?} must be a finite positive number!",
            field_name,
            field_val
        ));
    }
}

pub fn si_chk_num_gtz_fin<D, U>(
    errors: &mut ValidationErrors,
    field_val: &uom::si::Quantity<D, U, f64>,
    field_name: &str,
) where
    D: uom::si::Dimension + ?Sized,
    U: uom::si::Units<f64> + ?Sized,
{
    if !(*field_val > uom::si::Quantity::<D, U, f64>::ZERO && field_val.is_finite()) {
        errors.push(anyhow!(
            "{} = {:?} must be a finite number larger than zero!",
            field_name,
            field_val
        ));
    }
}

pub fn si_chk_num_eqz<T>(errors: &mut ValidationErrors, field_val: &T, field_name: &str)
where
    T: Debug + PartialEq + ConstZero,
{
    if *field_val != T::ZERO {
        errors.push(anyhow!("{} = {:?} must equal zero!", field_name, field_val));
    }
}

macro_rules! early_err {
    ($errors:expr, $name:expr) => {
        if !$errors.is_empty() {
            $errors.push(anyhow!("{} validation unfinished!", $name));
            return Err($errors);
        }
    };
}

macro_rules! early_fake_ok {
    ($self:expr) => {
        if $self.is_fake() {
            return Ok(());
        }
    };
}

pub(crate) use {early_err, early_fake_ok};

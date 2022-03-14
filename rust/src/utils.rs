extern crate ndarray;
use ndarray::{Array1, array, Axis, s, concatenate}; 
use ordered_float::NotNan; // 2.0.0


pub fn diff(x:&Array1<f64>) -> Array1<f64>{
    concatenate(Axis(0), 
        &[array![0.0].view(),
        (&x.slice(s![1..]) - &x.slice(s![..-1])).view()]
    ).unwrap()
}

/// Emulates numpy.argmax
/// Arguments:
/// arr: Array1<bool> -- array of bools
/// Returns index of first true value, or last index if all false
pub fn np_argmax(arr: &Array1<bool>) -> usize {
    arr.iter().position(|&x| x).unwrap_or(arr.len() - 1) // unwrap_or allows for default if not found
}

/// return max of 2 f64
pub fn max(a:f64, b:f64) -> f64 {
    a.max(b)
}

/// return min of 2 f64
pub fn min(a:f64, b:f64) -> f64 {
    a.min(b)
}

/// return max <f64> of arr
pub fn arrmax(arr:&Array1<f64>) -> f64 {
    arr
        .iter()
        .copied()
        .map(NotNan::new)
        .flatten() // ignore NAN values (errors from the previous line)
        .max()
        .map(NotNan::into_inner)
        .unwrap()    
}

/// return max <f64> of arr
pub fn arrmin(arr:&Array1<f64>) -> f64 {
    arr
        .iter()
        .copied()
        .map(NotNan::new)
        .flatten() // ignore NAN values (errors from the previous line)
        .min()
        .map(NotNan::into_inner)
        .unwrap()    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff(){
        assert_eq!(diff(&Array1::range(0.0, 3.0, 1.0)), array![0.0, 1.0, 1.0]);
    }
}
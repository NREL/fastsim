extern crate ndarray;
use ndarray::{Array1, array, Axis, s, concatenate}; 


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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff(){
        assert_eq!(diff(&Array1::range(0.0, 3.0, 1.0)), array![0.0, 1.0, 1.0]);
    }
}
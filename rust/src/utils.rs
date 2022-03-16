extern crate ndarray;
use ndarray::{Array, Array1, array, Axis, s, concatenate}; 
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
/// if array is of len() == 0, returns 0.
pub fn np_argmax(arr: &Array1<bool>) -> Option<usize> {
    let len = arr.len();
    if len == 0 {
        return None;
    }
    Some(arr.iter().position(|&x| x).unwrap_or(len - 1)) // unwrap_or allows for default if not found
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

// /// interpolation algorithm from http://www.cplusplus.com/forum/general/216928/
// fn interpolate(x_data:&Array1<f64>, y_data:&Array1<f64>, x:&f64, extrapolate:bool) -> f64 {
//     let size = x_data.len();
// 
//     let mut i = 0;
//     if x >= &x_data[size - 2] {
//         i = size - 2;
//     } else {
//         while x > &x_data[i + 1]{
//             i += 1;
//         }
//     }
//     let xl = &x_data[i];
//     let mut yl = &y_data[i]; 
//     let xr = &x_data[i + 1]; 
//     let mut yr = &y_data[i + 1];
//     if !extrapolate {
//         if x < xl{
//             yr = yl;
//         }
//         if x > xr{
//             yl = yr;
//         }
//     }
//     let dydx = (yr - yl) / (xr - xl);
//     yl + dydx * (x - xl)
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff(){
        assert_eq!(diff(&Array1::range(0.0, 3.0, 1.0)), array![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_that_np_argmax_finds_the_right_index_when_one_exists(){
        let xs: Array1<bool> = Array::from_vec(vec![false, false, true, false, false]);
        let idx = np_argmax(&xs).unwrap();
        let expected_idx: usize = 2;
        assert_eq!(idx, expected_idx)
    }

    #[test]
    fn test_that_np_argmax_yields_last_index_when_nothing_found(){
        let xs: Array1<bool> = Array::from_vec(vec![false, false, false, false, false]);
        let idx = np_argmax(&xs).unwrap();
        let expected_idx: usize = xs.len() - 1;
        assert_eq!(idx, expected_idx)
    }

    #[test]
    fn test_that_argmax_does_the_right_thing_on_an_empty_array(){
        let xs: Array1<bool> = Array::from_vec(vec![]);
        let idx = np_argmax(&xs);
        // unclear what should happen here; np.argmax throws a ValueError in the case of an empty vector
        // ... possibly we should return an Option type?
        let expected_idx:Option<usize> = None; 
        assert_eq!(idx, expected_idx);
    }

    // #[test]
    // fn test_that_interpolation_works(){
    //     let xs: Array1<f64> = Array::from_vec(vec![0.0,  1.0,  2.0,  3.0,  4.0]);
    //     let ys: Array1<f64> = Array::from_vec(vec![0.0, 10.0, 20.0, 30.0, 40.0]);
    //     let x: f64 = 0.5;
    //     let y_lookup = interpolate(xs, ys, x, false);
    // }
}
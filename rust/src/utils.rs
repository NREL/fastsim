extern crate ndarray;
use ndarray::{Array1, array, Axis, s, concatenate}; 
use ordered_float::NotNan; // 2.0.0


pub fn diff(x:&Array1<f64>) -> Array1<f64>{
    concatenate(Axis(0), 
        &[array![0.0].view(),
        (&x.slice(s![1..]) - &x.slice(s![..-1])).view()]
    ).unwrap()
}


/// Return first index of `arr` greater than `cut`
pub fn first_grtr(arr: &[f64], cut: f64) -> Option<usize> {
    let len = arr.len();
    if len == 0 {
        return None;
    }
    Some(arr.iter().position(|&x| x > cut).unwrap_or(len - 1)) // unwrap_or allows for default if not found
}

/// Return first index of `arr` equal to`cut`
pub fn first_eq(arr: &[f64], cut: f64) -> Option<usize> {
    let len = arr.len();
    if len == 0 {
        return None;
    }
    Some(arr.iter().position(|&x| x == cut).unwrap_or(len - 1)) // unwrap_or allows for default if not found
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
pub fn arrmax(arr:&[f64]) -> f64 {
    arr.iter().copied().fold(f64::NAN, f64::max)
}

/// return min <f64> of arr
pub fn arrmin(arr:&[f64]) -> f64 {
    arr.iter().copied().fold(f64::NAN, f64::min)
}

/// return min <f64> of arr
pub fn ndarrmin(arr:&Array1<f64>) -> f64 {
    arr.to_vec()
        .into_iter()
        .reduce(f64::min)
        .unwrap()
}

/// return max <f64> of arr
pub fn ndarrmax(arr:&Array1<f64>) -> f64 {
    arr.to_vec()
        .into_iter()
        .reduce(f64::max)
        .unwrap()
}

/// return cumsum <f64> of arr
pub fn ndarrcumsum(arr:&Array1<f64>) -> Array1<f64> {
    arr.iter().scan(0.0, |acc, &x| {
		*acc += x;
		Some(*acc)
	}).collect()
}

// TODO: if interpolation is used at each time step, change it to take native, fixed-size array
/// interpolation algorithm from http://www.cplusplus.com/forum/general/216928/
/// Arguments:
/// x : value at which to interpolate
pub fn interpolate(x:&f64, x_data:&Array1<f64>, y_data:&Array1<f64>, extrapolate:bool) -> f64 {
    let size = x_data.len();

    let mut i = 0;
    if x >= &x_data[size - 2] {
        i = size - 2;
    } else {
        while x > &x_data[i + 1]{
            i += 1;
        }
    }
    let xl = &x_data[i];
    let mut yl = &y_data[i]; 
    let xr = &x_data[i + 1]; 
    let mut yr = &y_data[i + 1];
    if !extrapolate {
        if x < xl{
            yr = yl;
        }
        if x > xr{
            yl = yr;
        }
    }
    let dydx = (yr - yl) / (xr - xl);
    yl + dydx * (x - xl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff(){
        assert_eq!(diff(&Array1::range(0.0, 3.0, 1.0)), array![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_that_first_eq_finds_the_right_index_when_one_exists(){
        let xs: [f64; 5] = [0.0, 1.2, 3.3, 4.4, 6.6];
        let idx = first_eq(&xs, 3.3).unwrap();
        let expected_idx: usize = 2;
        assert_eq!(idx, expected_idx)
    }

    #[test]
    fn test_that_first_eq_yields_last_index_when_nothing_found(){
        let xs: [f64; 5] = [0.0, 1.2, 3.3, 4.4, 6.6];
        let idx = first_eq(&xs, 7.0).unwrap();
        let expected_idx: usize = xs.len() - 1;
        assert_eq!(idx, expected_idx)
    }


    #[test]
    fn test_that_first_grtr_finds_the_right_index_when_one_exists(){
        let xs: [f64; 5] = [0.0, 1.2, 3.3, 4.4, 6.6];
        let idx = first_grtr(&xs, 3.0).unwrap();
        let expected_idx: usize = 2;
        assert_eq!(idx, expected_idx)
    }

    #[test]
    fn test_that_first_grtr_yields_last_index_when_nothing_found(){
        let xs: [f64; 5] = [0.0, 1.2, 3.3, 4.4, 6.6];
        let idx = first_grtr(&xs, 7.0).unwrap();
        let expected_idx: usize = xs.len() - 1;
        assert_eq!(idx, expected_idx)
    }

    #[test]
    fn test_that_ndarrmin_returns_the_min(){
        let xs = Array1::from_vec(vec![10.0, 80.0, 3.0, 3.2, 9.0]);
        let xmin = ndarrmin(&xs);
        assert_eq!(xmin, 3.0);
    }

    #[test]
    fn test_that_ndarrmax_returns_the_max(){
        let xs = Array1::from_vec(vec![10.0, 80.0, 3.0, 3.2, 9.0]);
        let xmax = ndarrmax(&xs);
        assert_eq!(xmax, 80.0);
    }

    #[test]
    fn test_ndarrcumsum_expected_output(){
        let xs = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let expected_ys = Array1::from_vec(vec![0.0, 1.0, 3.0, 6.0]);
        let ys = ndarrcumsum(&xs);
        let mut i: usize = 0;
        for (ye, y) in expected_ys.iter().zip(ys.iter()) {
            assert_eq!(ye, y, "unequal at {}", i);
            i += 1;
        }
    }

    // #[test]
    // fn test_that_argmax_does_the_right_thing_on_an_empty_array(){
    //     let xs: Array1<bool> = Array::from_vec(vec![]);
    //     let idx = first_grtr(&xs);
    //     // unclear what should happen here; np.argmax throws a ValueError in the case of an empty vector
    //     // ... possibly we should return an Option type?
    //     let expected_idx:Option<usize> = None; 
    //     assert_eq!(idx, expected_idx);
    // }

    // #[test]
    // fn test_that_interpolation_works(){
    //     let xs: Array1<f64> = Array::from_vec(vec![0.0,  1.0,  2.0,  3.0,  4.0]);
    //     let ys: Array1<f64> = Array::from_vec(vec![0.0, 10.0, 20.0, 30.0, 40.0]);
    //     let x: f64 = 0.5;
    //     let y_lookup = interpolate(xs, ys, x, false);
    // }
}
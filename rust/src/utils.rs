extern crate ndarray;
use ndarray::{Array1, array, Axis, s, concatenate}; 


pub fn diff(x:&Array1<f64>) -> Array1<f64>{
    concatenate(Axis(0), 
        &[array![0.0].view(),
        (&x.slice(s![1..]) - &x.slice(s![..-1])).view()]
    ).unwrap()
}


// TODO: implement this for emulating np.argmax:
// https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=a338a0aba3e8c901e53863d093f642d3

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff(){
        assert_eq!(diff(&Array1::range(0.0, 3.0, 1.0)), array![0.0, 1.0, 1.0]);
    }
}
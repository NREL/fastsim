extern crate ndarray;
use ndarray::{Array1, array, Axis, s, concatenate}; 


pub fn diff(x:&Array1<f64>) -> Array1<f64>{
    concatenate(Axis(0), 
        &[array![0.0].view(),
        (&x.slice(s![1..]) - &x.slice(s![..-1])).view()]
    ).unwrap()
}


// TODO: implement this for emulating np.argmax:
// nearly workable function for returning first true index of vector
// https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=ab83986ac69e13f9f1e97d7ced7b1dd8
// `first_true` fn seems to run faster than `position` method but probably needs to handle all false case


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff(){
        assert_eq!(diff(&Array1::range(0.0, 3.0, 1.0)), array![0.0, 1.0, 1.0]);
    }
}
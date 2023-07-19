use crate::utils;
use crate::validate::*;

pub trait Cases: Valid {
    fn real_cases() -> Vec<Self> {
        vec![Valid::valid()]
    }
    fn fake_cases() -> Vec<Self> {
        vec![]
    }
    fn invalid_cases() -> Vec<Self> {
        vec![]
    }
}

pub fn test_cases<T>()
where
    T: ObjState + Cases,
{
    for case_real in T::real_cases() {
        case_real.validate().unwrap();
        assert!(case_real.is_real());
        assert!(!case_real.is_fake());
    }
    for case_fake in T::fake_cases() {
        case_fake.validate().unwrap();
        assert!(!case_fake.is_real());
        assert!(case_fake.is_fake());
    }
    for case_invalid in T::invalid_cases() {
        case_invalid.validate().unwrap_err();
    }
}

pub fn test_vec_elems<T>()
where
    T: ObjState + Cases + Clone,
    Vec<T>: ObjState + Cases,
{
    if !T::invalid_cases().is_empty() {
        let invalid_elem = T::invalid_cases().first().unwrap().clone();
        for case in Vec::<T>::real_cases()
            .iter()
            .chain(Vec::<T>::fake_cases().iter())
        {
            for idx_invalid in 0..case.len() {
                let mut vec = case.clone();
                vec[idx_invalid] = invalid_elem.clone();
                vec.validate().unwrap_err();
            }
        }
    }
}

pub fn test_vec_sorted<T>()
where
    T: PartialOrd,
    Vec<T>: ObjState + Cases,
{
    for case_real in Vec::<T>::real_cases() {
        let case_flip = case_real.into_iter().rev().collect::<Vec<T>>();
        if !utils::is_sorted(&case_flip) {
            case_flip.validate().unwrap_err();
        }
    }
}

pub fn test_vec_duplicates<T>()
where
    T: PartialEq + Clone,
    Vec<T>: ObjState + Cases,
{
    for mut case_real in Vec::<T>::real_cases() {
        if !case_real.is_empty() {
            case_real.push(case_real.last().unwrap().clone());
            case_real.validate().unwrap_err();
        }
    }
}

macro_rules! check_cases {
    ($T:ty) => {
        #[test]
        fn check_cases() {
            test_cases::<$T>();
        }
    };
}

macro_rules! check_vec_elems {
    ($T:ty) => {
        #[test]
        fn check_vec_elems() {
            test_vec_elems::<$T>();
        }
    };
}

macro_rules! check_vec_sorted {
    ($T:ty) => {
        #[test]
        fn check_vec_sorted() {
            test_vec_sorted::<$T>();
        }
    };
}

macro_rules! check_vec_duplicates {
    ($T:ty) => {
        #[test]
        fn check_vec_duplicates() {
            test_vec_duplicates::<$T>();
        }
    };
}

pub(crate) use {check_cases, check_vec_duplicates, check_vec_elems, check_vec_sorted};

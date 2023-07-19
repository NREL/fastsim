pub struct A {
    pub a: i32,
    pub b: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive() {
        let a = A{a: 1, b: 2};
        let a2 = A{a: 1, b: 2};
    }
}

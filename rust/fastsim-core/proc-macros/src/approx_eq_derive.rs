use crate::imports::*;

pub fn approx_eq_derive(input: TokenStream) -> TokenStream {
    let ast: DeriveInput = syn::parse(input).unwrap();
    let name = &ast.ident;

    let mut fields = Vec::new();
    match ast.data {
        syn::Data::Struct(s) => {
            for field in s.fields.iter() {
                fields.push(field.clone());
            }
        }
        _ => panic!("#[derive(ApproxEq)] only works on structs"),
    }

    let field_names = fields
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect::<Vec<_>>();

    let mut generated = TokenStream2::new();
    generated.append_all(quote! {
        impl ApproxEq for #name {
            fn approx_eq(&self, other: &#name, tol: f64) -> bool {
                let mut approx_eq_vals: Vec<bool> = Vec::new();
                #(approx_eq_vals.push(self.#field_names.approx_eq(&other.#field_names, tol));)*
                approx_eq_vals.iter().all(|&x| x)
            }
        }
    });
    generated.into()
}

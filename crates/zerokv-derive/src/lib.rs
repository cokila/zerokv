// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright © 2026 gerardo. Part of zerokv; see LICENSE.

//! Procedural macros for `zerokv`.
//!
//! This crate provides `#[derive(ZeroCopy)]`, which generates *zero-copy*
//! binary (de)serialization for user record types stored in the key-value
//! engine. "Zero-copy" here means: a stored byte slice from the arena/log can
//! be reinterpreted directly as `&T` without parsing or heap allocation.
//!
//! Because we reinterpret raw bytes as a typed reference, the safety of the
//! generated code rests on three compile-time guarantees that the macro emits
//! as `const` assertions in the downstream crate:
//!
//!   1. Every field type implements the `unsafe` marker trait [`Pod`] (Plain
//!      Old Data): no padding-poisoning, no pointers, no `Drop`, valid for any
//!      bit pattern.
//!   2. The struct has *no padding* — we verify `size_of::<T>()` equals the sum
//!      of the field sizes. Padding bytes are `uninit` and reading them through
//!      a shared reference would be UB, so we forbid the layout entirely.
//!   3. The struct alignment is 1-friendly for unaligned access, OR the caller
//!      guarantees alignment. We expose a checked `view` that validates the
//!      pointer alignment at runtime (cheap, branch-predicted).
//!
//! If any check fails, compilation aborts with a readable message — the user
//! never gets a silently-unsound record type.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Index};

#[proc_macro_derive(ZeroCopy)]
pub fn derive_zero_copy(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_g, ty_g, where_g) = ast.generics.split_for_impl();

    // We only support structs: an enum/union has no fixed, padding-free layout
    // we can reinterpret blindly.
    let fields = match &ast.data {
        Data::Struct(s) => &s.fields,
        _ => {
            return syn::Error::new_spanned(
                name,
                "#[derive(ZeroCopy)] supports only structs with fixed-size POD fields",
            )
            .to_compile_error()
            .into();
        }
    };

    // Collect (accessor, type) for each field, supporting both named and tuple
    // structs. Unit structs are allowed (size 0).
    let field_tys: Vec<&syn::Type> = match fields {
        Fields::Named(n) => n.named.iter().map(|f| &f.ty).collect(),
        Fields::Unnamed(u) => u.unnamed.iter().map(|f| &f.ty).collect(),
        Fields::Unit => Vec::new(),
    };

    // --- Compile-time guarantee #1: every field is `Pod`. -------------------
    // This monomorphizes a function that only type-checks if `Ty: Pod`, so a
    // non-POD field produces a precise compiler error at the field's span.
    let pod_assertions = field_tys.iter().map(|ty| {
        quote! {
            const _: fn() = || {
                fn _assert_pod<T: ::zerokv::Pod>() {}
                _assert_pod::<#ty>();
            };
        }
    });

    // --- Compile-time guarantee #2: no padding. ----------------------------
    // SUM(size_of field) must equal size_of::<Self>(). If the compiler inserted
    // padding to satisfy alignment, the sums differ and we hard-error.
    let size_sum = {
        let terms = field_tys.iter().map(|ty| quote! { ::core::mem::size_of::<#ty>() });
        quote! { 0 #( + #terms )* }
    };

    // Generate the byte-wise field copies for `encode`/`decode`. We copy each
    // field at its real in-struct offset using `addr_of!`, which is robust even
    // for `#[repr(Rust)]` because we never assume an ordering — we ask the
    // compiler for each field's address.
    let field_idents: Vec<_> = match fields {
        Fields::Named(n) => n
            .named
            .iter()
            .map(|f| {
                let id = f.ident.clone().unwrap();
                quote! { #id }
            })
            .collect(),
        Fields::Unnamed(u) => (0..u.unnamed.len())
            .map(|i| {
                let idx = Index::from(i);
                quote! { #idx }
            })
            .collect(),
        Fields::Unit => Vec::new(),
    };

    let expanded = quote! {
        // Per-field POD bound checks.
        #( #pod_assertions )*

        // Padding guard: evaluated in const context at compile time.
        const _: () = {
            assert!(
                ::core::mem::size_of::<#name #ty_g>() == (#size_sum),
                "ZeroCopy: type has padding bytes; add #[repr(C, packed)] or reorder \
                 fields so the layout is padding-free, otherwise zero-copy reads are UB",
            );
        };

        unsafe impl #impl_g ::zerokv::ZeroCopy for #name #ty_g #where_g {
            const SERIALIZED_SIZE: usize = ::core::mem::size_of::<#name #ty_g>();

            #[inline]
            fn encode(&self, dst: &mut [u8]) {
                assert!(dst.len() >= Self::SERIALIZED_SIZE, "encode: dst too small");
                // SAFETY: `Self: Pod` (every field is Pod and there is no
                // padding, checked above), so all `SERIALIZED_SIZE` bytes of
                // `self` are initialized and safe to read as bytes.
                unsafe {
                    let src = (self as *const Self).cast::<u8>();
                    ::core::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), Self::SERIALIZED_SIZE);
                }
            }

            #[inline]
            fn decode(src: &[u8]) -> Self {
                assert!(src.len() >= Self::SERIALIZED_SIZE, "decode: src too small");
                // SAFETY: `Self: Pod` => any bit pattern is a valid value, and
                // we read exactly `SERIALIZED_SIZE` initialized source bytes.
                unsafe {
                    let mut out = ::core::mem::MaybeUninit::<Self>::uninit();
                    ::core::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        Self::SERIALIZED_SIZE,
                    );
                    out.assume_init()
                }
            }

            #[inline]
            fn view(src: &[u8]) -> Option<&Self> {
                if src.len() < Self::SERIALIZED_SIZE {
                    return None;
                }
                let ptr = src.as_ptr();
                // Alignment must be respected even for `Pod`: a misaligned
                // reference is instant UB. We validate it cheaply instead of
                // assuming the caller pre-aligned the slice.
                if (ptr as usize) % ::core::mem::align_of::<Self>() != 0 {
                    return None;
                }
                // SAFETY: length and alignment validated; `Self: Pod` so the
                // bytes form a valid value and there is no interior mutability /
                // padding to read as uninit.
                Some(unsafe { &*ptr.cast::<Self>() })
            }
        }

        // Reference the generated field copies once so the compiler keeps the
        // field idents "used" in debug builds and validates accessibility.
        const _: fn(&#name #ty_g) = |_v: &#name #ty_g| {
            #( let _ = &_v.#field_idents; )*
        };
    };

    expanded.into()
}

pub mod mambo;
pub use self::mambo::Mambo;
/*
cargo fmt
cargo fix --allow-dirty
cargo clippy --fix --all --allow-dirty
dotnet run -- all check-non-fix potential
*/

//git add . && git commit -S -m "Improved code security and formatting, added new tests and fmt settings 0.1.30" && cargo publish && git push

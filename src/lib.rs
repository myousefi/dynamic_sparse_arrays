pub mod buffer;
pub mod finds;
pub mod matrix;
pub mod moves;
pub mod operations;
pub mod pcsr;
pub mod pma;
pub mod utils;
pub mod vector;
pub mod views;
pub mod writes;

pub use crate::vector::{dynamicsparsevec, shrink_size, DynamicSparseVector};
pub use crate::matrix::DynamicSparseMatrix;
pub use crate::views::DynamicMatrixColView;
pub use crate::matrix::{dynamicsparse, deletecolumn!, deleterow!, addrow!, closefillmode!};

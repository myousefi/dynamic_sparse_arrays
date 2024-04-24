use crate::pma::{PackedMemoryArray, NoPredictor};
use std::cmp::max;
use std::ops::{Index, IndexMut};

pub struct DynamicSparseVector<K, T> {
    n: K,
    pma: PackedMemoryArray<K, T, NoPredictor>,
}

fn _guess_length<K, T, P>(pma: &PackedMemoryArray<K, T, P>) -> K
where
    K: Default + Ord,
{
    pma.iter().map(|e| e.0).max().unwrap_or_default()
}

fn _prepare_keys_vals<K, T, F>(keys: &mut Vec<K>, values: &mut Vec<T>, combine: F)
where
    K: Ord,
    F: Fn(T, T) -> T,
{
    assert_eq!(keys.len(), values.len());
    if keys.is_empty() {
        return;
    }
    let p = keys.iter().enumerate().sorted_by_key(|&(_, k)| k).map(|(i, _)| i).collect::<Vec<_>>();
    keys.sort();
    values.permute(&p);
    let mut write_pos = 0;
    let mut read_pos = 0;
    let mut prev_id = keys[read_pos].clone();
    while read_pos < keys.len() {
        read_pos += 1;
        let cur_id = keys[read_pos].clone();
        if prev_id == cur_id {
            values[write_pos] = combine(values[write_pos].clone(), values[read_pos].clone());
        } else {
            write_pos += 1;
            if write_pos < read_pos {
                keys[write_pos] = cur_id.clone();
                values[write_pos] = values[read_pos].clone();
            }
        }
        prev_id = cur_id;
    }
    keys.truncate(write_pos);
    values.truncate(write_pos);
}

fn _dynamicsparsevec<K, T, F>(i: Vec<K>, v: Vec<T>, combine: F, n: K) -> DynamicSparseVector<K, T>
where
    K: Ord + Clone,
    T: Clone,
    F: Fn(T, T) -> T,
{
    let mut keys = i;
    let mut values = v;
    _prepare_keys_vals(&mut keys, &mut values, combine);
    let pma = PackedMemoryArray::from_vecs(keys, values);
    DynamicSparseVector { n, pma }
}

pub fn dynamicsparsevec<K, T, F>(i: Vec<K>, v: Vec<T>, combine: F, n: Option<K>) -> DynamicSparseVector<K, T>
where
    K: Ord + Clone + Default,
    T: Clone + Default,
    F: Fn(T, T) -> T,
{
    assert_eq!(i.len(), v.len(), "keys & nonzeros vectors must have same length.");
    let n = n.unwrap_or_else(|| _guess_length(&i));
    _dynamicsparsevec(i, v, combine, n)
}

pub fn shrink_size<K, T>(v: &mut DynamicSparseVector<K, T>)
where
    K: Default + Ord,
{
    v.n = _guess_length(&v.pma);
}

impl<K, T> DynamicSparseVector<K, T>
where
    K: Ord + Clone + Default,
    T: Clone + Default,
{
    pub fn length(&self) -> K {
        self.n
    }

    pub fn size(&self) -> (K,) {
        (self.n,)
    }

    pub fn nnz(&self) -> usize {
        self.pma.nnz()
    }

    pub fn iter(&self) -> impl Iterator<Item = (K, T)> + '_ {
        self.pma.iter().map(|e| e.clone())
    }

    pub fn nonzeroinds(&self) -> Vec<K> {
        self.pma.iter().map(|e| e.0.clone()).collect()
    }

    pub fn nonzeros(&self) -> Vec<T> {
        self.pma.iter().map(|e| e.1.clone()).collect()
    }
}

impl<K, T> Index<K> for DynamicSparseVector<K, T>
where
    K: Ord + Clone,
    T: Default,
{
    type Output = T;

    fn index(&self, key: K) -> &Self::Output {
        self.pma.get(&key).unwrap_or(&T::default())
    }
}

impl<K, T> IndexMut<K> for DynamicSparseVector<K, T>
where
    K: Ord + Clone,
    T: Default,
{
    fn index_mut(&mut self, key: K) -> &mut Self::Output {
        if !self.pma.contains_key(&key) {
            self.n = max(self.n.clone(), key);
        }
        self.pma.entry(key).or_insert(T::default())
    }
}

impl<K, T> PartialEq for DynamicSparseVector<K, T>
where
    K: PartialEq,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.pma == other.pma
    }
}

impl<K, T> Eq for DynamicSparseVector<K, T>
where
    K: Eq,
    T: Eq,
{
}

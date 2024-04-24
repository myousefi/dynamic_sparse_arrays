use crate::pma::{PackedMemoryArray, NoPredictor};
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Matrix whose columns are indexed by an integer.
struct PackedCSC<K, T> {
    nb_partitions: usize,
    semaphores: Vec<Option<usize>>,
    pma: PackedMemoryArray<K, T, NoPredictor>,
}

impl<K, T> PackedCSC<K, T>
where
    K: Ord + Clone,
    T: Clone + From<usize>,
{
    fn new(row_keys: Vec<Vec<K>>, values: Vec<Vec<T>>, combine: fn(T, T) -> T) -> Self {
        let nb_semaphores = row_keys.len();
        let nb_values = values.iter().map(|v| v.len()).sum();
        assert_eq!(nb_semaphores, values.len());

        let mut pcsc_keys = Vec::with_capacity(nb_values + nb_semaphores);
        let mut pcsc_values = Vec::with_capacity(nb_values + nb_semaphores);

        for semaphore_id in 0..nb_semaphores {
            // Insert the semaphore
            pcsc_keys.push(semaphore_key::<K>());
            pcsc_values.push(T::from(semaphore_id));

            // Create the column
            let mut nkeys = row_keys[semaphore_id].clone();
            let mut nvalues = values[semaphore_id].clone();
            _prepare_keys_vals(&mut nkeys, &mut nvalues, combine);

            for j in 0..nkeys.len() {
                pcsc_keys.push(nkeys[j].clone());
                pcsc_values.push(nvalues[j].clone());
            }
        }

        let pma = PackedMemoryArray::from_vecs(pcsc_keys, pcsc_values);
        let mut semaphores = vec![None; nb_semaphores];

        for (pos, pair) in pma.iter().enumerate() {
            if pair.0 == semaphore_key::<K>() {
                let id = pair.1.clone().into();
                semaphores[id] = Some(pos);
            }
        }

        PackedCSC {
            nb_partitions: nb_semaphores,
            semaphores,
            pma,
        }
    }

    // TODO: Implement other methods
}

/// Matrix
struct MappedPackedCSC<K, L, T> {
    col_keys: Vec<Option<L>>,
    pcsc: PackedCSC<K, T>,
}

impl<K, L, T> MappedPackedCSC<K, L, T>
where
    K: Ord + Clone,
    L: Clone,
    T: Clone + From<usize>,
{
    fn new(row_keys: Vec<Vec<K>>, column_keys: Vec<L>, values: Vec<Vec<T>>, combine: fn(T, T) -> T) -> Self {
        let pcsc = PackedCSC::new(row_keys, values, combine);
        let col_keys = column_keys.into_iter().map(Some).collect();

        MappedPackedCSC { col_keys, pcsc }
    }

    // TODO: Implement other methods
}

fn semaphore_key<K: Default>() -> K {
    K::default()
}

fn _prepare_keys_vals<K: Ord, T>(keys: &mut Vec<K>, values: &mut Vec<T>, combine: fn(T, T) -> T) {
    let mut i = 0;
    while i < keys.len() {
        let mut j = i + 1;
        while j < keys.len() && keys[i] == keys[j] {
            values[i] = combine(values[i].clone(), values[j].clone());
            j += 1;
        }
        keys.drain(i + 1..j);
        values.drain(i + 1..j);
        i += 1;
    }
}

// TODO: Implement remaining functions and methods

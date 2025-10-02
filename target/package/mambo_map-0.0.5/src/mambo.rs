use std::{
    mem,
    sync::{Arc, Mutex, RwLock, TryLockError},
};
pub const PRIME_NUMBERS_TO_TAB: [u64; 64] = [
    0x2,                    //default
    0x5,                    //0
    0x7,                    //1
    0xb_u64,                // 2
    0x17_u64,               // 3
    0x2f_u64,               // 4
    0x61_u64,               // 5
    0xc7_u64,               // 6
    0x199_u64,              // 7
    0x337_u64,              // 8
    0x6cd_u64,              // 9
    0xd8d_u64,              // 10
    0x1b25_u64,             // 11
    0x36d1_u64,             // 12
    0x6efb_u64,             // 13
    0xe0d5_u64,             // 14
    0x1c7fb_u64,            // 15
    0x39d61_u64,            // 16
    0x75671_u64,            // 17
    0xee5f1_u64,            // 18
    0x1e40a3_u64,           // 19
    0x3d6eaf_u64,           // 20
    0x7cbf17_u64,           // 21
    0xfd51f9_u64,           // 22
    0x2026a59_u64,          // 23
    0x4149f67_u64,          // 24
    0x8495051_u64,          // 25
    0x10d3c05f_u64,         // 26
    0x222bc111_u64,         // 27
    0x45641269_u64,         // 28
    0x8ce98529_u64,         // 29
    0xfffffffb_u64,         // 30
    0x1fffffff7_u64,        // 31
    0x3ffffffd7_u64,        // 32
    0x7ffffffe1_u64,        // 33
    0xffffffffb_u64,        // 34
    0x1fffffffe7_u64,       // 35
    0x3fffffffd3_u64,       // 36
    0x7ffffffff9_u64,       // 37
    0xffffffffa9_u64,       // 38
    0x1ffffffffeb_u64,      // 39
    0x3fffffffff5_u64,      // 40
    0x7ffffffffc7_u64,      // 41
    0xfffffffffef_u64,      // 42
    0x1fffffffffc9_u64,     // 43
    0x3fffffffffeb_u64,     // 44
    0x7fffffffff8d_u64,     // 45
    0xffffffffffc5_u64,     // 46
    0x1ffffffffffaf_u64,    // 47
    0x3ffffffffffe5_u64,    // 48
    0x7ffffffffff7f_u64,    // 49
    0xfffffffffffd1_u64,    // 50
    0x1fffffffffff91_u64,   // 51
    0x3fffffffffffdf_u64,   // 52
    0x7fffffffffffc9_u64,   // 53
    0xfffffffffffffb_u64,   // 54
    0x1fffffffffffff3_u64,  // 55
    0x3ffffffffffffe5_u64,  // 56
    0x7ffffffffffffc9_u64,  // 57
    0xfffffffffffffa3_u64,  // 58
    0x1fffffffffffffff_u64, // 59
    0x3fffffffffffffc7_u64, // 60
    0x7fffffffffffffe7_u64, // 61
    0xffffffffffffffc5_u64, // 62
];

const EPSILON: f32 = 0.0001;
const SHRINK_THRESHOLD_FACTOR: f32 = 1.25;
const WASMOVED: bool = false;
const ONPLACE: bool = true;
const ELEM_NUMS_TO_READ_IN_MUTEX: u64 = 32;
type ElemBox<T> = (T, u64);
type Vecta<T> = Mutex<(bool, Vec<ElemBox<T>>)>;
type Shard<T> = (
    Mutex<u32>,   //locler os resize
    Mutex<usize>, //elems in srard
    RwLock<[Box<[Vecta<T>]>; 2]>,
);

enum WhatIdo {
    READ,
    REMOVE,
    INSERT,
}

impl PartialEq for WhatIdo {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (WhatIdo::INSERT, WhatIdo::INSERT) => true,
            (WhatIdo::REMOVE, WhatIdo::REMOVE) => true,
            (WhatIdo::READ, WhatIdo::READ) => true,
            _ => false,
        }
    }
}
///hashtable
pub struct Mambo<T: Copy> {
    data_arc: Arc<(f32, Box<[Shard<T>]>)>,
    how_edit_elem_without_update: Box<[i64]>,
    usize_smaller_u64: bool,
}

impl<T: Copy> Mambo<T> {
    ///  The num_shards table is divided into independent sections for optimization,
    ///  the optimal value is num_shards = the number of cores on your processor for multithreaded operation.
    ///  The redream_factor can be from 1.0 to 10.0, depending on how Mambo
    ///  is planned to be applied. ->10.0 when a lot of elements are planned (more than 10_000 ).
    ///  if there are few elements, about 1000 or less, it is better to use values tending to ->1.0.
    ///  At ->10, memory consumption per element decreases (depending on the system, at 1.0,
    ///  memory costs per element range from 32 to 64 bytes of overhead memory,
    ///  at ->10.0 per 1 element costs are reduced to ~10 bytes per element)
    pub fn new(num_shards: usize, redream_factor: f32) -> Result<Self, &'static str> {
        if !(1.0..11.0).contains(&redream_factor) {
            return Err("(1.0..11.0).contains( redream_factor)");
        }
        let shards = (0..num_shards)
            .map(|_| {
                let data: Box<[Vecta<T>]> = (0..PRIME_NUMBERS_TO_TAB[0])
                    .map(|_| Mutex::new((ONPLACE, vec![])))
                    .collect();

                (
                    Mutex::new(0_u32),
                    Mutex::new(0_usize),
                    RwLock::new([data, vec![].into_boxed_slice()]),
                )
            })
            .collect();
        Ok(Self {
            data_arc: Arc::new((redream_factor, shards)),
            how_edit_elem_without_update: vec![0; num_shards].into_boxed_slice(),
            usize_smaller_u64: (usize::MAX as u64) < (u64::MAX as u64),
        })
    }

    fn resize_shard(&self, shard_index: usize, new_len: usize) -> Result<bool, &'static str> {
        let shard = self
            .data_arc
            .1
            .get(shard_index)
            .ok_or("shard_index out of range")?;
        /*locker_resizer is needed for a global announcement that the size of the current shard is currently
        being edited. if locker_resizer is currently blocked, then other functions donot need to resize this shard.
        !NOTE!
        some compilers with aggressive optimization parameters may see that the data in locker_resizer
        is not being used and may remove the lock call to locker_resizer, which will lead to UB.
        therefore, garbage is written to locker_resizer, which does not affect the operation of the program.*/
        let mut locker_resizer = match shard.0.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => {
                return Ok(false);
            }
            Err(TryLockError::Poisoned(err)) => {
                panic!("Mutex poisoned: {:?}", err);
            }
        };
        *locker_resizer = locker_resizer.wrapping_add(1);

        let _ = {
            //non lock read
            /*
            checking that the vector number 0 has a length, which means that it is initialized,
            it must be of non-zero length. the vector number 1 has a length of 0. If it has a length of 0,
            it means that an error has occurred in the program or the resize_shard
            function is currently activated in another thread, which is not possible. In real work,
            this check should always end positively, and is needed to simplify
            the detection of errors during development.*/
            let v0_len = shard.2.read().unwrap()[0].len();
            let v1_len = shard.2.read().unwrap()[1].len();

            if v0_len > 0 && v1_len == 0 {
                v0_len
            } else if v1_len > 0 && v0_len > 0 {
                return Err("vec0 and vec1 > 0 ");
            } else {
                return Err("vec0 and vec1 == 0 ");
            }
        };

        {
            //a new vector is created, with a length of PRIME_NUMBERS_TO_TAB[new_index_in_prime]
            let new_vec_box: Box<[Vecta<T>]> = (0..new_len)
                .map(|_| Mutex::new((ONPLACE, vec![])))
                .collect();
            //write lock edits putting a new vector in the Rwlock
            let mut w_rlock = shard.2.write().unwrap();
            if w_rlock[1].len() == 0 {
                w_rlock[1] = new_vec_box;
            } else {
                return Err("vec 1 is resizes in this moment");
            }
            //println!("vec in rw 1len {}", w_rlock[0].len());
        }
        {
            //read non lock
            let r_lock = shard.2.read().unwrap();
            let v0_old = &r_lock[0];
            let v1_new = &r_lock[1];
            //iterating through the elements in the old vector and moving the elements to the new one
            //let mut elems_in_shard: usize = 0;
            for x in v0_old.iter() {
                let mut temp_old_mutexer = x.lock().unwrap();
                /*if the elements were moved before they were moved, an error is triggered.
                since the elements are moved only during copying, and if they have been moved and some other
                function is also moving them at the moment, this should not be possible,
                since locker_resizer must ensure that only one fn resize_shard is active at
                one time for the current shard.*/
                if temp_old_mutexer.0 == WASMOVED {
                    return Err(
                        "Unknown error element that should not be moved, was moved before moving",
                    );
                }
                temp_old_mutexer.0 = WASMOVED;

                for old_elem in temp_old_mutexer.1.iter() {
                    let mut new_mutex_elem =
                        v1_new[old_elem.1 as usize % v1_new.len()].lock().unwrap();

                    new_mutex_elem.1.push(*old_elem);
                    //elems_in_shard += 1;
                }
                //minimizing memory consumption, removing unused vectors
                if temp_old_mutexer.1.capacity() > 0 {
                    temp_old_mutexer.1 = Vec::with_capacity(0);
                }

                //trash changes
                *locker_resizer = locker_resizer.wrapping_add(temp_old_mutexer.1.len() as u32);
            }
        }
        {
            //write lock edits
            let mut w_lock = shard.2.write().unwrap();

            let old_1 = mem::replace(&mut w_lock[1], Vec::new().into_boxed_slice());
            w_lock[0] = old_1;
            #[cfg(test)]
            {
                //  println!("resize_shard {}  len {}", shard_index, w_lock[0].len())
            }
        }

        Ok(true)
    }

    #[inline(always)]
    fn search_vec<F>(&mut self, key: u64, operation: F) -> Result<(), &'static str>
    where
        F: FnOnce(&mut Vec<ElemBox<T>>) -> Result<WhatIdo, &'static str>,
    {
        let mut is_edit = false;
        let mut shard_capasity = 0;
        let shard_index = key % self.data_arc.1.len() as u64;
        let shard = &self.data_arc.1[shard_index as usize];

        {
            let r_lock = shard.2.read().unwrap();
            for x in 0..2 {
                let vecta = &r_lock[x & 0b1];

                if 0 != vecta.len() {
                    let mutexer = &mut vecta[key as usize % vecta.len()].lock().unwrap();
                    if ONPLACE == mutexer.0 {
                        shard_capasity = vecta.len();

                        match operation(&mut mutexer.1)? {
                            WhatIdo::REMOVE => {
                                self.how_edit_elem_without_update[shard_index as usize] =
                                    self.how_edit_elem_without_update[shard_index as usize]
                                        .checked_sub(1)
                                        .ok_or("how_edit_elem_without_update sub is owerflow")?;
                                is_edit = true;
                            }
                            WhatIdo::INSERT => {
                                self.how_edit_elem_without_update[shard_index as usize] =
                                    self.how_edit_elem_without_update[shard_index as usize]
                                        .checked_add(1)
                                        .ok_or("how_edit_elem_without_update add is owerflow")?;
                                is_edit = true;
                            }
                            _ => {
                                return Ok(());
                            }
                        };
                        break;
                    }
                }
            }
        } //unlock read r_lock
        if is_edit {
            // println!("shard_capasity: {}", shard_capasity);
            return self.is_resize_shard(shard_index as usize, shard_capasity, false);
        }

        Err("access error, constant employment")
    }
    fn is_resize_shard(
        &mut self,
        shard_index: usize,
        shard_capasity: usize,
        force_edit_global_counter: bool,
    ) -> Result<(), &'static str> {
        if let Some(new_len) = self.new_size_shard(
            shard_index as usize,
            shard_capasity,
            force_edit_global_counter,
        ) {
            if self.usize_smaller_u64 && (new_len > (usize::MAX as u64)) {
                return Err(
                    "if you see this error, it means that in your system usize::MAX < u64::MAX
                    and when increasing the size of the shard, a number from PRIME_NUMBERS_TO_TAB was requested that
                    is greater than usize::MAX. most likely, the usize on this device is 32 or 16 bits, which means that
                    you can put the orientation points in MamboMambo(redream_factor* (usize:: MAX/4))"
                    );
            }
            // println!("is_edit: {} {}", new_len, shard_capasity);

            self.resize_shard(shard_index, new_len as usize)?;
        }
        Ok(())
    }

    ///Resizing should only be done if the occupancy rate is higher than the redream_factor.
    /// if it is higher and these are not the last elements in PRIME_NUMBERS_TO_TAB,
    /// then its size becomes PRIME_NUMBERS_TO_TAB[+1],
    /// if the number of elements is so large that you can use PRIME_NUMBERS_TO_TAB[-1]
    ///  and there will still be room under SHRINK_THRESHOLD_FACTOR before exceeding redream_factor,
    ///  the capacity size decreases by PRIME_NUMBERS_TO_TAB[-1].
    fn new_size_shard(
        &mut self,
        shard_index: usize,
        shard_capasity: usize,
        force_edit_global_counter: bool,
    ) -> Option<u64> {
        let shard_i = &mut self.how_edit_elem_without_update[shard_index];

        if !force_edit_global_counter && shard_i.abs_diff(0) < ELEM_NUMS_TO_READ_IN_MUTEX {
            return None;
        }

        let elems_in_me = {
            let mut mutexer = self.data_arc.1[shard_index].1.lock().unwrap();

            if *shard_i > 0 {
                *mutexer = mutexer.checked_add(shard_i.abs_diff(0) as usize).unwrap();
            } else {
                *mutexer = mutexer.checked_sub(shard_i.abs_diff(0) as usize).unwrap();
            }
            *shard_i = 0;
            *mutexer
        };
        //println!("elems in me: {}", elems_in_me);
        let index_my_prime = self.difer_index(shard_capasity);

        if elems_in_me as f32 / (shard_capasity as f32 + EPSILON) > self.data_arc.0 {
            if index_my_prime + 1 < PRIME_NUMBERS_TO_TAB.len() {
                Some(PRIME_NUMBERS_TO_TAB[index_my_prime + 1])
            } else {
                None
            }
        } else if index_my_prime > 0 {
            let t = PRIME_NUMBERS_TO_TAB[index_my_prime - 1];
            if self.data_arc.0 > (elems_in_me as f32 * SHRINK_THRESHOLD_FACTOR) / t as f32 {
                Some(PRIME_NUMBERS_TO_TAB[index_my_prime - 1])
            } else {
                None
            }
        } else {
            None
        }
    }
    ///finds the index of the number closest to the target in PRIME_NUMBER_TO_TAB
    fn difer_index(&self, target: usize) -> usize {
        let target = target as u64;

        PRIME_NUMBERS_TO_TAB
            .iter()
            .enumerate()
            .min_by_key(|&(_, &prime)| prime.abs_diff(target))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }
    /// Some of the Mambo structure data is thread-independent,
    /// and each copy via arc_clone provides a convenient instance of the Mambo structure that can be used
    /// in a new thread. all data regarding the elements of the Mambo hash table can be obtained from any
    /// stream that has an instance of arc_clone.
    pub fn arc_clone(&self) -> Self {
        Self {
            data_arc: Arc::clone(&self.data_arc),
            how_edit_elem_without_update: vec![0; self.data_arc.1.len()].into_boxed_slice(),
            usize_smaller_u64: self.usize_smaller_u64,
        }
    }
    /// inserting an element insert(&mut self, key: usize, elem: T) requests a key in u64 format
    ///  and the element itself, if an element with such a key is already in the table,
    ///  the error Err("there is already an element") is returned,
    ///  if the flag already_is_err ==true,
    ///  then if the element with the equivalent If the key is already in the table,
    ///  the method will return Ok(), but the old element will remain in the table.
    pub fn insert(&mut self, key: u64, elem: T, already_is_err: bool) -> Result<(), &'static str> {
        self.search_vec(key, |vecta| {
            for x in vecta.iter() {
                //Searching for an element and an equivalent key
                if x.1 == key as u64 {
                    if already_is_err {
                        return Err("there is already an element");
                    } else {
                        //if there was no insertion due to the fact that the element is already in the table,
                        //and already_is_err is true, the operation is considered as a read.
                        return Ok(WhatIdo::READ);
                    }
                }
            }
            //
            vecta.push((elem, key as u64));

            Ok(WhatIdo::INSERT)
        }) //search_vec
    }
    /// to remove the fragment. you need to return the key key: u64
    /// if none_is_err == true and the element is not in the table, Ok(None) is returned,
    /// if none_is_err == false, Err("element not in map").
    /// if an element with the same key is in the table,
    ///  it is deleted from the table and returned as Ok(Some(T .clone()))
    pub fn remove(&mut self, key: u64, none_is_err: bool) -> Result<Option<T>, &'static str> {
        let mut ret_after_remove: Option<T> = None;

        self.search_vec(key, |vecta| {
            for x in 0..vecta.len() {
                if vecta[x].1 == key as u64 {
                    let last = *vecta.last().ok_or("ver item is empty")?;

                    ret_after_remove = Some(vecta[x].0.clone());
                    vecta[x] = last;
                    vecta.pop();
                    return Ok(WhatIdo::REMOVE);
                }
            }
            if none_is_err {
                Err("elem not in map")
            } else {
                Ok(WhatIdo::READ)
            }
        })?;
        Ok(ret_after_remove)
    }
    ///!Attention!! DO NOT CALL read INSIDE THE read CLOSURE!!! THIS MAY CAUSE THE THREAD TO SELF-LOCK!!
    /// -
    ///  reading. to access an item for reading, you need to request it using the key.
    ///  and the element is read and processed in the
    ///  RFy closure: FnOnce(&mut T) -> Result<(), &'static str>,
    ///  since &mut T is Mutex.lock(). while processing is taking place in read<RFy>,
    ///  the shard in which this element is located cannot:
    ///  1 change its size.
    ///  2: since Mutex contains elements from 0 to redream_factor
    ///  (a parameter in the Mamdo::new constructor),
    ///  access in other threads for elements in one Mutex is blocked.
    ///   to summarize, the faster the closure is resolved inside read, the better.
    pub fn read<RFy>(&mut self, key: u64, ridler: RFy) -> Result<(), &'static str>
    where
        RFy: FnOnce(&mut T) -> Result<(), &'static str>,
    {
        self.search_vec(key, |vecta| {
            for (t, hahs) in vecta.iter_mut() {
                if *hahs == key as u64 {
                    ridler(t)?;
                    return Ok(WhatIdo::READ);
                }
            }
            Err("element not in map")
        })
    }
    /// The number of elements in the hash table.
    ///  returns the number of all items in the table.note.
    ///  for optimization in multithreaded environments,
    ///
    /// !!!item count! changes do NOT occur EVERY TIME an item is DELETED/INSERTED.!!!
    ///  with a large number of elements and in a multithreaded environment,
    ///  it may not be critical, but when there are few elements, when elems_im_me is called,
    ///  it may return 0 even if there are elements in Mambo.
    ///  This is a forced decision to increase productivity.
    pub fn elems_im_me(&self) -> Result<usize, &'static str> {
        let mut elems = 0_usize;
        for x in self.data_arc.1.iter() {
            let t = x.1.lock().unwrap();
            elems = elems
                .checked_add(*t)
                .ok_or("err elems.checked_add(elems in shard )")?;
        }
        Ok(elems)
    }
}

impl<T: Copy> Drop for Mambo<T> {
    ///  deleting a Mambo instance. Since Mambo uses only secure rust APIs,
    ///  it does not need to implement the Drop trace, but since when inserting
    ///  or deleting an element, calls insert() or remove() do not change the global
    ///  element counter every time these methods are called so that the discrepancy
    ///  is minimal, each time an instance of Mambo is deleted, the value is forcibly
    ///  saved from a local variable. to the global internal Mutex
    fn drop(&mut self) {
        let mut ive = vec![0usize; 0];

        for (shard, shard_i) in self
            .data_arc
            .1
            .iter()
            .zip(self.how_edit_elem_without_update.iter())
        {
            let mut mutexer = shard.1.lock().unwrap();
            let shard_capasity = {
                let rwr = shard.2.read().unwrap();
                let len_r1 = rwr[1].len();
                if 0 != len_r1 {
                    len_r1
                } else {
                    rwr[0].len()
                }
            };

            if *shard_i > 0 {
                *mutexer = mutexer.checked_add(shard_i.abs_diff(0) as usize).unwrap();
                //println!("mux {}", shard_i.abs_diff(0));
            } else {
                *mutexer = mutexer.checked_sub(shard_i.abs_diff(0) as usize).unwrap();
            }
            //println!("len: {}", shard_capasity);
            ive.push(shard_capasity);
        }

        for (shard_index, &shard_capasity) in ive.iter().enumerate() {
            self.is_resize_shard(shard_index as usize, shard_capasity, true)
                .unwrap();
            //println!("drop{}", shard_index);
        }
    }
}

#[cfg(test)]
mod tests_n {

    use super::*;

    // use core::time;
    //use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    use std::u64;

    #[test]
    fn based_insert_test() {
        let shards = 5;
        let mut mambo = Mambo::<u32>::new(shards, 5.0).unwrap();
        println!("{}", mambo.data_arc.1.len());

        for x in 0..50_000 {
            if x % 10 == 9 {
                let elems = mambo.elems_im_me().unwrap();

                assert!(
                    elems.abs_diff(x) <= ELEM_NUMS_TO_READ_IN_MUTEX as usize * shards,
                    "elems.abs_diff(x):{} {}",
                    elems.abs_diff(x),
                    ELEM_NUMS_TO_READ_IN_MUTEX as usize * shards
                );

                //println!("{}", elems);
            }

            assert_eq!(mambo.insert(x as u64, x as u32 * 2 as u32, false), Ok(()));
        }

        for x in 0..50_000 {
            assert_eq!(
                mambo.read(x, |el| {
                    assert_eq!(*el, x as u32 * 2);
                    *el += 1;
                    Ok(())
                }),
                Ok(())
            );
        }

        for x in 0..50_000 {
            let inv_x = 50000 - x;
            let elems = mambo.elems_im_me().unwrap();
            assert!(
                elems.abs_diff(inv_x) <= ELEM_NUMS_TO_READ_IN_MUTEX as usize * shards,
                "elems.abs_diff(x):{} {}",
                elems.abs_diff(inv_x),
                ELEM_NUMS_TO_READ_IN_MUTEX as usize * shards
            );
            assert_eq!(mambo.remove(x as u64, true).is_ok(), true);
        }
        assert_eq!(mambo.remove(11, true), Err("elem not in map"));

        return;
    }

    #[test]
    fn based_threars_test() {
        {
            //return;
            const NUM_THREADS: usize = 10;
            const OPS_PER_THREAD: usize = 1000;
            const TEST_ELEMES: usize = 1000;
            let mambo = Mambo::<u64>::new(16, 5.0).unwrap();

            fn pair(u: usize) -> usize {
                let mut u: usize = u as usize;
                for m in 0..12 {
                    u ^= u.rotate_left(m * 1).wrapping_add(u.rotate_right(3 * m))
                        ^ PRIME_NUMBERS_TO_TAB[10 + m as usize] as usize;
                }
                u as usize
            }

            let mut std_handles = Vec::new();
            let std_barrier = Arc::new(std::sync::Barrier::new(NUM_THREADS + 1));

            for tt in 1..NUM_THREADS + 1 {
                let barrier_clone = Arc::clone(&std_barrier);

                let mut ra_clone = mambo.arc_clone();

                let handle = thread::spawn(move || {
                    barrier_clone.wait();

                    //println!("t: {}  np {:x}", tt, ptr::null_mut::<u8>() as usize);
                    //return;

                    if tt % 4 == 100 {
                        for _ in 0..OPS_PER_THREAD {
                            for o in 0..TEST_ELEMES {
                                let pai = pair(o);
                                ra_clone
                                .read(o as u64, |ind| {
                                    assert_eq!(
                                        *ind,
                                        pai as u64,
                                        " based_threars_test non eq read  index: {}   rea: {}   in map: {}",o,pai as u32,*ind
                                    );
                                    Ok(())
                                })
                                .unwrap();
                            }
                        }
                    } else {
                        let elem_read_write = pair(pair(tt)) % TEST_ELEMES;

                        let iterations = pair(pair(elem_read_write)) % OPS_PER_THREAD;

                        for _ in 0..iterations {
                            for yy in 0..elem_read_write {
                                let index = tt + (NUM_THREADS * 10_000_000 * yy);

                                // println!("in {}     el {}", index,);
                                ra_clone.insert(index as u64, yy as u64, false).unwrap();
                            }
                            // println!("+++++++++++++++==");
                            for yy in 0..elem_read_write {
                                //println!("{}", yy);
                                let index = tt + (NUM_THREADS * 10_000_000 * yy);
                                let ga = ra_clone.remove(index as u64, true);

                                if ga.is_err() && 200 > TEST_ELEMES {
                                    println!("index {}", index);

                                    if ra_clone.data_arc.1.len() == 1 {
                                        for inn in 0..(TEST_ELEMES as f32 * (1.0 / 0.6)) as usize {
                                            let _ = ra_clone.read(inn as u64, |xx| {
                                                print!("| {} ", xx);
                                                Ok(())
                                            });
                                            if inn % 4 == 3 {
                                                println!()
                                            }
                                        }
                                    } else {
                                        panic!("ra_clone.shards.len()==1else");
                                    }
                                }

                                /*if yy % 7 == 0 {
                                    let ddd = 10;
                                    ra_clone
                                        .resize_shard(
                                            &ra_clone.shards[yy % ra_clone.shards.len()],
                                            &ddd,
                                        )
                                        .unwrap();
                                }*/
                            }
                            //println!("{}", xx);
                        }
                    }
                });

                std_handles.push(handle);
            }

            std_barrier.wait();

            for handle in std_handles {
                handle.join().unwrap();
            }
        }
        // println!(
        //     "TEST_BREAK_RESIZE {}",
        //     TEST_BREAK_RESIZE.load(Ordering::Relaxed)
        // );
        // println!("TEST_RESIZE {}", TEST_RESIZE.load(Ordering::Relaxed));
        // println!("TEST_SWAP_MAX {}", TEST_SWAP_MAX.load(Ordering::Relaxed));
        // println!("create: {}", mem_me::TEST_CREATE.load(Ordering::Acquire));
        // println!("destroy: {}", mem_me::TEST_DESTROY.load(Ordering::Acquire));
        //assert!(false);
    }

    #[test]
    fn read_write() {
        for tre in (1..20).step_by(1) {
            //const NUM_THREADS: usize = 50;
            let num_treads: usize = tre;
            const NUM_ELEMS: usize = 80;
            const TOTAL_OPS: u64 = 500_000;
            let ops_threads: u64 = TOTAL_OPS / num_treads as u64;
            {
                let mut mambo = Mambo::<u64>::new(16, 10.0).unwrap();
                let mut std_handles = Vec::new();
                let std_start = Instant::now();
                let std_barrier = Arc::new(std::sync::Barrier::new(num_treads + 1));

                for i in 0..NUM_ELEMS {
                    mambo.insert(i as u64, 1, false).unwrap();
                }

                for _ in 0..num_treads {
                    let barrier_clone = Arc::clone(&std_barrier);

                    let mut ra_clone = mambo.arc_clone();

                    let handle = thread::spawn(move || {
                        barrier_clone.wait();

                        for i in 0..ops_threads {
                            let mut od: u64 = i as u64;
                            for _ in 0..3 {
                                od = od.rotate_left(43).wrapping_add(!i as u64);
                            }
                            let _ = ra_clone
                                .read(od % NUM_ELEMS as u64, |x| {
                                    *x += 1;
                                    Ok(())
                                })
                                .unwrap();
                        }
                    });

                    std_handles.push(handle);
                }

                std_barrier.wait();

                for handle in std_handles {
                    handle.join().unwrap();
                }
                if true {
                    println!(
                        "{}Mop/S: {:.3}",
                        "only read  ",
                        TOTAL_OPS as f64 / std_start.elapsed().as_micros() as f64
                    );
                } else {
                    println!(
                        "[{},  {:.3}]",
                        tre,
                        TOTAL_OPS as f64 / std_start.elapsed().as_micros() as f64
                    );
                }
            }
        }
        // assert!(false);
    }

    #[test]
    fn read_write_ers() {
        let mambo = Mambo::<u64>::new(1, 10.0).unwrap();
        let std_start = Instant::now();
        let elem_in_cycle = 10u64;
        let cycles = 200_00u64;
        for xx in (1..cycles).step_by(1) {
            let mut clom = mambo.arc_clone();
            //println!("{}", clom.elems_im_me().unwrap());
            for yy in 0..elem_in_cycle {
                clom.insert((yy * cycles * 100) + xx, 0, true).unwrap();
            }
        }

        println!(
            "{}Mop/S: {:.4}",
            "only read  ",
            (elem_in_cycle * cycles) as f64 / std_start.elapsed().as_micros() as f64
        );

        // assert!(false);
    }
}

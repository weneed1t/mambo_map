#![deny(clippy::indexing_slicing)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::as_conversions)]
#![deny(clippy::arithmetic_side_effects)]
#![deny(clippy::integer_division)]
//#![deny(clippy::expect_used)]
#![deny(clippy::unreachable)]
#![deny(clippy::todo)]
#![deny(clippy::float_cmp)]
#![forbid(unsafe_code)]

use std::mem;
use std::sync::{Arc, Mutex, RwLock, TryLockError};
//================================================================++
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
//================================================================--
const EELEM_NUMS_TO_READ_IN_MUTEX: u64 = 4;
const DEFAULT_NEW_SHARD_SIZE: usize = 2;
const SHRINK_THRESHOLD_FACTOR: f32 = 1.25;
const FORE_THRESHOLD_FACTOR: f32 = 0.10;
const EPSILON: f32 = 0.0001;
const WASMOVED: bool = false;
const ONPLACE: bool = true;

#[cfg(test)]
use std::sync::OnceLock;
#[cfg(test)]
static TEST_HOW_RESTORE_RW_L: OnceLock<Mutex<usize>> = OnceLock::new();
#[cfg(test)]
static TEST_HOW_RESTORE_MUTEX_ELEM: OnceLock<Mutex<usize>> = OnceLock::new();
#[cfg(test)]
static TEST_HOW_RESTORE_MUTEX_SIZE: OnceLock<Mutex<usize>> = OnceLock::new();

#[cfg(test)]
static TEST_HOW_CALL_RESIZE: OnceLock<Mutex<usize>> = OnceLock::new();

#[cfg(test)]
static TEST_HOW_BREAK_CALL_RESIZE: OnceLock<Mutex<usize>> = OnceLock::new();

#[cfg(test)]
fn test_how_restore_rw_l() -> &'static Mutex<usize> {
    TEST_HOW_RESTORE_RW_L.get_or_init(|| Mutex::new(0))
}

#[cfg(test)]
fn test_how_restore_mutex_elem() -> &'static Mutex<usize> {
    TEST_HOW_RESTORE_MUTEX_ELEM.get_or_init(|| Mutex::new(0))
}

#[cfg(test)]
fn test_how_restore_mutex_size() -> &'static Mutex<usize> {
    TEST_HOW_RESTORE_MUTEX_SIZE.get_or_init(|| Mutex::new(0))
}

#[cfg(test)]
fn test_how_call_resize() -> &'static Mutex<usize> {
    TEST_HOW_CALL_RESIZE.get_or_init(|| Mutex::new(0))
}

#[cfg(test)]
fn test_how_break_call_resize() -> &'static Mutex<usize> {
    TEST_HOW_BREAK_CALL_RESIZE.get_or_init(|| Mutex::new(0))
}

/// SAFE INTEGER CONVERSION MACRO
///
/// This macro safely converts between integer types using TryFrom,
/// preventing silent truncation or data loss.
///
/// STRATEGIES:
/// 1. Result: Returns Result<T, &'static str> for use with the '?' operator.
/// 2. Panic: Uses 'unwrap' or 'expect' for cases where failure is a bug.
/// 3. Custom Error: Allows attaching a specific message to the error.
///
/// ARGUMENTS:
/// - $x: The value to convert (e.g., ///key, u8::MAX).
/// - $t: The target type (e.g., usize, i64).
///
/// USAGE EXAMPLES:
///
/// // A. Handle errors with '?'
/// let idx = checked_cast!(key => usize)?;
/// let idx = checked_cast!(key => usize, err "Key out of range")?;
///
/// // B. Panic on failure
/// let val = checked_cast!(some_u64 => usize, unwrap);
/// let val = checked_cast!(some_u64 => usize, expect "Should fit");
///
/// // C. In closures or logic blocks
/// let is_valid = |p: u16| checked_cast!(p => usize, unwrap) < limit;
///
/// SAFETY:
/// Uses TryFrom trait. Prefer the Result variant (?) for recoverable errors.
/// Use expect/unwrap only when conversion is guaranteed by logic.
#[macro_export]
macro_rules! checked_cast {
    // No extra actions - returns Result with default error message
    ($x:expr => $t:ty) => {
        $crate::checked_cast!($x => $t, return_result)
    };

    // Panic with custom message (for unrecoverable errors)
    ($x:expr => $t:ty, expect $msg:expr) => {{
        //use std::convert::TryInto;
        <$t>::try_from($x).expect($msg)
    }};

    // Simple panic with default message (for quick prototyping)
    ($x:expr => $t:ty, unwrap) => {{
        use std::convert::TryInto;
        <$t>::try_from($x).unwrap()
    }};

    // Returns Result with custom error (for use with ? operator)
    ($x:expr => $t:ty, err $err:expr) => {{
        //use std::convert::TryInto;
        <$t>::try_from($x).map_err(|_| $err)
    }};

    // Returns Result with detailed default error message
    ($x:expr => $t:ty, return_result) => {{
        use std::convert::TryInto;
        <$t>::try_from($x).map_err(|_| concat!(
            "conversion failed: cannot cast `",
            stringify!($x),
            "` to `",
            stringify!($t),
            "`"
        ))
    }};
}

///ONPLACE and WASMOVED are tags that cannot be touched;
/// they are used instead of enums to mark nodes during program execution that
/// have been moved.<br>
const _: () = assert!(ONPLACE != WASMOVED, "ONPLACE == WASMOVED");
const PANIC_POISONED_IN_MUTEX_AND_RWLOCK: bool = false;
//================================================================++

type ElemBox<T> = (T, u64);
type Vecta<T> = Mutex<(bool, Vec<ElemBox<T>>)>;
type InRw<T> = (Mutex<usize>, /* elems in srard */ [Box<[Vecta<T>]>; 2]);
type Shard<T> = (
    Mutex<u32>, //locler os for resize
    RwLock<InRw<T>>,
);
//================================================================--
enum WhatIdo {
    Read,
    Remove,
    Insert,
}

impl PartialEq for WhatIdo {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (WhatIdo::Insert, WhatIdo::Insert)
                | (WhatIdo::Remove, WhatIdo::Remove)
                | (WhatIdo::Read, WhatIdo::Read)
        )
    }
}
///hashtable
pub struct Mambo<T: Clone> {
    data_arc: Arc<(f32, Box<[Shard<T>]>)>,
    how_edit_elem_without_update: Box<[i64]>,
}

impl<T: Clone> Mambo<T> {
    ///  The `num_shards` table is divided into independent sections for
    /// optimization,<br>  the optimal value is `num_shards` = the number of
    /// cores on your processor for multithreaded operation.<br>
    ///  The `redream_factor` can be from 1.0 to 10.0, depending on how
    /// Mambo<br>  is planned to be applied. ->10.0 when a lot of elements
    /// are planned (more than `10_000` ).<br>  if there are few elements,
    /// about 1000 or less, it is better to use values tending to ->1.0.<br>
    ///  At ->10, memory consumption per element decreases (depending on the
    /// system, at 1.0,<br>  memory costs per element range from 32 to 64
    /// bytes of overhead memory,<br>  at ->10.0 per 1 element costs are
    /// reduced to ~10 bytes per element)<br>
    pub fn new(num_shards: usize, redream_factor: f32) -> Result<Self, &'static str> {
        if !(1.0..11.0).contains(&redream_factor) {
            return Err("(1.0..11.0).contains( redream_factor)");
        }
        let shards = (0..num_shards)
            .map(|_| (Mutex::new(0_u32), RwLock::new(Self::new_into_shard_data())))
            .collect();
        Ok(Self {
            data_arc: Arc::new((redream_factor, shards)),
            how_edit_elem_without_update: vec![0; num_shards].into_boxed_slice(),
        })
    }
    /// secure access to the table element, if the mutex element is
    /// poisoned,<br> a new empty mutex is created.<br>
    fn safe_mutex_elem<'m>(
        mutexer: &'m Vecta<T>,
        i_non_save_edits: &mut i64,
    ) -> std::sync::MutexGuard<'m, (bool, Vec<ElemBox<T>>)> {
        match mutexer.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                #[cfg(test)]
                {
                    #![allow(clippy::arithmetic_side_effects)]
                    let mut t =
                        test_how_restore_mutex_elem().lock().expect("unreal poisoned in test");
                    *t += 1;
                }

                const { assert!(!PANIC_POISONED_IN_MUTEX_AND_RWLOCK, "safe_mutex_elem is poisoned") };
                let mut guard = poisoned.into_inner();

                *i_non_save_edits = i_non_save_edits.checked_sub(checked_cast!(guard.1.len() => i64, expect "guard.1.len() conversion to i64 failed")).unwrap_or(0);

                *guard = (ONPLACE, vec![]);
                mutexer.clear_poison();
                guard
            }
        }
    }
    ///uses the access method .`write()`<br>
    ///secure access to the rwlock , if the rwlock is poisoned,<br>
    ///a new empty rwlock is created with a standard long<br>
    fn save_rwlock_write(
        rwlock: &RwLock<InRw<T>>,
        forsed_repoisoned: bool,
    ) -> std::sync::RwLockWriteGuard<'_, InRw<T>> {
        match rwlock.write() {
            Ok(mut guard) => {
                if forsed_repoisoned {
                    *guard = Self::new_into_shard_data();
                }
                guard
            }
            Err(poisoned) => {
                #[cfg(test)]
                {
                    #![allow(clippy::arithmetic_side_effects)]
                    let mut t = test_how_restore_rw_l().lock().expect("unreal poisoned in test");
                    *t += 1;
                }
                const { assert!(!PANIC_POISONED_IN_MUTEX_AND_RWLOCK, "safe_mutex_elem is poisoned") };
                let mut guard = poisoned.into_inner();

                *guard = Self::new_into_shard_data();
                rwlock.clear_poison();
                guard
            }
        }
    }
    ///uses the .`rear()` access method,
    ///BUT IF the rwlock IS CORRUPTED, IT IS OVERWRITTEN USING THE .`write()`
    /// METHOD.<br> secure access to the rwlock, if the rwlock is
    /// poisoned,<br> a new empty rwlock is created with the standard
    /// long<br>
    fn save_rwlock_read(
        rwlock: &RwLock<InRw<T>>,
        forsed_repoisoned: bool,
    ) -> std::sync::RwLockReadGuard<'_, InRw<T>> {
        for _ in 0..10 {
            if let Ok(guard) = rwlock.read() {
                return guard;
            }

            let _t = Self::save_rwlock_write(rwlock, forsed_repoisoned);
        }
        panic!(
            "It was not possible to clean up the Rwlock poisoning in 10 iterations,
            which most likely means that this is an irreparable error or very deep damage.
            or the RwLock manages to get corrupted between for calls."
        );
    }
    //creates initial empty data for the Rwlock
    fn new_into_shard_data() -> InRw<T> {
        let data: Box<[Vecta<T>]> =
            (0..DEFAULT_NEW_SHARD_SIZE).map(|_| Mutex::new((ONPLACE, vec![]))).collect();
        (Mutex::new(0_usize), [data, vec![].into_boxed_slice()])
    }

    ///uses the .`rear()` access method,
    ///BUT IF the rwlock IS CORRUPTED, IT IS OVERWRITTEN USING THE .`write()`
    /// METHOD.<br> checking that the vector number 0 has a length, which
    /// means that it is initialized,<br> it must be of non-zero length. the
    /// vector number 1 has a length of 0. If it has a length of 0,<br>
    /// it means that an error has occurred in the program or the
    /// `resize_shard`<br> function is currently activated in another
    /// thread, which is not possible. In real work,<br> this check should
    /// always end positively, and is needed to simplify<br> the detection
    /// of errors during development.<br>
    fn lock_get_rw_cap(rwl: &RwLock<InRw<T>>) -> usize {
        let (v0_len, v1_len) = {
            let realder = Self::save_rwlock_read(rwl, false);
            (realder.1[0].len(), realder.1[1].len())
        };

        if v0_len > 0 && v1_len == 0 {
            v0_len
        } else if v1_len > 0 && v0_len > 0 {
            panic!(
                "
                vec0 and vec1 > 0 this is not possible because if vec0 and vec1 > 0,
                it means that several resize_shards are currently being executed, 
                or the previous resize_shard call ended with a fatal error"
            );
        } else {
            panic!(
                    "
                vec0 and vec1 == 0 this is an impossible state, most likely there was an error calculating
                the new length of the vec0 array, since vec1 should have a length of 0 since vec1 has a non-zero
                length only during resize_shard execution, if vec1 has a non-zero length, then either another
                resize_shard is executed in another thread or the previous resize_shard call has ended panic");
        }
    }

    ///locker_resizer and fileter is needed for a global announcement that the
    /// size of the current shard is currently being edited. if
    /// locker_resizer is currently blocked, then other functions donot need to
    /// resize this shard. NOTE!
    ///some compilers with aggressive optimization parameters may see that the
    /// data in locker_resizer is not being used and may remove the lock
    /// call to locker_resizer, which will lead to UB.
    //therefore, garbage is written to locker_resizer, which does not affect the
    // operation of the program.
    fn locker_resizer<'a>(&self, shard: &'a Shard<T>) -> Option<std::sync::MutexGuard<'a, u32>> {
        match shard.0.try_lock() {
            Ok(guard) => Some(guard),
            Err(TryLockError::WouldBlock) => None,
            Err(TryLockError::Poisoned(err)) => {
                #[cfg(test)]
                {
                    #![allow(clippy::arithmetic_side_effects)]
                    let mut t =
                        test_how_restore_mutex_size().lock().expect("unreal poisoned in test");
                    *t += 1;
                }

                if PANIC_POISONED_IN_MUTEX_AND_RWLOCK {
                    panic!("lock Mutex poisoned: {:?}", err);
                } else {
                    let mut guard = err.into_inner();
                    shard.0.clear_poison();
                    *guard = 0;

                    let _t = Self::save_rwlock_write(&shard.1, true);

                    Some(guard)
                }
            }
        }
    }

    fn create_temp_rw_shard(&self, new_len: usize, shard: &Shard<T>) {
        //a new vector is created, with a length of
        // PRIME_NUMBERS_TO_TAB[new_index_in_prime]
        let new_vec_box: Box<[Vecta<T>]> =
            (0..new_len).map(|_| Mutex::new((ONPLACE, vec![]))).collect();
        //write lock edits putting a new vector in the Rwlock
        let mut w_rlock = Self::save_rwlock_write(&shard.1, false);
        w_rlock.1[1] = new_vec_box;
    }

    fn swap_elems_from_old_srard_into_temp_shards(
        shard_index: usize,
        v1_new: &[Vecta<T>],
        v0_old: &[Vecta<T>],
        locker_resizer: &mut u32,
        how_edit_elem_without_update: &mut [i64],
    ) {
        //iterating through the elements in the old vector and moving the elements to
        // the new one let mut elems_in_shard: usize = 0;
        for x in v0_old {
            let mut temp_old_mutexer =
                    Self::safe_mutex_elem(x, how_edit_elem_without_update.get_mut(shard_index).expect("This is an unrealistic state since the indexes should not be larger than the array."));
            /*if the elements were moved before they were moved, an error is triggered.
            since the elements are moved only during copying, and if they have been moved and some other
            function is also moving them at the moment, this should not be possible,
            since locker_resizer must ensure that only one fn resize_shard is active at
            one time for the current shard.*/
            assert!(
                temp_old_mutexer.0 != WASMOVED,
                "
                    Unknown error element that should not be moved, was moved before moving,
                    then either another resize_shard is executed in another thread or the
                    previous resize_shard call has ended panic"
            );
            temp_old_mutexer.0 = WASMOVED;

            for old_elem in &temp_old_mutexer.1 {
                let diver = checked_cast!(old_elem.1 => usize, expect "old_elem.1 conversion to usize failed");
                let remainder =
                    diver.checked_rem(v1_new.len()).expect("division by zero: v1_new.len() == 0");
                let v1_item = v1_new
                    .get(remainder)
                    .expect("v1_new access failed: index out of bounds during rehashing");

                let edit_elem = how_edit_elem_without_update.get_mut(shard_index).expect(
                    "shard_index access failed: how_edit_elem_without_update index out of bounds",
                );

                let mut new_mutex_elem = Self::safe_mutex_elem(v1_item, edit_elem);

                new_mutex_elem.1.push(old_elem.clone());
            }
            //minimizing memory consumption, removing unused vectors
            if temp_old_mutexer.1.capacity() > 0 {
                temp_old_mutexer.1 = Vec::with_capacity(0);
            }

            //trash changes
            *locker_resizer = locker_resizer.wrapping_add(checked_cast!(temp_old_mutexer.1.len() => u32, expect "temp_old_mutexer.1.len() conversion to u32 failed"));
        }
    }

    /// errors during shard resizing are fatal. and in most cases,<br>
    /// it is impossible to restore data integrity<br>
    /// in the shard if an error occurred during self.resize_shard.<br>
    fn resize_shard(&mut self, shard_index: usize, new_len: usize) -> bool {
        let shard = self.data_arc.1.get(shard_index).expect(
            "This is an unrealistic state since the indexes should not be larger than the array.",
        );

        #[cfg(test)]
        {
            #![allow(clippy::arithmetic_side_effects)]
            let mut t = test_how_call_resize().lock().expect("unreal poisoned in test");
            *t += 1;
        }

        let mut locker_resizer = if let Some(lock_ret) = self.locker_resizer(shard) {
            lock_ret
        } else {
            #[cfg(test)]
            {
                #![allow(clippy::arithmetic_side_effects)]
                let mut t = test_how_break_call_resize().lock().expect("unreal poisoned in test");
                *t += 1;
            }
            return false;
        };

        *locker_resizer = locker_resizer.wrapping_add(1);

        {
            let _: usize = Self::lock_get_rw_cap(&shard.1);
        }; //drop

        self.create_temp_rw_shard(new_len, shard);
        {
            //read non lock
            let r_lock = Self::save_rwlock_read(&shard.1, false);

            let v0_old = &r_lock.1[0];
            let v1_new = &r_lock.1[1];
            //iterating through the elements in the old vector and moving the elements to
            // the new one let mut elems_in_shard: usize = 0;

            Self::swap_elems_from_old_srard_into_temp_shards(
                shard_index,
                v1_new,
                v0_old,
                &mut locker_resizer,
                &mut self.how_edit_elem_without_update,
            );
        }

        {
            //write lock edits
            let mut w_lock = Self::save_rwlock_write(&shard.1, false);

            w_lock.1[0] = mem::replace(&mut w_lock.1[1], Vec::new().into_boxed_slice());
        }

        true
    }
    ///`search_vec` searches for a Mutex element in one of the 2 vectors in the
    /// `RWlock`.<br> because the operation of changing the length of the
    /// shard size (`resize_shard`) is not blocking.<br> if `resize_shard`
    /// is executed in another thread during the `search_vec` call,<br>
    /// the Mutex target element can be located either in the old vec (index 0 )
    /// or in the temp vec(index 1 )<br>
    fn search_vec<F>(&mut self, key: u64, operation: F)
    where
        F: FnOnce(&mut Vec<ElemBox<T>>) -> WhatIdo,
    {
        let mut shard_capasity = 0;
        let divisor = checked_cast!(self.data_arc.1.len() => u64, expect "data_arc.1.len() conversion to u64 failed");

        let shard_index = key.checked_rem(divisor).expect("division by zero: divisor is zero");
        let shard = self.data_arc.1.get(checked_cast!(shard_index => usize, expect "shard_index conversion to usize failed")).expect(
    "data_arc.1 access failed: shard_index is out of bounds for the current shard array",
);

        let mut is_edit = false;
        {
            let r_lock = Self::save_rwlock_read(&shard.1, false);
            //the target Mutex element can be located either in the old vec (index 0 ) or
            // in the temp vec(index 1 ), if the element with the desired index
            // in the 0 vector has the attribute ONPLACE == mutexer.0,
            // it means it was moved to the vector with index 1.
            for x in 0..2 {
                let vecta = r_lock.1
    .get(x & 0b1)
    .expect("r_lock.1 access failed: bitwise index x & 0b1 is out of bounds (expected array size of at least 2)");

                if !vecta.is_empty() {
                    let shard_idx = checked_cast!(shard_index => usize, expect "shard_index conversion to usize failed");

                    // 1. Get the initial value safely
                    let temp_how_edit = *self.how_edit_elem_without_update
    .get(shard_idx)
    .expect("how_edit_elem_without_update access failed: shard_idx is out of bounds during pre-check");

                    // 2. Safely obtain both the vector element and the mutable counter

                    let key_usize =
                        checked_cast!(key => usize, expect "key conversion to usize failed");
                    let index = key_usize
                        .checked_rem(vecta.len())
                        .expect("division by zero: vecta.len() == 0");

                    let mut mutexer = Self::safe_mutex_elem(
    vecta.get(index)
        .expect("vecta access failed: calculated index is out of bounds"),
    self.how_edit_elem_without_update
        .get_mut(shard_idx)
        .expect("how_edit_elem_without_update access failed: shard_idx is out of bounds for mutable update"),
);

                    //if the counter was changed in safe_mutex_elem,
                    // it means that some elements were deleted, this is equivalent to the deletion
                    // operation, so is_edit = true; since counter
                    // synchronization may be required.
                    if temp_how_edit != *self.how_edit_elem_without_update
    .get(checked_cast!(shard_index => usize, expect "shard_index conversion to usize failed"))
    .expect("how_edit_elem_without_update access failed: shard_index is out of bounds during final edit check") 
{
    is_edit = true;
}

                    if ONPLACE == mutexer.0 {
                        shard_capasity = vecta.len();

                        match operation(&mut mutexer.1) {
                            //the types of operations are read write and delete,
                            //if a write and or delete is performed, then the local change counter
                            // changes
                            WhatIdo::Remove => {
                                let counter = self.how_edit_elem_without_update
        .get_mut(checked_cast!(shard_index => usize, expect "shard_index conversion to usize failed"))
        .expect("how_edit_elem_without_update access failed: shard_index is out of bounds during Remove operation");

                                *counter = counter
        .checked_sub(1)
        .expect("how_edit_elem_without_update sub is overflow (attempted to subtract from zero)");

                                is_edit = true;
                                break;
                            }
                            WhatIdo::Insert => {
                                let counter = self.how_edit_elem_without_update
        .get_mut(checked_cast!(shard_index => usize, expect "shard_index conversion to usize failed"))
        .expect("how_edit_elem_without_update access failed: shard_index is out of bounds during Insert operation");

                                *counter = counter
                                    .checked_add(1)
                                    .expect("how_edit_elem_without_update add is overflow");

                                is_edit = true;
                                break;
                            }

                            //if there was a read, the counter is not updated.
                            _ => {
                                if !is_edit {
                                    return;
                                }
                                break;
                            }
                        };
                    }
                }
            }
        } //unlock read r_lock
        if is_edit {
            self.is_resize_shard(checked_cast!(shard_index => usize, expect "shard_index conversion to usize failed"), shard_capasity, false);
            return;
        }
        panic!("access error, constant employment");
    }
    fn is_resize_shard(
        &mut self,
        shard_index: usize,
        shard_capasity: usize,
        force_edit_global_counter: bool,
    ) {
        if let Some(new_len) =
            self.new_size_shard(shard_index, shard_capasity, force_edit_global_counter)
        {
            assert!(usize::try_from(new_len).is_ok(), "if you see this error, it means that in your system usize::MAX < u64::MAX
            and when increasing the size of the shard, a number from PRIME_NUMBERS_TO_TAB was requested that
            is greater than usize::MAX. most likely, the usize on this device is 32 or 16 bits, which means that
            you can put the orientation points in MamboMambo(redream_factor* (usize:: MAX/4))");

            self.resize_shard(
                shard_index,
                checked_cast!(new_len => usize, expect "new_len conversion to usize failed"),
            );
        }
    }

    ///Resizing should only be done if the occupancy rate is higher than the
    /// `redream_factor`.<br> if it is higher and these are not the last
    /// elements in `PRIME_NUMBERS_TO_TAB`,<br> then its size becomes
    /// `PRIME_NUMBERS_TO_TAB`[+1],<br> if the number of elements is so
    /// large that you can use `PRIME_NUMBERS_TO_TAB`[-1]<br> and there will
    /// still be room under `SHRINK_THRESHOLD_FACTOR` before exceeding
    /// `redream_factor`,<br> the capacity size decreases by
    /// `PRIME_NUMBERS_TO_TAB`[-1].<br> Determines if and how a shard should
    /// be resized based on current load and configuration.<br> <br>
    /// The method evaluates multiple factors to decide whether to:<br>
    /// - Upsize: when occupancy exceeds the configured threshold and larger
    ///   prime sizes are available<br>
    /// - Downsize: when occupancy is low enough to save memory without
    ///   impacting performance<br>
    /// - Maintain current size: when the shard is optimally sized for its
    ///   current load<br>
    ///
    /// The resize decision uses prime numbers for table sizes to improve hash
    /// distribution<br> and avoid common modulus patterns that can cause
    /// clustering.<br>
    #[allow(clippy::absurd_extreme_comparisons)] //if EELEM_NUMS_TO_READ_IN_MUTEX == 0
    fn new_size_shard(
        &mut self,
        shard_index: usize,
        shard_capasity: usize,
        force_edit_global_counter: bool,
    ) -> Option<u64> {
        // Local counter tracking uncommitted operations (inserts/deletes) since last
        // resize check
        let shard_i = self.how_edit_elem_without_update
    .get_mut(shard_index)
    .expect("how_edit_elem_without_update access failed: shard_index is out of bounds while creating a mutable shard reference");

        // Early exit optimization: avoid expensive operations when conditions don't
        // warrant resize This prevents frequent resize operations for small,
        // low-concurrency workloads

        #[allow(clippy::as_conversions)]
        const ENTRIM_F32: f32 = EELEM_NUMS_TO_READ_IN_MUTEX as f32;
        #[allow(clippy::as_conversions)]
        let shard_capasity_f32: f32 = shard_capasity as f32;
        #[allow(clippy::as_conversions)]
        let arc_strong_ctr_f32: f32 = Arc::strong_count(&self.data_arc) as f32;

        if !force_edit_global_counter
            && shard_i.abs_diff(0) < EELEM_NUMS_TO_READ_IN_MUTEX
            && (arc_strong_ctr_f32 * ENTRIM_F32) < (shard_capasity_f32 * FORE_THRESHOLD_FACTOR)
        {
            return None;
        }

        // Atomically update and retrieve the actual element count from shared state
        // This synchronizes the local operation counter with the global shard counter
        let elems_in_me = {
            let rwler = Self::save_rwlock_read(
    &self.data_arc.1
        .get(shard_index)
        .expect("data_arc.1 access failed: shard_index is out of bounds during read lock acquisition")
        .1,     false);

            let mut mutexer = rwler.0.lock().unwrap_or_else(|_| {
                panic!("Fatal corruption of the hash function, data cannot be recovered Mutex<usize> in Rwlock in shard: {} is poisoned", shard_index)
            });
            // Apply pending operations to the global counter
            // Positive values indicate net inserts, negative indicate net deletes
            if *shard_i > 0 {
                *mutexer = mutexer
        .checked_add(checked_cast!(shard_i.abs_diff(0) => usize, expect "shard_i.abs_diff(0) conversion to usize failed"))
        .expect("new_size_shard add is overflow");
            } else {
                *mutexer = mutexer.checked_sub(checked_cast!(shard_i.abs_diff(0) => usize, expect "shard_i.abs_diff(0) conversion to usize failed")).unwrap_or(0);
            }
            // Reset local counter now that changes are committed to global state
            *shard_i = 0;
            // Return the updated element count for resize decision making
            *mutexer
        };
        // Find the index of the current capacity in our prime number table
        let index_my_prime = self.difer_index(shard_capasity);
        // Decision: Upsize if current load factor exceeds configured threshold
        // Load factor = elements / capacity, we add EPSILON to avoid division by zero
        #[allow(clippy::as_conversions)]
        let ele_inme_f32 = elems_in_me as f32;
        #[allow(clippy::as_conversions)]
        let shard_cap_f32 = shard_capasity as f32;

        if ele_inme_f32 / (shard_cap_f32 + EPSILON) > self.data_arc.0 {
            let new_index = index_my_prime.checked_add(1).expect("overflow in index_my_prime + 1");

            // Only upsize if we have a larger prime available in our table
            if new_index < PRIME_NUMBERS_TO_TAB.len() {
                Some(
        *PRIME_NUMBERS_TO_TAB
            .get(new_index)
            .expect("PRIME_NUMBERS_TO_TAB access failed: index is out of bounds despite passing the length check")
)
            } else {
                None
            }
        } else if index_my_prime > 0 {
            // Decision: Consider downsizing if we're not at the smallest prime size
            let new_index =
                index_my_prime.checked_sub(1).expect("subtraction underflow: index_my_prime < 1");
            let t = *PRIME_NUMBERS_TO_TAB
                    .get(new_index)
                    .expect("PRIME_NUMBERS_TO_TAB access failed: index_my_prime is either 0 or exceeds the table bounds");

            // Downsizing heuristic: Only shrink if the new load factor would be comfortably
            // below the threshold even after applying a shrink safety margin
            // This prevents thrashing between sizes for borderline cases
            #[allow(clippy::as_conversions)]
            let t_32 = t as f32;
            if self.data_arc.0 > (ele_inme_f32 * SHRINK_THRESHOLD_FACTOR) / t_32 {
                Some(t)
            } else {
                None
            }
        } else {
            None
        }
    }
    /// finds the index of the number closest to the target in
    /// `PRIME_NUMBER_TO_TAB`
    fn difer_index(&self, target: usize) -> usize {
        let target = checked_cast!(target => u64, expect "target conversion to u64 failed");

        PRIME_NUMBERS_TO_TAB
            .iter()
            .enumerate()
            .min_by_key(|&(_, &prime)| prime.abs_diff(target))
            .map_or(0, |(index, _)| index)
    }

    /// insert an element T with the key: u64.if there is already an element
    /// with the same key: u64 in the table,<br> then when `force_replace`
    /// == true, the old element T will be replaced by the new element T,<br>
    /// while the old element T will be returned as Some(T.clone()). if
    /// `force_replace` == false,<br> the old element will not be replaced
    /// and the function will returned old elem as Some(T.clone()).<br>
    /// if there is no element with this key: u64,<br>
    /// a new element will be added to the table and the function will output
    /// None<br>
    pub fn insert(&mut self, key: u64, elem: &T, force_replace: bool) -> Option<T> {
        let mut returned_elem: Option<T> = None;
        self.search_vec(key, |vecta| {
            for x in vecta.iter_mut() {
                //Searching for an element and an equivalent key
                if x.1 == key {
                    //if there was no insertion due to the fact that the element is already in the
                    // table,
                    returned_elem = Some(x.0.clone());
                    if force_replace {
                        *x = (elem.clone(), key);
                    }
                    //the operation is considered as a read.
                    return WhatIdo::Read;
                }
            }
            vecta.push((elem.clone(), key));

            WhatIdo::Insert
        }); //search_vec

        returned_elem
    }
    /// to remove the fragment. you need to return the key key: u64<br>
    /// if an element was in the table and it was deleted, but the function
    /// returns Some(T.clone())<br>
    // if there was no such element in the hash table, None will be returned<br>
    pub fn remove(&mut self, key: u64) -> Option<T> {
        let mut ret_after_remove: Option<T> = None;

        self.search_vec(key, |vecta| {
            for x in 0..vecta.len() {
                // 1. Safely check the key using .get()
                if vecta
                    .get(x)
                    .expect("vecta access failed: index x out of bounds during key search")
                    .1
                    == key
                {
                    if let Some(last) = vecta.last() {
                        let last_cloned = last.clone();

                        // 2. Access the target element mutably to perform the swap
                        let target = vecta.get_mut(x).expect(
                            "vecta access failed: index x out of bounds during element swap",
                        );

                        ret_after_remove = Some(target.0.clone());

                        // Replace the element at current index with the cloned last element
                        *target = last_cloned;

                        // 3. Remove the original last element
                        vecta.pop();

                        return WhatIdo::Remove;
                    }
                    return WhatIdo::Read;
                }
            }

            WhatIdo::Read
        });
        ret_after_remove
    }

    ///-!Attention!! DO NOT CALL read INSIDE THE read CLOSURE!!! THIS MAY CAUSE
    /// THE THRead TO SELF-LOCK!! -
    ///  reading. to access an item for reading, you need to request it using
    /// the key.<br>  and the element is read and processed in the<br>
    ///  RFy closure:FnOnce(Option<&mut T>),<br>
    ///  since &mut T is Mutex.lock(). while processing is taking place in
    /// read<RFy>,<br>  the shard in which this element is located
    /// cannot:<br>  1 change its size.<br>
    ///  2: since Mutex contains elements from 0 to redream_factor<br>
    ///  (a parameter in the Mamdo::new constructor),<br>
    ///  access in other threads for elements in one Mutex is blocked.<br>
    ///   to summarize, the faster the closure is resolved inside read, the
    /// better.<br>
    pub fn read<RFy>(&mut self, key: u64, ridler: RFy)
    where
        RFy: FnOnce(Option<&mut T>),
    {
        self.search_vec(key, |vecta| {
            for (t, hahs) in vecta.iter_mut() {
                if *hahs == key {
                    ridler(Some(t));
                    return WhatIdo::Read;
                }
            }
            ridler(None);
            WhatIdo::Read
        });
    }
    /// The number of elements in the hash table.<br>
    ///  returns the number of all items in the table.note.<br>
    ///  for optimization in multithreaded environments,<br>
    ///
    /// !!!item count! changes do NOT occur EVERY TIME an item is
    /// DELETED/InsertED.!!!<br>  with a large number of elements and in a
    /// multithreaded environment,<br>  it may not be critical, but when
    /// there are few elements, when `elems_im_me` is called,<br>
    ///  it may return 0 even if there are elements in Mambo.<br>
    ///  This is a forced decision to increase productivity.<br>
    #[must_use]
    pub fn elems_im_me(&self) -> usize {
        let mut elems = 0_usize;
        for (shard_index, x) in self.data_arc.1.iter().enumerate() {
            let x = Self::save_rwlock_read(&x.1, false);
            let t =x.0.lock().unwrap_or_else(|_| panic!("poisoning the Mutex<usize> that stores the number of items in new_size_shard in the shard : {}",shard_index));
            elems = elems.checked_add(*t).unwrap_or_else(|| {
                panic!("err elems.checked_add(elems in shard {} )", shard_index)
            });
        }
        elems
    }
    ///`filter()` takes the closure `RFy`: FnMut(&mut T, u64) -> bool<br>
    /// This closure is applied sequentially to all elements of Mambo. <br>
    /// If FnMut(&mut T, u64) -> bool returns false for a specific element,<br>
    ///  that element is deleted; if it returns true, that element is kept.<br>
    /// While the filter is running,<br>
    ///  it is not possible to apply functions that change the size of the shard
    /// blocked by the filter.<br>
    pub fn filter<RFy>(&mut self, mut filtator: RFy)
    where
        RFy: FnMut(&mut T, u64) -> bool,
    {
        for (i, shard) in self.data_arc.1.iter().enumerate() {
            /*locker_resizer and fileter is needed for a global announcement that the size of the current shard is currently
            being edited. if locker_resizer is currently blocked, then other functions donot need to resize this shard.
            !NOTE!
            some compilers with aggressive optimization parameters may see that the data in locker_resizer
            is not being used and may remove the lock call to locker_resizer, which will lead to UB.
            therefore, garbage is written to locker_resizer, which does not affect the operation of the program.*/
            let mut locker_resizer = match shard.0.lock() {
                Ok(guard) => guard,
                Err(err) => {
                    if PANIC_POISONED_IN_MUTEX_AND_RWLOCK {
                        panic!("lock Mutex poisoned: {:?}", err);
                    } else {
                        let mut guard = err.into_inner();
                        shard.0.clear_poison();
                        *guard = 0;

                        let _t = Self::save_rwlock_write(&shard.1, true);
                        *self.how_edit_elem_without_update
    .get_mut(i)
    .expect("how_edit_elem_without_update access failed: index i is out of bounds while resetting counter to zero") = 0;

                        continue;
                    }
                }
            };
            *locker_resizer = locker_resizer.wrapping_add(1);

            let mut rw_w = Self::save_rwlock_write(&shard.1, false);

            /*  vec0 and vec1 > 0 this is not possible because if vec0 and vec1 > 0,
            it means that several resize_shards are currently being executed,
            or the previous resize_shard call ended with a fatal error*/
            if !rw_w.1[1].is_empty() {
                *rw_w = Self::new_into_shard_data();
                break;
            }

            let mut elems_in_shard: usize = 0;

            for mux in &rw_w.1[0] {
                let mut mutexer =
                    Self::safe_mutex_elem(
    mux,   self.how_edit_elem_without_update
        .get_mut(i)
        .expect("how_edit_elem_without_update access failed: index i is out of bounds when calling safe_mutex_elem")
);

                assert!(
                    WASMOVED != mutexer.0,
                    "
                    Unknown error element that should not be moved,
                    then either another resize_shard is executed in another thread or the
                    previous resize_shard call has ended panic"
                );

                let mut tepma_el = Vec::<ElemBox<T>>::new();

                for elk in &mut mutexer.1 {
                    if filtator(&mut elk.0, elk.1) {
                        tepma_el.push(elk.clone());

                        elems_in_shard = elems_in_shard
                            .checked_add(1)
                            .expect("overflow in elems_in_shard increment");
                    }
                } //for in mutex
                mutexer.0 = ONPLACE;
                mutexer.1 = tepma_el;
            } //for in rw

            rw_w.0 = Mutex::new(elems_in_shard);
            let edit_counter = self.how_edit_elem_without_update
                .get_mut(i)
                .expect("how_edit_elem_without_update access failed: index out of bounds during mutex poison recovery");

            *edit_counter = 0;

            /*
            Record garbage so that the compiler does not remove the locker_resizer lock during optimization.

            Note: I have had negative experiences with MSVC compiler optimization for C++,
            which could consider some code unnecessary and remove it, resulting in hard-to-detect bugs.
            Most likely, the Rust compiler will not do such nasty things, but at the moment I am playing it safe,
            since the damage to performance from these operations is less than the margin of error.
            */
            *locker_resizer = locker_resizer.wrapping_add(checked_cast!(elems_in_shard => u32, expect "elems_in_shard conversion to u32 failed"));
        }
        //for shard
    }
}

impl<T: Clone> Clone for Mambo<T> {
    /// Some of the Mambo structure data is thread-independent,<br>
    /// and each copy via `arc_clone` provides a convenient instance of the
    /// Mambo structure that can be used<br> in a new thread. all data
    /// regarding the elements of the Mambo hash table can be obtained from
    /// any<br> stream that has an instance of `arc_clone`.<br>
    fn clone(&self) -> Self {
        Self {
            data_arc: Arc::clone(&self.data_arc),
            how_edit_elem_without_update: vec![0; self.data_arc.1.len()].into_boxed_slice(),
        }
    }
}

impl<T: Clone> Drop for Mambo<T> {
    ///  deleting a Mambo instance. Since Mambo uses only secure rust APIs,<br>
    ///  it does not need to implement the Drop trace, but since when
    /// inserting<br>  or deleting an element, calls `insert()` or
    /// `remove()` do not change the global<br>  element counter every time
    /// these methods are called so that the discrepancy<br>  is minimal,
    /// each time an instance of Mambo is deleted, the value is forcibly<br>
    ///  saved from a local variable. to the global internal Mutex<br>
    fn drop(&mut self) {
        let mut ive = vec![0usize; 0];

        for shard in &self.data_arc.1 {
            let shard_capasity = {
                let rwr = Self::save_rwlock_read(&shard.1, false);
                let len_r1 = rwr.1[1].len();
                if 0 == len_r1 { rwr.1[0].len() } else { len_r1 }
            };
            ive.push(shard_capasity);
        }
    }
}

#[cfg(test)]
mod tests_n {
    #![allow(clippy::as_conversions)]
    #![allow(clippy::indexing_slicing)]
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::arithmetic_side_effects)]
    #![deny(clippy::integer_division)]
    //#![deny(clippy::expect_used)]
    #![allow(clippy::unreachable)]
    #![allow(clippy::todo)]
    #![allow(clippy::float_cmp)]
    #![allow(clippy::integer_division)]
    use std::time::Instant;
    use std::{panic, thread, time};

    use super::*;

    #[test]
    fn based_insert_test() {
        let shards = 2;
        let mut mambo = Mambo::<u32>::new(shards, 5.0).unwrap();
        println!("{}", mambo.data_arc.1.len());

        for x in 0..30_000 {
            if x % 10 == 9 {
                let elems = mambo.elems_im_me();

                assert!(
                    elems.abs_diff(x) <= EELEM_NUMS_TO_READ_IN_MUTEX as usize * shards,
                    "elems.abs_diff(x):{} {}",
                    elems.abs_diff(x),
                    EELEM_NUMS_TO_READ_IN_MUTEX as usize * shards
                );

                //println!("{}", elems);
            }
            let x = x as u32;
            let y = x + 10;

            assert_eq!(mambo.insert(x as u64, &x, false), None);
            assert_eq!(mambo.insert(x as u64, &x, true), Some(x));
            assert_eq!(mambo.insert(x as u64, &x, false), Some(x));

            assert_eq!(mambo.insert(x as u64, &y, false), Some(x));
            assert_eq!(mambo.insert(x as u64, &y, true), Some(x));
            assert_eq!(mambo.insert(x as u64, &y, false), Some(y));

            assert_eq!(mambo.remove(x as u64), Some(y));
            assert_eq!(mambo.insert(x as u64, &x, false), None);
        }

        for x in 0..30_000 {
            mambo.read(x, |el| {
                let el = el.unwrap();

                assert_eq!(*el, x as u32);
                *el += 1;
            });

            mambo.read(x + 1_000_000, |el| {
                assert_eq!(el, None);
            });
        }

        for x in 0..30_000 {
            let inv_x = 30_000 - x;
            let elems = mambo.elems_im_me();
            assert!(
                elems.abs_diff(inv_x) <= EELEM_NUMS_TO_READ_IN_MUTEX as usize * shards,
                "elems.abs_diff(x):{} {}",
                elems.abs_diff(inv_x),
                EELEM_NUMS_TO_READ_IN_MUTEX as usize * shards
            );
            let x = x as u64;
            //println!("{:?}  {}", mambo.remove(x as u64), x);
            assert_eq!(mambo.remove(x), Some(x as u32 + 1));
        }
        print_restores()
    }

    #[test]
    fn based_example() {
        const NUM_THREAS: usize = 10;
        const OPS_PER_THREED: usize = 10;
        let mut std_handles = Vec::new();
        let shift = 1_000_000u64;
        let global_counter = Arc::new(Mutex::new(0usize));
        //=======================================================================
        //The number of shards, a shard is an independent piece of the hash table,
        //the optimal number of shards = the number of threads.
        let shards = 16;
        /*elements that are added to the hash table
        A hash table has a data storage topology
        Arc<(f32, Box<[(Mutex<u32>, RwLock<(Mutex<usize>, [Box<[Mutex<(bool, Vec<(T, u64)>)>]>; 2])>)]>)>.
        Each individual Mutex can have an average of 1.0 to 10.0 elements.
        the more elements there are in the Mutex, the lower the overhead of storing in memory per element,
        but the higher the chance that one thread will block another when reading one element.*/
        let elems_in_mutex = 7.0;
        //returns Err(&str) if the data is incorrect
        let mut mambo = Mambo::<String>::new(shards, elems_in_mutex).unwrap();

        for tt in 1..NUM_THREAS {
            let arc_global_counter = Arc::clone(&global_counter);
            //
            //
            /*mambo.clone().clone() because the mambo instance has local data that is individual
            for each instance and global data that is stored in Arc() and shared by all threads.
            to simplify the creation of a new instance, mambo.clone() is used.*/
            let mut mambo_arc = mambo.clone(); //Trait Clone

            std_handles.push(thread::spawn(move || {
                for key in 0..OPS_PER_THREED {
                    let key = (tt + (key * OPS_PER_THREED * 10)) + shift as usize;

                    let elem_me = format!("mambo elem {}", key);

                    /*to insert an element, if an element with the same key value already exists,
                    it will return Some(T.clone())
                    false indicates whether it is necessary to forcibly replace the element with a new one,
                    even if there is already an element with such a key*/
                    assert_eq!(mambo_arc.insert(key as u64, &elem_me, false), None);

                    assert_eq!(
                        mambo_arc.insert(key as u64, &elem_me, false),
                        Some(elem_me.clone())
                    );
                    /*reading, because the Mutex is kept open during reading,
                    as long as there is an active read operation in the shard,
                    a resizing operation and a filter operation cannot be applied to the shard.*/
                    //NEVER CALL A RECURSIVE Read INSIDE A read() CLOSURE, AS THIS MAY LEAD TO
                    // MUTUAL LOCKING!!
                    mambo_arc.read(key as u64, |ind| {
                        let ind = ind.unwrap();

                        assert_eq!(
                            ind.clone(),
                            elem_me,
                            " non eq read  key: {}   rea: {}   in map: {}",
                            key,
                            elem_me,
                            ind.clone()
                        );
                    });

                    let key_to_filter = {
                        let mut mutexer = arc_global_counter.lock().unwrap();
                        let key_to_filter = *mutexer;
                        *mutexer += 1;
                        key_to_filter
                    };
                    let elem_me = format!(
                        "elem {} {} theread: {}{}",
                        key_to_filter,
                        if key_to_filter > 9 { "" } else { " " },
                        tt,
                        if tt > 9 { "" } else { " " },
                    );
                    assert_eq!(mambo_arc.insert(key_to_filter as u64, &elem_me, false), None);
                }

                for key in 0..OPS_PER_THREED {
                    let key = tt + (key * OPS_PER_THREED * 10) + shift as usize;
                    let elem_me = format!("mambo elem {}", key);
                    //deleting an element, if such an element exists,
                    // it will be deleted from the table and returned as (T.clone())
                    assert_eq!(mambo_arc.remove(key as u64), Some(elem_me.clone()));
                }
            }));
        }

        for handle in std_handles {
            handle.join().unwrap();
        }

        println!("before filter:");
        mambo.filter(|mut_elem, key_u64| {
            /*The filter option is needed when you need to filter out all the elements in the hashtable
            or change them, passes a closure with the desired parameters RFy: FnMut(&mut T, u64) -> bool,
            where &mut T is an element and u64 is its key.bool is a decision to delete an item or not,
            if bool == true, then this item will not be deleted from the cache table. if bool ==  false,
            then the element will be deleted*/
            println!(
                "    elem: {}, {} key: {}",
                *mut_elem,
                if key_u64 > 9 { "" } else { " " },
                key_u64
            );

            *mut_elem += " is even";
            /* an example that leaves in the table only those elements that are divisible
             * by 2 without remainder */
            if 0 == key_u64 % 2 {
                return true;
            }
            false
        });

        println!("after filter: ");
        mambo.filter(|mut_elem, key_u64| {
            println!(
                "    elem: {:<28}, {} key: {}",
                *mut_elem,
                if key_u64 > 9 { "" } else { " " },
                key_u64
            );

            true
        });
        print_restores()
    }

    #[test]
    fn based_threars_test() {
        {
            const NUM_THREEDS: usize = 10;
            const OPS_PER_THREED: usize = 1000;
            const TEST_ELEMES: usize = 1000;
            let mambo = Mambo::<u64>::new(16, 5.0).unwrap();

            fn pair(u: usize) -> usize {
                let mut u: usize = u;
                for m in 0..12 {
                    u ^= u.rotate_left(m).wrapping_add(u.rotate_right(3 * m))
                        ^ PRIME_NUMBERS_TO_TAB[10 + m as usize] as usize;
                }
                u
            }

            let mut std_handles = Vec::new();
            let std_barrier = Arc::new(std::sync::Barrier::new(NUM_THREEDS + 1));

            for tt in 1..NUM_THREEDS + 1 {
                let barrier_clone = Arc::clone(&std_barrier);

                let mut ra_clone = mambo.clone();

                let handle = thread::spawn(move || {
                    barrier_clone.wait();

                    //println!("t: {}  np {:x}", tt, ptr::null_mut::<u8>() as usize);
                    //return;

                    if tt % 4 == 100 {
                        for _ in 0..OPS_PER_THREED {
                            for o in 0..TEST_ELEMES {
                                let pai = pair(o);
                                ra_clone
                                .read(o as u64, |ind| {
                                    let ind = ind.unwrap();
                                    assert_eq!(
                                        *ind,
                                        pai as u64,
                                        " based_threars_test non eq read  index: {}   rea: {}   in map: {}",o,pai as u32,*ind
                                    );
                                });
                            }
                        }
                    } else {
                        let elem_read_write = pair(pair(tt)) % TEST_ELEMES;

                        let iterations = pair(pair(elem_read_write)) % OPS_PER_THREED;

                        for _ in 0..iterations {
                            for yy in 0..elem_read_write {
                                let index = tt + (NUM_THREEDS * 10_000_000 * yy);
                                let yy = yy as u64;
                                ra_clone.insert(index as u64, &yy, false);
                            }
                            // println!("+++++++++++++++==");
                            for yy in 0..elem_read_write {
                                //println!("{}", yy);
                                let index = tt + (NUM_THREEDS * 10_000_000 * yy);
                                let _ = ra_clone.remove(index as u64).unwrap();

                                if 200 > TEST_ELEMES {
                                    println!("index {}", index);

                                    if ra_clone.data_arc.1.len() == 1 {
                                        for inn in 0..(TEST_ELEMES as f32 * (1.0 / 0.6)) as usize {
                                            ra_clone.read(inn as u64, |xx| {
                                                let xx = xx.unwrap();
                                                print!("| {} ", xx);
                                            });
                                            if inn % 4 == 3 {
                                                println!()
                                            }
                                        }
                                    } else {
                                        panic!("ra_clone.shards.len()==1else");
                                    }
                                }
                            }
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

        print_restores()
    }

    #[test]
    fn read_write() {
        for tre in (1..20).step_by(1) {
            //const NUM_THREADS: usize = 50;
            let num_treads: usize = tre;
            const NUM_ELEMS: usize = 20000;
            const TOTAL_OPS: u64 = 200_000;
            let ops_threads: u64 = TOTAL_OPS / num_treads as u64;
            {
                let mut mambo = Mambo::<u64>::new(100, 1.0).unwrap();
                let mut std_handles = Vec::new();
                let std_start = Instant::now();

                for i in 0..NUM_ELEMS {
                    let t = 0;
                    assert_eq!(mambo.insert(i as u64, &t, false), None);
                }

                for _ in 0..num_treads {
                    let mut ra_clone = mambo.clone();

                    let handle = thread::spawn(move || {
                        for i in 0..ops_threads {
                            let mut od: u64 = i;
                            for _ in 0..3 {
                                od = od.rotate_left(43).wrapping_add(!i);
                            }
                            ra_clone.read(od % NUM_ELEMS as u64, |x| {
                                let x = x.unwrap();
                                *x += 1;
                            });
                        }
                    });

                    std_handles.push(handle);
                }

                for handle in std_handles {
                    handle.join().unwrap();
                }
                if true {
                    println!(
                        "| threads: {:<2} |  read   | M.op/S: {:.3} |",
                        tre,
                        TOTAL_OPS as f64 / std_start.elapsed().as_micros() as f64
                    );
                }
            }
        }
        //assert!(false);
        print_restores()
    }

    #[test]
    fn test_gloabal_ctr_edit() {
        let mut mambo1 = Mambo::new(1, 10.0).unwrap();
        let mut mambo2 = Mambo::new(1, 4.0).unwrap();
        let mut mambo3 = Mambo::new(1, 1.0).unwrap();

        let mut rtccs = 0;
        let mut rtccs_max = 0;
        let mut rtccs_max_f = 0.0;

        for ii in 0..70 {
            mambo1.insert(ii as u64, &1u8, false);
            mambo2.insert(ii as u64, &1u8, false);
            mambo3.insert(ii as u64, &1u8, false);

            let ctr1 = {
                let guard = mambo1.data_arc.1[0].1.try_read().unwrap();
                *guard.0.lock().unwrap()
            };

            let ctr2 = {
                let guard = mambo2.data_arc.1[0].1.try_read().unwrap();
                *guard.0.lock().unwrap()
            };

            let ctr3 = {
                let guard = mambo3.data_arc.1[0].1.try_read().unwrap();
                *guard.0.lock().unwrap()
            };
            //
            //assert_eq!(ctr1, ii);

            if ctr1 != ii || ctr2 != ii || ctr3 != ii {
                rtccs += 1;

                println!("i:{}  ctr1: {}   ctr2: {}   ctr3: {}", ii, ctr1, ctr2, ctr3);

                let dif = ctr1.abs_diff(ii);

                if dif > rtccs_max {
                    rtccs_max = dif
                }

                if dif == rtccs_max {
                    rtccs_max_f += dif as f64;
                }
            }

            //
        }

        println!("\n \n \n \n \n all mistates: {}", rtccs);
        println!("dif mistates: {}", rtccs_max_f / rtccs as f64);
        //   mambo.insert(3, &1u8, false);
        print_restores()
    }

    #[test]
    fn test_gloabal_ctr_edit_removeca_insert() {
        let mut mambo1 = Mambo::new(1, 10.0).unwrap();

        mambo1.insert(0, &1u8, false);
        mambo1.insert(1, &1u8, false);
        mambo1.insert(2, &1u8, false);
        mambo1.insert(3, &1u8, false);

        for ii in 0..50 {
            let ctr1 = {
                let guard = mambo1.data_arc.1[0].1.try_read().unwrap();
                *guard.0.lock().unwrap()
            };
            // println!(" i ctr: {}", ctr1);

            mambo1.insert(ii % 4, &1u8, false);

            assert!(ctr1 <= 4);
        }

        for ii in 0..100 {
            let ctr1 = {
                let guard = mambo1.data_arc.1[0].1.try_read().unwrap();
                *guard.0.lock().unwrap()
            };
            //println!("r ctr: {}", ctr1);

            mambo1.remove(ii % 4);

            assert!(ctr1 <= 4);
        }
        print_restores()
    }

    #[test]
    fn based_panic_test() {
        {
            const NUM_THREADS: usize = 50;
            const OPS_PER_THREED: usize = 30;
            const TEST_ELEMES: usize = 50;
            let mambo = Mambo::<u64>::new(1, 5.0).unwrap();
            let uwelar = Arc::new(Mutex::new(0usize));

            fn pair(u: usize) -> usize {
                let mut u: usize = u;
                for m in 0..12 {
                    u ^= u.rotate_left(m).wrapping_add(u.rotate_right(3 * m))
                        ^ PRIME_NUMBERS_TO_TAB[10 + m as usize] as usize;
                }
                u
            }

            let mut std_handles = Vec::new();

            for tt in 1..NUM_THREADS + 1 {
                let mut ra_clone = mambo.clone();
                let uwls = Arc::clone(&uwelar);
                thread::sleep(time::Duration::from_millis(2));
                let handle = thread::spawn(move || {
                    let mut vec_keys_to_remove = vec![0; 0];

                    let mut uw_elem = 0usize;

                    let elem_read_write = pair(pair(tt)) % TEST_ELEMES;

                    let iterations = pair(pair(elem_read_write)) % OPS_PER_THREED;

                    for _ in 0..iterations {
                        for yy in 0..elem_read_write {
                            let index = tt + (NUM_THREADS * 10_000_000 * yy);
                            let yy = yy as u64;
                            ra_clone.insert(index as u64, &yy, false);
                            vec_keys_to_remove.push(index as u64);
                        }

                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let mut ra_clone = ra_clone.clone();
                            // let _ = std::panic::take_hook();
                            for &yy in vec_keys_to_remove.iter() {
                                ra_clone.read(yy, |x| {
                                    if yy % 7 == 0 {
                                        panic!("")
                                    }

                                    if let Some(val) = x {
                                        *val += 1;
                                    } else {
                                        uw_elem += 1;
                                    }
                                });
                            }
                        }));

                        //let b = (tt + elem_read_write) * ii;
                        if true {
                            //if b % 11 == 0 || b % 31 == 0 || true {
                            // println!("+++++++++++++++==");
                            for yy in vec_keys_to_remove.iter() {
                                let _ = ra_clone.remove(*yy);
                            }
                            vec_keys_to_remove = vec![0; 0];
                        }
                    }

                    let mut x = uwls.lock().unwrap();

                    *x += uw_elem;
                });

                std_handles.push(handle);
            }

            //panic!();
            for handle in std_handles {
                let x = handle.join();
                if x.is_err() {
                    println!("___ {:?}", x);
                };
            }

            let mut mut_all_real_elems = 0usize;
            let ra_x = mambo.data_arc.1[0].1.read().unwrap();

            for e in ra_x.1[0].iter() {
                mut_all_real_elems += e.lock().unwrap().1.len();
            }

            let yyyy = mambo.data_arc.1[0].1.read().unwrap();
            let xxxx = yyyy.0.lock().unwrap();
            println!(
                "in mut  {} in rw {}   in REAL: {}",
                xxxx,
                yyyy.1[0].len(),
                mut_all_real_elems
            );
            println!("mut_all_real_elems \n: {}", (mut_all_real_elems.abs_diff(*xxxx)));

            println!("\n\n\n was live elems: {} \n\n\n", *uwelar.lock().unwrap());
            // assert!(false);
        }
        print_restores()
    }

    #[test]
    fn based_panic_example() {
        let shards = 3;
        let elems_in_mutex = 7.0;
        const NUM_THREADS: usize = 100;
        const OPS_PER_THREED: usize = 10000;
        let mut mambo = Mambo::<String>::new(shards, elems_in_mutex).unwrap();
        let mut std_handles = Vec::new();
        for tt in 0..NUM_THREADS {
            let mut mambo_arc = mambo.clone();

            std_handles.push(thread::spawn(move || {
                for key in 0..OPS_PER_THREED {
                    let key = key + (OPS_PER_THREED * tt);

                    let elem_me = format!("mambo elem{}", key);

                    assert_eq!(mambo_arc.insert(key as u64, &elem_me, false), None);

                    mambo_arc.read(key as u64, |ind| {
                        if ind.is_none() {
                            return;
                        }
                        let ind = ind.unwrap();

                        assert_eq!(
                            ind.clone(),
                            elem_me,
                            " non eq read  key: {}   rea: {}   in map: {}",
                            key,
                            elem_me,
                            ind.clone()
                        );
                    });
                }
                if tt == 50 {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut mus = mambo_arc.data_arc.1[0].0.lock().unwrap();

                        if 1 == 1 {
                            panic!("");
                        }
                        *mus = 0;
                    }));
                }

                if tt == 51 {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut mus = mambo_arc.data_arc.1[1].1.write().unwrap();

                        if 1 == 1 {
                            panic!("");
                        }
                        mus.1[1] = vec![].into_boxed_slice();
                    }));
                }

                if tt == 52 {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut mus = mambo_arc.data_arc.1[3].0.lock().unwrap();

                        if 1 == 1 {
                            panic!("");
                        }
                        *mus += 1;
                    }));
                }

                if tt == 53 {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let rmus = mambo_arc.data_arc.1[1].1.read().unwrap();
                        let mut mux = rmus.0.lock().unwrap();

                        if 1 == 11 {
                            panic!("");
                        }
                        *mux += 1;
                    }));
                }
            }));
        }

        for handle in std_handles {
            let x = handle.join();
            if x.is_err() {
                println!("dym {:?}", x);
            }
        }

        let mut veve = vec![];

        mambo.filter(|_, a| {
            veve.push(a);
            //println!("{}", *e);
            true
        });

        veve.sort();
        // for x in veve.iter() {
        //    println!("{}", *x);
        // }

        println!("len: {}", veve.len());
        print_restores()
    }

    #[test]
    fn filter_test() {
        let mut mambo = Mambo::<u64>::new(30, 4.0).unwrap();
        let elem_in_cycle = 10u64;
        let cycles = 300u64;
        let mut i_key = 0;
        for xx in (1..cycles).step_by(1) {
            let mut clom = mambo.clone();
            //println!("{}", clom.elems_im_me().unwrap());
            for yy in 0..elem_in_cycle {
                assert_eq!(clom.insert((yy * cycles * 100) + xx, &i_key, false), None);
                i_key += 1;
            }
        }
        let mut all_elem = 0;
        mambo.filter(|_x, _key| {
            //println!("elem: {}", _x);
            if *_x % 31 == 0 {
                all_elem += 1;
                true
            } else {
                false
            }
        });

        mambo.filter(|_x, _key| {
            //println!("elem: {}", _x);
            assert_eq!(*_x % 31, 0);
            true
        });
        assert_eq!(mambo.elems_im_me(), all_elem);
        print_restores();

        // assert!(false);
    }

    #[test]

    fn read_write_2() {
        #![allow(clippy::float_cmp)]

        for tre in (1..20).step_by(1) {
            //const NUM_THREADS: usize = 50;
            let num_treads: usize = tre;
            const NUM_ELEMS: usize = 4000;
            const TOTAL_OPS: u64 = 1_000_000;

            let ops_threads: u64 = TOTAL_OPS / num_treads as u64;
            {
                let mut mambo = Mambo::<u64>::new(200, 4.0).unwrap();
                let mut std_handles = Vec::new();
                let std_start = Instant::now();

                for i in 0..NUM_ELEMS {
                    let t = 0;
                    assert_eq!(mambo.insert(i as u64, &t, false), None);
                }

                for _ in 0..num_treads {
                    let mut ra_clone = mambo.clone();

                    let handle = thread::spawn(move || {
                        for i in 0..ops_threads {
                            let mut od: u64 = i;
                            for _ in 0..3 {
                                od = od.rotate_left(43).wrapping_add(!i);
                            }
                            ra_clone.read(od % NUM_ELEMS as u64, |x| {
                                let x = x.unwrap();
                                *x += 1;
                            });
                        }
                    });

                    std_handles.push(handle);
                }

                for handle in std_handles {
                    handle.join().unwrap();
                }
                /*
                                if true {
                                    println!(
                                        "| threads: {} | {} | M.op/S: {:.3} |",
                                        if tre > 9 {
                                            format!("{}", tre)
                                        } else {
                                            format!("{} ", tre)
                                        },
                                        " read ",
                                        TOTAL_OPS as f64 / std_start.elapsed().as_micros() as f64
                                    );
                                    }
                */

                if true {
                    println!(
                        "| threads: {:<2} |  read  | M.op/S: {:.3} |",
                        tre,
                        TOTAL_OPS as f64 / std_start.elapsed().as_micros() as f64
                    );
                }
            }
        }
        print_restores();
        assert!(false == false);
    }
    /*
    use std::collections::BTreeSet;
    #[test]
    fn testbtree() {
        let std_start = Instant::now();
        for _ in 0..10_000u64 {
            let mut x2 = BTreeSet::<u64>::new();
            for x in 0..1000u64 {
                let x = (x.rotate_left(7).wrapping_add(x.wrapping_add(487623))) % 1000;

                x2.insert(x);
            }
        }

        println!("{}", 10_000_000.0 / std_start.elapsed().as_micros() as f32);

        assert!(false)
    }*/

    #[test]
    fn restore_rw_check() {
        let mut mambo = Mambo::<u64>::new(4, 4.0).unwrap();

        for ii in 0..200 {
            mambo.insert(ii, &ii, false);
        }

        let mambo_move = mambo.clone();
        let _ = panic::catch_unwind(move || {
            let t_l = mambo_move.data_arc.1[0].1.write().unwrap();

            if true == true {
                panic!("self poisin");
            }
            let _no_use = t_l;
        });

        for ii in 0..200 {
            let elem = mambo.remove(ii);
            // println!("{:?}", elem);

            if ii % 4 == 0 {
                assert_eq!(elem, None);
            }

            // mambo.insert(ii, &ii, false);
        }

        print_restores();

        // assert!(false);
    }

    #[test]
    fn restore_elem_mutex_check() {
        let mut mambo = Mambo::<u64>::new(1, 10.0).unwrap();

        for ii in 0..200 {
            mambo.insert(ii, &ii, false);
        }

        let mambo_move = mambo.clone();
        let _ = panic::catch_unwind(move || {
            let t_l = mambo_move.data_arc.1[0].1.read().unwrap();

            let mut _no_use = t_l.1[0][0].lock().unwrap();
            if true == true {
                panic!("self poisin");
            }
            _no_use.0 = false;
        });

        for ii in 0..200 {
            let elem = mambo.remove(ii);
            //println!("{:?}", elem);

            if ii % 23 == 0 {
                assert_eq!(elem, None);
            }

            // mambo.insert(ii, &ii, false);
        }

        print_restores();

        // assert!(false);
    }

    fn print_restores() {
        println!("\n=============================================================");
        let t_m_e = *test_how_restore_mutex_elem().lock().unwrap();
        let t_m_s = *test_how_restore_mutex_size().lock().unwrap();
        let t_r_w = *test_how_restore_rw_l().lock().unwrap();

        //
        let t_h_c = *test_how_call_resize().lock().unwrap();
        let t_h_b_c = *test_how_break_call_resize().lock().unwrap();

        println!("was restore: elems mutex:{}   mutex size:{}   rw shard:{} ", t_m_e, t_m_s, t_r_w);
        println!("               ");
        println!("call resize:{}   break:{}  ", t_h_c, t_h_b_c);
        println!("\n=============================================================");
    }
}

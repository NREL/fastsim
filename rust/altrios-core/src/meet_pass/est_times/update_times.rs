use super::super::disp_imports::*;

#[readonly::make]
#[derive(Debug, PartialEq, Clone, Copy)]
struct EstTimeNext {
    pub time_next: si::Time,
    pub est_idx: EstIdx,
}
impl EstTimeNext {
    pub fn new(time_next: si::Time, est_idx: EstIdx) -> Self {
        assert!(!time_next.is_nan());
        EstTimeNext { time_next, est_idx }
    }
}

impl PartialOrd for EstTimeNext {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for EstTimeNext {}

impl Ord for EstTimeNext {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time_next
            .partial_cmp(&self.time_next)
            .unwrap()
            .then_with(|| other.est_idx.cmp(&self.est_idx))
    }
}

#[readonly::make]
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
struct EstTimePrev {
    pub time_prev: si::Time,
    pub time_sub: si::Time,
    pub est_idx: EstIdx,
}
impl EstTimePrev {
    pub fn new(time_prev: si::Time, time_sub: si::Time, est_idx: EstIdx) -> Self {
        assert!(!time_prev.is_nan());
        assert!(!time_sub.is_nan());
        EstTimePrev {
            time_prev,
            time_sub,
            est_idx,
        }
    }
}

impl Eq for EstTimePrev {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for EstTimePrev {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Run shortest path forward on estimated time network
/// This adjusts the linking of all join nodes and sets the scheduled times
pub(super) fn update_times_forward(est_times: &mut [EstTime], time_depart: si::Time) {
    est_times[0].time_sched = time_depart;
    est_times[1].time_sched = time_depart;

    let mut queue = BinaryHeap::new();
    queue.push(EstTimeNext::new(time_depart, 1));

    while !queue.is_empty() {
        let mut idx_curr = queue.pop().unwrap().est_idx;
        let mut idx_next = est_times[idx_curr.idx()].idx_next;
        assert!(!est_times[idx_curr.idx()].time_sched.is_nan());
        assert!(est_times[idx_next.idx()].time_sched.is_nan());

        // Find and swap join nodes
        {
            // Iterate while the next node is a non-passed, non-base join node
            let idx_save = idx_next;
            let mut est_next = &est_times[idx_next.idx()];
            while est_next.link_event.est_type == EstType::Fake
                && est_times[est_next.idx_next.idx()].time_sched.is_nan()
                && est_next.idx_next.idx() != est_times.len() - 1
            {
                idx_next = est_next.idx_next;
                est_next = &est_times[idx_next.idx()];
            }

            // If iteration occurred, swap the links appropriately
            if idx_save != idx_next {
                let idx_base = est_times[idx_next.idx()].idx_prev;

                est_times[idx_curr.idx()].idx_next = idx_next;
                est_times[idx_base.idx()].idx_next = idx_save;

                est_times[idx_save.idx()].idx_prev = idx_base;
                est_times[idx_next.idx()].idx_prev = idx_curr;
            }
        }

        // Iterate until reaching any join node (but also process the first node)
        loop {
            // Add the next alt node to processing if it exists
            let idx_next_alt = est_times[idx_curr.idx()].idx_next_alt;
            if idx_next_alt != EST_IDX_NA {
                est_times[idx_next_alt.idx()].time_sched = est_times[idx_curr.idx()].time_sched;
                queue.push(EstTimeNext::new(
                    est_times[idx_next_alt.idx()].time_sched_next(),
                    idx_next_alt,
                ));
            }

            assert!(est_times[idx_next.idx()].time_sched.is_nan());
            est_times[idx_next.idx()].time_sched = est_times[idx_curr.idx()].time_sched_next();

            idx_curr = idx_next;
            idx_next = est_times[idx_next.idx()].idx_next;

            // Break if the next node is a join node or the last node
            if est_times[idx_next.idx()].idx_prev_alt != EST_IDX_NA
                || est_times[idx_next.idx()].link_event.est_type == EstType::Fake
            {
                break;
            }
        }

        // If the node has not been passed, add it
        if est_times[idx_next.idx()].time_sched.is_nan() {
            // If this is the last node, finish it and do not add it
            if est_times[idx_next.idx()].idx_next == EST_IDX_NA {
                est_times[idx_next.idx()].time_sched = est_times[idx_curr.idx()].time_sched;
            } else {
                assert!(est_times[idx_curr.idx()].link_event.est_type != EstType::Fake);
                queue.push(EstTimeNext::new(
                    est_times[idx_curr.idx()].time_sched_next(),
                    idx_curr,
                ));
            }
        } else {
            assert!(est_times[idx_curr.idx()].link_event.est_type == EstType::Fake);
            assert!(idx_curr == est_times[idx_next.idx()].idx_prev_alt);
        }
    }
}

/// Run shortest path backward on estimated time network
/// This adjusts the linking of all split nodes
/// and adjusts the scheduled times to reflects the shortest paths
pub(super) fn update_times_backward(est_times: &mut [EstTime]) {
    let mut is_est_passed = vec![false; est_times.len()];

    is_est_passed[est_times.len() - 1] = true;
    is_est_passed[est_times.len() - 2] = true;

    let mut queue = BinaryHeap::new();
    queue.push(EstTimePrev::new(
        si::Time::ZERO,
        si::Time::ZERO,
        est_times.len() as EstIdx - 2,
    ));

    while !queue.is_empty() {
        let (mut idx_curr, time_sub) = {
            let est_time_prev = queue.pop().unwrap();
            (est_time_prev.est_idx, est_time_prev.time_sub)
        };
        let mut idx_prev = est_times[idx_curr.idx()].idx_prev;
        assert!(is_est_passed[idx_curr.idx()]);
        assert!(!is_est_passed[idx_prev.idx()]);

        // Find and swap split nodes
        {
            // Iterate while the prev node is a non-passed, non-base split node
            let idx_save = idx_prev;
            let mut est_prev = &est_times[idx_prev.idx()];
            while est_prev.link_event.est_type == EstType::Fake
                && !is_est_passed[est_prev.idx_prev.idx()]
                && est_prev.idx_prev != EST_IDX_NA
            {
                idx_prev = est_prev.idx_prev;
                est_prev = &est_times[idx_prev.idx()];
            }

            // If iteration occured, swap the links appropriately
            if idx_save != idx_prev {
                let idx_base = est_times[idx_prev.idx()].idx_next;

                est_times[idx_curr.idx()].idx_prev = idx_prev;
                est_times[idx_base.idx()].idx_prev = idx_save;

                est_times[idx_save.idx()].idx_next = idx_base;
                est_times[idx_prev.idx()].idx_next = idx_curr;

                let time_to_next = est_times[idx_save.idx()].time_to_next;
                est_times[idx_save.idx()].time_to_next = est_times[idx_prev.idx()].time_to_next;
                est_times[idx_prev.idx()].time_to_next = time_to_next;

                let dist_to_next = est_times[idx_save.idx()].dist_to_next;
                est_times[idx_save.idx()].dist_to_next = est_times[idx_prev.idx()].dist_to_next;
                est_times[idx_prev.idx()].dist_to_next = dist_to_next;
            }
        }

        // Iterate until reaching any split node (but process the first node)
        loop {
            let idx_prev_alt = est_times[idx_curr.idx()].idx_prev_alt;
            if idx_prev_alt != EST_IDX_NA {
                let time_sched = est_times[idx_curr.idx()].time_sched;
                let time_sub_alt = est_times[idx_prev_alt.idx()].time_sched - time_sched;
                est_times[idx_prev_alt.idx()].time_sched = time_sched;
                is_est_passed[idx_prev_alt.idx()] = true;
                queue.push(EstTimePrev::new(time_sched, time_sub_alt, idx_prev_alt));
            }

            assert!(!is_est_passed[idx_prev.idx()]);
            est_times[idx_prev.idx()].time_sched -= time_sub;
            is_est_passed[idx_prev.idx()] = true;

            idx_curr = idx_prev;
            idx_prev = est_times[idx_prev.idx()].idx_prev;

            // Break if the prev node is a split node or the first node
            if est_times[idx_prev.idx()].idx_next_alt != EST_IDX_NA
                || est_times[idx_prev.idx()].link_event.est_type == EstType::Fake
            {
                break;
            }
        }

        // If the node has not been passed, add it
        if !is_est_passed[idx_prev.idx()] {
            // If this is the second node, finish it and the first node and do not add them
            if est_times[idx_prev.idx()].idx_prev == EST_IDX_NA {
                est_times[idx_prev.idx()].time_sched = est_times[idx_curr.idx()].time_sched;
                est_times[EST_IDX_NA.idx()].time_sched = est_times[idx_curr.idx()].time_sched;
                is_est_passed[idx_prev.idx()] = true;
                is_est_passed[EST_IDX_NA.idx()] = true;
            } else {
                assert!(est_times[idx_curr.idx()].link_event.est_type != EstType::Fake);
                queue.push(EstTimePrev::new(
                    est_times[idx_curr.idx()].time_sched - est_times[idx_prev.idx()].time_to_next,
                    time_sub,
                    idx_curr,
                ));
            }
        } else {
            assert!(est_times[idx_curr.idx()].link_event.est_type == EstType::Fake);
            assert!(idx_curr == est_times[idx_prev.idx()].idx_next_alt);
        }
    }
}

use super::disp_imports::*;
use crate::consist::Consist;
use uc::SPEED_DIFF_JOIN;
use uc::TIME_NAN;

mod est_time_structs;
mod update_times;

use est_time_structs::*;
use update_times::*;

/// Estimated time node for dispatching
/// Specifies the expected time of arrival when taking the shortest path with no delays
#[derive(Debug, Clone, Copy, Serialize, Deserialize, SerdeAPI)]
pub struct EstTime {
    /// Scheduled time of arrival at the node
    pub time_sched: si::Time,
    /// Time required to get to the next node when passing at speed "velocity"
    pub time_to_next: si::Time,
    /// Distance to the next node
    pub dist_to_next: si::Length,
    /// Speed at which the train will pass this node assuming no delays
    pub velocity: si::Velocity,

    /// Index of the next node in the network when traveling along the shortest path from this node
    pub idx_next: EstIdx,
    /// Index of the alternate next node (if it exists)
    /// Used if the shortest path is blocked up ahead
    pub idx_next_alt: EstIdx,
    /// Index of the previous node if the shortest path was taken to reach this node
    pub idx_prev: EstIdx,
    /// Index of the alternate previous node (if it exists)
    pub idx_prev_alt: EstIdx,

    /// Combination of link index and est type for this node
    /// Fake events have null link index
    pub link_event: LinkEvent,
}

impl EstTime {
    pub fn time_sched_next(&self) -> si::Time {
        self.time_sched + self.time_to_next
    }
}
impl Default for EstTime {
    fn default() -> Self {
        Self {
            time_sched: TIME_NAN,
            time_to_next: si::Time::ZERO,
            dist_to_next: si::Length::ZERO,
            velocity: si::Velocity::ZERO,
            idx_next: EST_IDX_NA,
            idx_next_alt: EST_IDX_NA,
            idx_prev: EST_IDX_NA,
            idx_prev_alt: EST_IDX_NA,
            link_event: Default::default(),
        }
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, SerdeAPI)]
#[altrios_api(
    pub fn get_running_time_hours(&self) -> f64 {
        (self.val.last().unwrap().time_sched - self.val.first().unwrap().time_sched).get::<si::hour>()
    }
)]
pub struct EstTimeNet {
    #[api(skip_get, skip_set)]
    pub val: Vec<EstTime>,
}
impl EstTimeNet {
    pub fn new(val: Vec<EstTime>) -> Self {
        Self { val }
    }
}

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn check_od_pair_valid(
    origs: Vec<Location>,
    dests: Vec<Location>,
    network: Vec<Link>,
) -> PyResult<()> {
    if let Err(error) = get_link_idx_options(&origs, &dests, &network) {
        Err(error.into())
    } else {
        Ok(())
    }
}

/// Get link indexes that lead to the destination (CURRENTLY ALLOWS LOOPS THAT
/// ARE TOO SMALL TO FIT THE TRAIN!)
pub fn get_link_idx_options(
    origs: &[Location],
    dests: &[Location],
    links: &[Link],
) -> Result<(IntSet<LinkIdx>, Vec<Location>), anyhow::Error> {
    // Allocate our empty link_idxs processing vector with enough initial space.
    let mut link_idxs_proc = Vec::<LinkIdx>::with_capacity(64.max(dests.len()));
    // Ensure our link_idx_set is initialized with the same capacity. We use
    // the set to ensure we only process each link_idx once.
    let mut link_idx_set =
        IntSet::<LinkIdx>::with_capacity_and_hasher(link_idxs_proc.capacity(), Default::default());
    // Go ahead and put all of the destination link indexes into our processing vector.
    link_idxs_proc.extend(dests.iter().map(|x| x.link_idx));

    // Keep track of the updated length of the origins.
    let mut origs = origs.to_vec();
    let mut origs_len_new = 0;

    // Now, pop each link_idx from the processing vector and process it.
    while let Some(link_idx) = link_idxs_proc.pop() {
        // If the link_idx has not yet been processed, process it.
        if !link_idx_set.contains(&link_idx) {
            link_idx_set.insert(link_idx);
            let origs_start = origs_len_new;
            for i in origs_start..origs.len() {
                if link_idx == origs[i].link_idx {
                    origs.swap(i, origs_len_new);
                    origs_len_new += 1;
                }
            }

            // If no origins were found, add the appropriate previous idxs to
            // the processing vector.
            if origs_start == origs_len_new {
                if let Some(&idx_prev) = links[link_idx.idx()].idx_prev.real() {
                    link_idxs_proc.push(idx_prev);
                    if let Some(&idx_prev_alt) = links[link_idx.idx()].idx_prev_alt.real() {
                        link_idxs_proc.push(idx_prev_alt);
                    }
                }
            }
        }
    }

    if link_idx_set.contains(&LinkIdx::default()) {
        bail!("Link idx options is not allowed to contain fake link idx!");
    }

    // No paths found, so return an error.
    if origs_len_new == 0 {
        bail!(
            "No valid paths found from any origin to any destination!\norigs: {:?}\ndests: {:?}",
            origs,
            dests
        );
    }
    origs.truncate(origs_len_new);

    // Return the set of processed link indices.
    Ok((link_idx_set, origs))
}

/// Convert sequence of train states to sequence of estimated times
/// that will be added to the estimated time network.
fn update_est_times_add(
    est_times_add: &mut Vec<EstTime>,
    movement: &[SimpleState],
    link_pts: &[LinkPoint],
    length: si::Length,
) {
    est_times_add.clear();
    let state_first = movement.first().unwrap();

    // Initialize location indices
    let mut pt_idx_back = 0;
    while link_pts[pt_idx_back].offset <= state_first.offset - length {
        pt_idx_back += 1;
    }
    let mut pt_idx_front = pt_idx_back;
    while link_pts[pt_idx_front].offset <= state_first.offset {
        pt_idx_front += 1;
    }

    // Convert movement to estimated times at linkPoints
    let mut offset_next = link_pts[pt_idx_front]
        .offset
        .min(link_pts[pt_idx_back].offset + length);
    for i in 1..movement.len() {
        // Add estimated times while in range
        while offset_next <= movement[i].offset {
            let dist_diff_x2 = 2.0 * (movement[i].offset - offset_next);
            let velocity = (movement[i].velocity * movement[i].velocity
                - (movement[i].velocity - movement[i - 1].velocity)
                    / (movement[i].time - movement[i - 1].time)
                    * dist_diff_x2)
                .sqrt();
            let time_to_next = movement[i].time - dist_diff_x2 / (movement[i].velocity + velocity);

            // Add either an arrive or a clear event depending on which happened earlier
            let link_event =
                if link_pts[pt_idx_back].offset + length < link_pts[pt_idx_front].offset {
                    pt_idx_back += 1;
                    if pt_idx_back == 1 {
                        offset_next = link_pts[pt_idx_front]
                            .offset
                            .min(link_pts[pt_idx_back].offset + length);
                        continue;
                    }
                    LinkEvent {
                        link_idx: link_pts[pt_idx_back - 1].link_idx,
                        est_type: EstType::Clear,
                    }
                } else {
                    pt_idx_front += 1;
                    LinkEvent {
                        link_idx: link_pts[pt_idx_front - 1].link_idx,
                        est_type: EstType::Arrive,
                    }
                };

            est_times_add.push(EstTime {
                time_to_next,
                dist_to_next: offset_next,
                velocity,
                link_event,
                ..Default::default()
            });
            offset_next = link_pts[pt_idx_front]
                .offset
                .min(link_pts[pt_idx_back].offset + length);
        }
    }
}

/// Insert a new estimated time into the network.
/// Insertion may not occur if the estimated time would be a duplicate.
/// Returns true if insertion occurred.
fn insert_est_time(
    est_times: &mut Vec<EstTime>,
    est_alt: &mut EstTime,
    link_event_map: &mut LinkEventMap,
    est_insert: &EstTime,
) -> bool {
    let mut is_insert = false;
    loop {
        let idx_push = est_times.len().try_into().unwrap();
        let idx_next = est_times[est_alt.idx_prev.idx()].idx_next;

        // If the insert time can be inserted directly, insert it and return true
        if idx_next == EST_IDX_NA {
            let est_prev = &mut est_times[est_alt.idx_prev.idx()];
            est_prev.idx_next = idx_push;
            est_prev.time_to_next = est_insert.time_to_next - est_prev.time_to_next;
            est_prev.dist_to_next = est_insert.dist_to_next - est_prev.dist_to_next;

            link_event_map
                .entry(est_insert.link_event)
                .or_default()
                .insert(est_prev.idx_next);
            let idx_old = est_alt.idx_prev;
            est_alt.idx_prev = est_prev.idx_next;

            est_times.push(*est_insert);
            est_times.last_mut().unwrap().idx_prev = idx_old;
            is_insert = true;
            break;
        }

        // If the insert time is the same as the next estimated time, update stored values, do not insert, and return false
        let est_match = &est_times[idx_next.idx()];
        if est_match.link_event == est_insert.link_event
            && (est_insert.velocity - est_match.velocity).abs() < SPEED_DIFF_JOIN
        {
            est_alt.idx_prev = idx_next;
            break;
        }

        // If there is no alternate node, insert a fake one
        let est_prev = &mut est_times[est_alt.idx_prev.idx()];
        if est_prev.idx_next_alt == EST_IDX_NA {
            est_prev.idx_next_alt = idx_push;
            est_times.push(*est_alt);
            est_alt.idx_prev = idx_push;
        }
        // Otherwise, update info est_alt
        else {
            est_alt.idx_prev = est_prev.idx_next_alt;
        }
    }

    est_alt.time_to_next = est_insert.time_to_next;
    est_alt.dist_to_next = est_insert.dist_to_next;
    is_insert
}

/// Update join paths and perform the space match, saving the ones that were extended
fn update_join_paths_space(
    est_join_paths_prev: &[EstJoinPath],
    est_join_paths: &mut Vec<EstJoinPath>,
    est_idxs_temp: &mut Vec<EstIdx>,
    est_time_add: &EstTime,
    est_times: &[EstTime],
    is_event_seen: bool,
) {
    assert!(est_join_paths.is_empty());
    assert!(est_idxs_temp.is_empty());

    for est_join_path in est_join_paths_prev {
        let mut est_time_prev = &est_times[est_join_path.est_idx_next.idx()];
        // Do not save the join path if it stops
        if est_time_prev.idx_next == EST_IDX_NA {
            continue;
        }

        // For arrive events, do not change the space match status
        let link_idx_match = if est_time_add.link_event.est_type == EstType::Arrive {
            est_join_path.link_idx_match
        }
        // For clear events, continue processing if a space match happened or is happening
        else if est_join_path.has_space_match()
            || est_join_path.link_idx_match == est_time_add.link_event.link_idx
        {
            track::LINK_IDX_NA
        }
        // For clear events, save the join path and continue if a space match has not happened
        // Note that est_join_path.idx_next does not change because the number of clear events can differ prior to a space match
        else {
            est_join_paths.push(*est_join_path);
            continue;
        };

        // If the join path cannot possibly continue, skip it. TODO: Verify that this is correct
        if !is_event_seen {
            continue;
        }

        // If the join path has already matched in space, find all estimated times that extend the join path along the travel path and save them
        // Note, new space matches are not handled here since they may be advanced over multiple est idxs
        if est_join_path.has_space_match() {
            // Iterate over all alternate nodes
            loop {
                // If the event matches, push a new extended join path
                if est_time_add.link_event == est_times[est_time_prev.idx_next.idx()].link_event {
                    est_join_paths.push(EstJoinPath::new(link_idx_match, est_time_prev.idx_next));
                }
                // Break when there are no more alternate nodes to check
                if est_time_prev.idx_next_alt == EST_IDX_NA {
                    break;
                }
                est_time_prev = &est_times[est_time_prev.idx_next_alt.idx()];
            }
        }
        // Advance to all possible next arrive events and check for space match using clear event
        else {
            loop {
                // Loop until reaching an event match, an arrive event, or the end
                loop {
                    // Add alternate node to the processing stack (if applicable)
                    if est_time_prev.idx_next_alt != EST_IDX_NA {
                        est_idxs_temp.push(est_time_prev.idx_next_alt)
                    }
                    debug_assert!(est_time_prev.idx_next != EST_IDX_NA);
                    let est_time_next = &est_times[est_time_prev.idx_next.idx()];

                    // If the event matches, push the new join path and stop advancing
                    if est_time_add.link_event == est_time_next.link_event {
                        debug_assert!(
                            est_time_add.link_event.est_type == EstType::Arrive
                                || link_idx_match.is_fake()
                        );
                        est_join_paths
                            .push(EstJoinPath::new(link_idx_match, est_time_prev.idx_next));
                        break;
                    }
                    // Break when reaching an arrive event
                    if est_time_next.link_event.est_type == EstType::Arrive
                        || est_time_next.idx_next == EST_IDX_NA
                    {
                        break;
                    }
                    est_time_prev = est_time_next;
                }
                match est_idxs_temp.pop() {
                    None => break,
                    Some(est_idx) => est_time_prev = &est_times[est_idx.idx()],
                };
            }
        }
    }
}

/// Check speed difference for space matched join paths and perform join for the best speed match (if sufficiently good)
/// Returns true if a join occurred.
fn perform_speed_join(
    est_join_paths: &[EstJoinPath],
    est_times: &mut Vec<EstTime>,
    est_time_add: &EstTime,
) -> bool {
    let mut velocity_diff_join = SPEED_DIFF_JOIN;
    let mut est_idx_join = EST_IDX_NA;
    for est_join_path in est_join_paths {
        if est_join_path.has_space_match() {
            let velocity_diff = (est_times[est_join_path.est_idx_next.idx()].velocity
                - est_time_add.velocity)
                .abs();
            if velocity_diff < velocity_diff_join {
                velocity_diff_join = velocity_diff;
                est_idx_join = est_join_path.est_idx_next;
            }
        }
    }

    if velocity_diff_join < SPEED_DIFF_JOIN {
        // TODO: Add assertion from C++

        let est_idx_last = (est_times.len() - 1).try_into().unwrap();
        // Join to specified estimated time
        loop {
            // If the target join node has a free previous index, join and return as successful
            let est_time_join = &mut est_times[est_idx_join.idx()];
            if est_time_join.idx_prev == EST_IDX_NA {
                est_time_join.idx_prev = est_idx_last;

                let est_time_prev = &mut est_times[est_idx_last.idx()];
                est_time_prev.idx_next = est_idx_join;
                est_time_prev.time_to_next = est_time_add.time_to_next - est_time_prev.time_to_next;
                est_time_prev.dist_to_next = est_time_add.dist_to_next - est_time_prev.dist_to_next;
                return true;
            }

            // If the target join node has a free previous alt index, attach a fake node
            if est_time_join.idx_prev_alt == EST_IDX_NA {
                let est_idx_attach = est_times.len().try_into().unwrap();
                est_times[est_idx_join.idx()].idx_prev_alt = est_idx_attach;
                est_times.push(EstTime {
                    idx_next: est_idx_join,
                    ..Default::default()
                });
                est_idx_join = est_idx_attach;
            }
            // Otherwise, traverse the previous alt index
            else {
                est_idx_join = est_time_join.idx_prev_alt;
            }
        }
    }
    false
}

/// For arrive events with an entry in the link event map, add new join paths
fn add_new_join_paths(
    link_event_add: &LinkEvent,
    link_event_map: &LinkEventMap,
    est_join_paths_save: &mut Vec<EstJoinPath>,
) {
    // Only add join paths for arrive events
    if link_event_add.est_type != EstType::Arrive {
        return;
    }
    if let Some(est_idxs_next) = link_event_map.get(link_event_add) {
        let mut est_idxs_new;
        // If there are no join paths, make a new join path for each est idx in the link event map entry
        let est_idxs_push = if est_join_paths_save.is_empty() {
            est_idxs_next
        }
        // If there are remaining join paths, use it to eliminate est idxs from the cloned link event map entry
        else {
            est_idxs_new = est_idxs_next.clone();
            for est_join_path in &*est_join_paths_save {
                est_idxs_new.remove(&est_join_path.est_idx_next);
            }
            &est_idxs_new
        };

        // Push a new join path for each value in est_idxs_push
        for est_idx in est_idxs_push {
            est_join_paths_save.push(EstJoinPath::new(link_event_add.link_idx, *est_idx));
        }
    }
}

pub fn make_est_times(
    speed_limit_train_sim: &SpeedLimitTrainSim,
    network: &[Link],
) -> anyhow::Result<(EstTimeNet, Consist)> {
    let dests = &speed_limit_train_sim.dests;
    let (link_idx_options, origs) =
        get_link_idx_options(&speed_limit_train_sim.origs, dests, network)?;

    let mut est_times = Vec::with_capacity(network.len() * 10);
    let mut consist_out = None;
    let mut saved_sims = Vec::<SavedSim>::with_capacity(16.max(network.len() / 10));
    let mut link_event_map =
        LinkEventMap::with_capacity_and_hasher(est_times.capacity(), Default::default());
    let time_depart = speed_limit_train_sim.state.time;

    // Push initial fake nodes
    est_times.push(EstTime {
        idx_next: 1,
        ..Default::default()
    });
    est_times.push(EstTime {
        time_to_next: time_depart,
        idx_prev: 0,
        ..Default::default()
    });

    // Add origin estimated times
    for orig in origs {
        ensure!(
            orig.offset == si::Length::ZERO,
            "Origin offset must be zero!"
        );
        ensure!(
            !orig.is_front_end,
            "Origin must be relative to the tail end!"
        );
        ensure!(orig.link_idx.is_real(), "Origin link idx must be real!");

        let mut est_alt = EstTime {
            time_to_next: time_depart,
            dist_to_next: orig.offset,
            idx_prev: 1,
            ..Default::default()
        };

        insert_est_time(
            &mut est_times,
            &mut est_alt,
            &mut link_event_map,
            &EstTime {
                time_to_next: time_depart,
                dist_to_next: orig.offset,
                velocity: si::Velocity::ZERO,
                link_event: LinkEvent {
                    link_idx: orig.link_idx,
                    est_type: EstType::Arrive,
                },
                ..Default::default()
            },
        );
        insert_est_time(
            &mut est_times,
            &mut est_alt,
            &mut link_event_map,
            &EstTime {
                time_to_next: time_depart,
                dist_to_next: orig.offset + speed_limit_train_sim.state.length,
                velocity: si::Velocity::ZERO,
                link_event: LinkEvent {
                    link_idx: orig.link_idx,
                    est_type: EstType::Clear,
                },
                ..Default::default()
            },
        );

        saved_sims.push(SavedSim {
            train_sim: {
                let mut train_sim = Box::new(speed_limit_train_sim.clone());
                train_sim.extend_path(network, &[orig.link_idx])?;
                train_sim
            },
            join_paths: vec![],
            est_alt,
        });
    }

    // Fix distances for different origins
    {
        let mut est_idx_fix = 1;
        while est_idx_fix != EST_IDX_NA {
            est_times[est_idx_fix.idx()].dist_to_next = si::Length::ZERO;
            est_idx_fix = est_times[est_idx_fix.idx()].idx_next_alt;
        }
    }

    let mut movement = Vec::<SimpleState>::with_capacity(32);
    let mut est_times_add = Vec::<EstTime>::with_capacity(32);
    let mut est_idxs_store = Vec::<EstIdx>::with_capacity(32);
    let mut est_join_paths_save = Vec::<EstJoinPath>::with_capacity(16);
    let mut est_idxs_end = Vec::<EstIdx>::with_capacity(8);

    // Iterate and process all saved sims
    while !saved_sims.is_empty() {
        let mut sim = saved_sims.pop().unwrap();
        let mut has_split = false;
        ensure!(
            sim.train_sim.link_idx_last().unwrap().is_real(),
            "Last link idx must be real! Link points={:?}",
            sim.train_sim.link_points()
        );

        'path: loop {
            sim.update_movement(&mut movement)?;
            update_est_times_add(
                &mut est_times_add,
                &movement,
                sim.train_sim.link_points(),
                sim.train_sim.state.length,
            );

            for est_time_add in &est_times_add {
                // Check for joins only if it has split from an old path
                if has_split {
                    update_join_paths_space(
                        &sim.join_paths,
                        &mut est_join_paths_save,
                        &mut est_idxs_store,
                        est_time_add,
                        &est_times,
                        link_event_map.contains_key(&est_time_add.link_event),
                    );

                    // If the join succeeds, break to the outer loop because this sim has finished being processed
                    if perform_speed_join(&est_join_paths_save, &mut est_times, est_time_add) {
                        est_join_paths_save.clear();
                        sim.join_paths.clear();
                        break 'path;
                    }

                    add_new_join_paths(
                        &est_time_add.link_event,
                        &link_event_map,
                        &mut est_join_paths_save,
                    );

                    std::mem::swap(&mut sim.join_paths, &mut est_join_paths_save);
                    est_join_paths_save.clear();
                }
                if insert_est_time(
                    &mut est_times,
                    &mut sim.est_alt,
                    &mut link_event_map,
                    est_time_add,
                ) {
                    has_split = true;
                }
            }

            // If finished, add destination node to final processing (all links should be clear)
            if sim.train_sim.is_finished() {
                est_idxs_end.push((est_times.len() - 1).try_into().unwrap());
                if consist_out.is_none() {
                    consist_out = Some(sim.train_sim.loco_con);
                }
                break;
            }
            // Otherwise, append the next link options and continue simulating
            else {
                let link_idx_prev = &sim.train_sim.link_idx_last().unwrap();
                let link_idx_next = network[link_idx_prev.idx()].idx_next;
                let link_idx_next_alt = network[link_idx_prev.idx()].idx_next_alt;
                ensure!(
                    link_idx_next.is_real(),
                    "Link idx next cannot be fake when making est times! link_idx_prev={link_idx_prev:?}"
                );

                if !link_idx_options.contains(&link_idx_next) {
                    ensure!(
                        link_idx_options.contains(&link_idx_next_alt),
                        "Unexpected end of path reached! prev={link_idx_prev:?}, next={link_idx_next:?}, next_alt={link_idx_next_alt:?}"
                    );
                    sim.train_sim.extend_path(network, &[link_idx_next_alt])?;
                } else {
                    if link_idx_options.contains(&link_idx_next_alt) {
                        let mut new_sim = sim.clone();
                        new_sim
                            .train_sim
                            .extend_path(network, &[link_idx_next_alt])?;
                        new_sim.check_dests(dests);
                        saved_sims.push(new_sim);
                    }
                    sim.train_sim.extend_path(network, &[link_idx_next])?;
                }
                sim.check_dests(dests);
            }
        }
    }

    // Finish the estimated time network
    ensure!(est_times.len() < (EstIdx::MAX as usize) - est_idxs_end.len());

    let mut est_idx_alt = EST_IDX_NA;
    for est_idx_end in est_idxs_end.iter().rev() {
        est_times.push(EstTime {
            idx_next: est_times.len() as EstIdx + 1,
            idx_prev: *est_idx_end,
            idx_prev_alt: est_idx_alt,
            ..Default::default()
        });
        est_idx_alt = est_times.len() as EstIdx - 1;
        est_times[est_idx_end.idx()].idx_next = est_idx_alt;
        est_times[est_idx_end.idx()].time_to_next = si::Time::ZERO;
        est_times[est_idx_end.idx()].dist_to_next = si::Length::ZERO;
    }

    est_times.push(EstTime {
        idx_prev: est_times.len() as EstIdx - 1,
        ..Default::default()
    });
    est_times.shrink_to_fit();

    for (idx, est_time) in est_times.iter().enumerate() {
        // Verify that all prev idxs are valid
        assert!((est_time.idx_prev != EST_IDX_NA) != (idx <= 1));
        // Verify that the next idxs are valid
        assert!((est_time.idx_next != EST_IDX_NA) != (idx == est_times.len() - 1));
        // Verify that no fake nodes have both idx prev alt and idx next alt
        assert!(
            est_time.link_event.est_type != EstType::Fake
                || est_time.idx_prev_alt == EST_IDX_NA
                || est_time.idx_next_alt == EST_IDX_NA
        );

        let est_time_prev = est_times[est_time.idx_prev.idx()];
        let est_time_next = est_times[est_time.idx_next.idx()];
        let est_idx = idx as EstIdx;
        // Verify that prev est time is linked to current est time
        assert!(est_time_prev.idx_next == est_idx || est_time_prev.idx_next_alt == est_idx);
        // Verify that next est time is linked to current est time
        assert!(
            est_time_next.idx_prev == est_idx
                || est_time_next.idx_prev_alt == est_idx
                || idx == est_times.len() - 1
        );

        // Verify that current est time is not the alternate of both the previous and next est times
        assert!(
            est_time_prev.idx_next_alt != est_idx
                || est_time_next.idx_prev_alt != est_idx
                || idx == 0
                || idx == est_times.len() - 1
        );
    }

    update_times_forward(&mut est_times, time_depart);
    update_times_backward(&mut est_times);

    // TODO: Write complete network validation function!

    Ok((EstTimeNet::new(est_times), consist_out.unwrap()))
}

#[cfg(feature = "pyo3")]
#[pyfunction(name = "make_est_times")]
pub fn make_est_times_py(
    speed_limit_train_sim: SpeedLimitTrainSim,
    network: Vec<Link>,
) -> PyResult<(EstTimeNet, Consist)> {
    Ok(make_est_times(&speed_limit_train_sim, &network)?)
}

use super::super::min_speed;
use super::super::SpeedLimit;
use crate::imports::*;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd, SerdeAPI)]
#[altrios_api]
pub struct SpeedPoint {
    #[api(skip_set)]
    pub offset: si::Length,
    #[api(skip_set)]
    pub speed_limit: si::Velocity,
}

impl GetOffset for SpeedPoint {
    fn get_offset(&self) -> si::Length {
        self.offset
    }
}

impl ObjState for SpeedPoint {
    fn validate(&self) -> Result<(), crate::validate::ValidationErrors> {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset, "Offset");
        si_chk_num(&mut errors, &self.speed_limit, "Speed");
        errors.make_err()
    }
}

#[ext(InsertSpeed)]
pub impl Vec<SpeedPoint> {
    /// Add a speed limit to speed points.
    /// The new speed limit must not start before the beginning of the current speed points.
    fn insert_speed(&mut self, speed_limit: &SpeedLimit) {
        debug_assert!(speed_limit.is_valid());
        debug_assert!(!self.is_fake());
        debug_assert!(self.is_valid());
        debug_assert!(self.first().unwrap().offset <= speed_limit.offset_start);

        // If the new speed is entirely after the end of all other speed points
        if self.last().unwrap().offset <= speed_limit.offset_start {
            let speed_old = self.last().unwrap().speed_limit;
            let speed_new = min_speed(speed_old, speed_limit.speed);
            if speed_old != speed_new {
                // If the new speed is strictly after the end of speed points
                if self.last().unwrap().offset < speed_limit.offset_start {
                    // Insert the new speed at its offset and add the old speed at the end
                    self.reserve(2);
                    self.push(SpeedPoint {
                        offset: speed_limit.offset_start,
                        speed_limit: speed_new,
                    });
                    self.push(SpeedPoint {
                        offset: speed_limit.offset_end,
                        speed_limit: speed_old,
                    });
                }
                // If the new speed matches the end location and the new speed equals the one prior
                else if self.len() > 1 && self[self.len() - 2].speed_limit == speed_new {
                    //Shift the offset of the last speed
                    self.last_mut().unwrap().offset = speed_limit.offset_end;
                }
                // If the new speed matches the end location and the new speed does not equal the one prior
                else {
                    // Overwrite the old last speed and add the old speed at the end
                    self.last_mut().unwrap().speed_limit = speed_new;
                    self.push(SpeedPoint {
                        offset: speed_limit.offset_end,
                        speed_limit: speed_old,
                    });
                }
            }
        } else {
            // Determine the range of impacted speed points
            let mut idx_start = 0usize;
            while speed_limit.offset_start > self[idx_start].offset {
                idx_start += 1;
            }
            let mut idx_end = self.len() - 1;
            while self[idx_end].offset > speed_limit.offset_end {
                idx_end -= 1;
            }

            // If the speed starts at an offset not already in speeds
            if speed_limit.offset_start < self[idx_start].offset {
                let speed_old = self[idx_start - 1].speed_limit;
                let speed_new = min_speed(speed_old, speed_limit.speed);

                // Insert if it is a more restrictive speed
                if speed_old != speed_new {
                    self.insert(
                        idx_start,
                        SpeedPoint {
                            offset: speed_limit.offset_start,
                            speed_limit: speed_new,
                        },
                    );
                    idx_start += 1;
                    idx_end += 1;
                }
            }

            // If the old speed does not end at offset end
            if self[idx_end].offset < speed_limit.offset_end {
                let speed_old = self[idx_end].speed_limit;

                // If the speed is different, insert the old speed at offset end
                if speed_old != min_speed(speed_old, speed_limit.speed) {
                    self.insert(
                        idx_end + 1,
                        SpeedPoint {
                            offset: speed_limit.offset_end,
                            speed_limit: speed_old,
                        },
                    );
                    idx_end += 1;
                }
            }

            // OLD VERSION:
            // Update and erase all speed points in range as appropriate
            while idx_start < idx_end {
                let speed_new = min_speed(self[idx_start].speed_limit, speed_limit.speed);
                if idx_start > 0 && self[idx_start - 1].speed_limit == speed_new {
                    self.remove(idx_start);
                    idx_end -= 1;
                } else {
                    self[idx_start].speed_limit = speed_new;
                    idx_start += 1;
                }
            }

            // Check and remove last speed point if applicable
            if idx_start > 0 && self[idx_start - 1].speed_limit == self[idx_start].speed_limit {
                self.remove(idx_start);
            }

            // NEW VERSION: CURRENTLY DOES NOT WORK
            // let idx_start = idx_start;
            // let idx_end = idx_end.max(idx_start);

            // // Update all speed points in range
            // for idx in idx_start..idx_end {
            //     let speed_new = min_speed(self[idx].speed, speed_limit.speed);
            //     if idx > 0 && self[idx - 1].speed == speed_new {
            //         let mut skipped = 1usize;

            //         //Update, shift, and drain all speed points in range
            //         for idx in idx + 1..idx_end {
            //             let speed_new = min_speed(self[idx].speed, speed_limit.speed);
            //             if self[idx - skipped - 1].speed == speed_new {
            //                 skipped += 1;
            //             } else {
            //                 self[idx - skipped] = SpeedPoint {
            //                     speed: speed_new,
            //                     ..self[idx]
            //                 };
            //             }
            //         }
            //         self.drain(idx_end - skipped..idx_end);
            //         break;
            //     }
            //     self[idx].speed = speed_new;
            // }

            // // Check and remove last speed point if applicable
            // if idx_end > 0 && self[idx_end - 1].speed == self[idx_end].speed {
            //     self.remove(idx_end);
            // }
        }
    }
}

impl ObjState for [SpeedPoint] {
    fn is_fake(&self) -> bool {
        self.is_empty()
    }
    fn validate(&self) -> ValidationResults {
        early_fake_ok!(self);
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Speed point");
        early_err!(errors, "Speed points");

        if !self.windows(2).all(|w| w[0].offset <= w[1].offset) {
            errors.push(anyhow!("Speed point offsets must be sorted!"));
        }
        if self.windows(3).any(|w| w[0].offset == w[2].offset) {
            errors.push(anyhow!(
                "Speed point offsets must not repeat more than twice!"
            ));
        }
        errors.make_err()
    }
}

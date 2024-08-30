use crate::imports::*;

/// Skews the peak of a curve to a specified new x-value, redistributing other
/// x-values linearly, preserving relative distances between peak and endpoints.
/// Arguments:
/// ----------
/// x: x-values of original curve (i.e. mc_pwr_out_perc when skewing motor
/// efficiency map for RustVehicle)  
/// y: y-values of the original curve (i.e. mc_eff_map when skewing motor
/// efficiency map RustVehicle)  
/// new_peak_x: new x-value at which to relocate peak
pub fn skewness_shift(
    x: &Array1<f64>,
    y: &Array1<f64>,
    new_peak_x: f64,
) -> anyhow::Result<(Array1<f64>, Array1<f64>)> {
    let y_max = y
        .clone()
        .into_iter()
        .reduce(f64::max)
        .with_context(|| "could not find maximum of y array")?;

    // Get index for maximum y-value. Use greatest index, if maximum occurs at
    // multiple indexes.
    let index_y_max = get_index_for_value(&y, y_max)?;

    // making vector versions of x and y arrays to manipulate
    let x_vec = x.to_vec();
    let y_vec = y.to_vec();
    let mut x_new_left = vec![];
    let mut y_new_left = vec![];
    let mut x_new_right = vec![];
    let mut y_new_right = vec![];

    // If points exist to the left of the peak
    if (index_y_max != 0) && (new_peak_x != y[0]) {
        for x_val in x_vec[0..index_y_max].iter() {
            x_new_left.push(
                x_vec[0] + (x_val - x_vec[0]) / (x_vec[index_y_max] - x[0]) * (new_peak_x - x[0]),
            )
        }
        y_new_left.append(y_vec[0..index_y_max].to_vec().as_mut());
    }

    // If points exist to the right of the peak
    if (index_y_max != y.len() - 1) && (new_peak_x != x_vec[x.len() - 1]) {
        for x_val in x_vec[index_y_max + 1..x.len()].iter() {
            x_new_right.push(
                new_peak_x
                    + (x_val - x[index_y_max]) / (x[x.len() - 1] - x[index_y_max])
                        * (x[x.len() - 1] - new_peak_x),
            )
        }
        y_new_right.append(y_vec[index_y_max + 1..y.len()].to_vec().as_mut());
    }

    let mut x_new = vec![];
    x_new.append(x_new_left.as_mut());
    x_new.push(new_peak_x);
    x_new.append(x_new_right.as_mut());

    let mut y_new = vec![];
    y_new.append(y_new_left.as_mut());
    y_new.push(y_max);
    y_new.append(y_new_right.as_mut());

    // Quality checks
    if x_new.len() != y_new.len() {
        return Err(anyhow!(
            "New x array and new y array do not have same length."
        ));
    }
    if x_new[0] != x[0] {
        return Err(anyhow!(
            "The first value of the new x array does not match the first value of the old x array."
        ));
    }
    let y_new_max = y_new
        .clone()
        .into_iter()
        .reduce(f64::max)
        .with_context(|| "could not find maximum of new y array")?;
    let new_index_y_max = get_index_for_value(&y_new.clone().try_into()?, y_new_max)?;
    if x_new[new_index_y_max] != new_peak_x {
        return Err(anyhow!(
            "The maximum in the new y array is not in the correct location."
        ));
    }
    if x_new[x_new.len() - 1] != x[x.len() - 1] {
        return Err(anyhow!(
            "The last value of the new x array does not equal the last value of the old x array."
        ));
    }

    Ok((x_new.try_into()?, y_new.try_into()?))
}

/// Gets the index for the a value in an array. If the value occurs more than
/// once in the array, chooses the largest index for which the value occurs.  
/// Arguments:  
/// ----------  
/// array: array to get index for  
/// value: value to check array for and get index of  
fn get_index_for_value(array: &Array1<f64>, value: f64) -> anyhow::Result<usize> {
    let mut index: usize = 0;
    let mut max_index_vec = vec![];
    for val in array.iter() {
        if val == &value {
            max_index_vec.push(index);
        }
        index = index + 1;
    }
    Ok(max_index_vec
        .iter()
        .max()
        .with_context(|| "Value not found in array.")?
        .to_owned())
}

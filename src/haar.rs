use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;

/// Represents a single weighted rectangle forming part of a Haar-like feature.
///
/// Coordinates and dimensions are defined relative to the model's base detection window
/// (e.g., 24x24 pixels).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HaarRect {
    /// Horizontal relative offset from the top-left corner of the detection window.
    pub x: usize,
    /// Vertical relative offset from the top-left corner of the detection window.
    pub y: usize,
    /// Width of the rectangle in base window units.
    pub width: usize,
    /// Height of the rectangle in base window units.
    pub height: usize,
    /// Weight/multiplier applied to the sum of pixel intensities within this rectangle.
    pub weight: f32,
}

/// A Haar-like feature composed of multiple weighted rectangles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaarFeature {
    /// Vector of rectangles that make up the feature (typically 2 to 3 rectangles).
    pub rects: Vec<HaarRect>,
}

/// A weak decision stump classifier within a cascade stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakClassifier {
    /// The geometric Haar feature to evaluate.
    pub feature: HaarFeature,
    /// Unscaled decision threshold for the normalized feature value.
    pub threshold: f32,
    /// Accumulator output added if the normalized feature value is less than `threshold * std_dev`.
    pub left_val: f32,
    /// Accumulator output added if the normalized feature value is greater or equal to `threshold * std_dev`.
    pub right_val: f32,
}

/// A stage in the cascade containing multiple weak decision classifiers.
///
/// All weak classifiers in a stage must be evaluated to produce a cumulative score.
/// If the score falls below `stage_threshold`, the candidate window is immediately rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeStage {
    /// Minimum threshold cumulative sum required to pass this cascade stage.
    pub stage_threshold: f32,
    /// List of weak classifiers in this stage.
    pub classifiers: Vec<WeakClassifier>,
}

/// Represents a detected bounding box in image coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Detection {
    /// Top-left X coordinate in pixels.
    pub x: usize,
    /// Top-left Y coordinate in pixels.
    pub y: usize,
    /// Width of the detected bounding box.
    pub width: usize,
    /// Height of the detected bounding box.
    pub height: usize,
}

/// An Integral Image (Summed-Area Table) for fast $O(1)$ rectangle evaluations.
///
/// Holds both the cumulative sum of grayscale pixel values (`sum`) and the
/// cumulative sum of squared pixel values (`sqsum`) to compute regional means
/// and standard deviations in constant time.
pub struct IntegralImage {
    /// Width of the source image in pixels.
    pub width: usize,
    /// Height of the source image in pixels.
    pub height: usize,
    /// Row stride of the internal buffers, equal to `width + 1` to account for padding.
    pub stride: usize,
    /// Cumulative sum array of size `(width + 1) * (height + 1)`.
    pub sum: Vec<u64>,
    /// Cumulative squared sum array of size `(width + 1) * (height + 1)`.
    pub sqsum: Vec<u64>,
}

impl IntegralImage {
    /// Allocates a new `IntegralImage` with pre-budgeted memory buffers.
    ///
    /// # Arguments
    /// * `width` - Frame width in pixels.
    /// * `height` - Frame height in pixels.
    pub fn new(width: usize, height: usize) -> Self {
        let stride = width + 1;
        let size = stride * (height + 1);
        Self {
            width,
            height,
            stride,
            sum: vec![0; size],
            sqsum: vec![0; size],
        }
    }

    /// Re-populates the integral and squared integral buffers from an RGB (0x00RRGGBB) buffer.
    ///
    /// Reuses existing heap allocations to eliminate dynamic memory allocation overhead
    /// inside the hot video frame loop.
    ///
    /// # Arguments
    /// * `buffer` - Slice of 32-bit ARGB/XRGB pixels.
    /// * `width` - Frame width in pixels.
    /// * `height` - Frame height in pixels.
    pub fn update(&mut self, buffer: &[u32], width: usize, height: usize) {
        self.width = width;
        self.height = height;
        self.stride = width + 1;
        let needed_len = self.stride * (height + 1);

        if self.sum.len() < needed_len {
            self.sum.resize(needed_len, 0);
            self.sqsum.resize(needed_len, 0);
        }

        let stride = self.stride;

        for y in 0..height {
            let mut row_sum = 0u64;
            let mut row_sqsum = 0u64;

            let buf_offset = y * width;
            let curr_row_stride = (y + 1) * stride;
            let prev_row_stride = y * stride;

            for x in 0..width {
                // Fetch pixel color without redundant bounds checks.
                let pixel = unsafe { *buffer.get_unchecked(buf_offset + x) };
                let r = ((pixel >> 16) & 0xFF) as u64;
                let g = ((pixel >> 8) & 0xFF) as u64;
                let b = (pixel & 0xFF) as u64;

                // Fixed-point integer approximation for ITU-R BT.601 luminance:
                // Equivalent to: (r * 0.299 + g * 0.587 + b * 0.114)
                // Uses bit-shift right (>> 8) instead of division to save CPU cycles.
                let gray = (r * 77 + g * 150 + b * 29) >> 8;

                row_sum += gray;
                row_sqsum += gray * gray;

                let idx = curr_row_stride + (x + 1);
                let prev_row_idx = prev_row_stride + (x + 1);

                // Update dynamic programming integral image state:
                // I(x, y) = I(x, y-1) + row_sum(x)
                unsafe {
                    *self.sum.get_unchecked_mut(idx) =
                        *self.sum.get_unchecked(prev_row_idx) + row_sum;
                    *self.sqsum.get_unchecked_mut(idx) =
                        *self.sqsum.get_unchecked(prev_row_idx) + row_sqsum;
                }
            }
        }
    }
}

/// Highly optimized internal structure holding pre-calculated 1D flat memory offsets
/// for a scaled feature rectangle's four corners (`off00`, `off10`, `off01`, `off11`).
///
/// Pre-computing these relative offsets avoids row multiplication `(y * stride + x)`
/// inside the inner scanning window loop.
#[derive(Clone, Copy)]
struct FastFeatureRect {
    /// 1D memory offset for top-left corner (x, y).
    off00: usize,
    /// 1D memory offset for top-right corner (x + w, y).
    off10: usize,
    /// 1D memory offset for bottom-left corner (x, y + h).
    off01: usize,
    /// 1D memory offset for bottom-right corner (x + w, y + h).
    off11: usize,
    /// Rectangle weight factor.
    weight: f32,
}

/// Optimized representation of a weak classifier using pre-computed 1D offset rects.
#[derive(Clone)]
struct FastWeakClassifier {
    rects: Vec<FastFeatureRect>,
    threshold: f32,
    left_val: f32,
    right_val: f32,
}

/// Optimized representation of a cascade stage using pre-computed classifiers.
#[derive(Clone)]
struct FastStage {
    stage_threshold: f32,
    classifiers: Vec<FastWeakClassifier>,
}

/// Pre-computed 1D relative memory offsets for the entire scanning window boundary corners.
/// Used to calculate mean intensity and variance in $O(1)$ time.
#[derive(Clone, Copy)]
struct WindowOffsets {
    p00: usize,
    p10: usize,
    p01: usize,
    p11: usize,
}

/// Main Cascade Classifier container supporting loading from JSON and multithreaded detection.
#[derive(Serialize, Deserialize)]
pub struct CascadeClassifier {
    /// Base training window width in pixels (e.g., 24).
    pub window_width: usize,
    /// Base training window height in pixels (e.g., 24).
    pub window_height: usize,
    /// Stages of the cascade classifier model.
    pub stages: Vec<CascadeStage>,
}

impl CascadeClassifier {
    /// Deserializes a cascade classifier from a JSON specification file.
    pub fn from_json_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let cascade = serde_json::from_reader(reader)?;
        Ok(cascade)
    }

    /// Pre-scales all feature rectangle coordinates and calculates their 1D relative memory offsets
    /// for a specific scale level and buffer stride.
    fn prepare_fast_stages(&self, scale: f32, stride: usize) -> Vec<FastStage> {
        self.stages
            .iter()
            .map(|stage| FastStage {
                stage_threshold: stage.stage_threshold,
                classifiers: stage
                    .classifiers
                    .iter()
                    .map(|c| FastWeakClassifier {
                        threshold: c.threshold,
                        left_val: c.left_val,
                        right_val: c.right_val,
                        rects: c
                            .feature
                            .rects
                            .iter()
                            .map(|r| {
                                let rx = (r.x as f32 * scale).round() as usize;
                                let ry = (r.y as f32 * scale).round() as usize;
                                let rw = (r.width as f32 * scale).round() as usize;
                                let rh = (r.height as f32 * scale).round() as usize;

                                FastFeatureRect {
                                    off00: ry * stride + rx,
                                    off10: ry * stride + (rx + rw),
                                    off01: (ry + rh) * stride + rx,
                                    off11: (ry + rh) * stride + (rx + rw),
                                    weight: r.weight,
                                }
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect()
    }

    /// Evaluates all cascade stages for a single sliding window position.
    ///
    /// # Performance
    /// Marked with `#[inline(always)]` and uses `unsafe { get_unchecked() }` to omit
    /// boundary checks. Uses pre-calculated 1D offsets to eliminate multiplication operations.
    ///
    /// # Returns
    /// `true` if the candidate window passes all stages (face found), `false` otherwise.
    #[inline(always)]
    fn eval_window_fast(
        stages: &[FastStage],
        sum: &[u64],
        sqsum: &[u64],
        win_offset: usize,
        win_off: &WindowOffsets,
        inv_area: f32,
        area_f64: f64,
    ) -> bool {
        unsafe {
            // $O(1)$ window sum lookup using 4 corner points in integral image:
            // Sum = P11 + P00 - P10 - P01
            let total_sum = (*sum.get_unchecked(win_offset + win_off.p11)
                + *sum.get_unchecked(win_offset + win_off.p00)
                - *sum.get_unchecked(win_offset + win_off.p10)
                - *sum.get_unchecked(win_offset + win_off.p01)) as f64;

            // $O(1)$ window squared sum lookup for variance evaluation
            let total_sqsum = (*sqsum.get_unchecked(win_offset + win_off.p11)
                + *sqsum.get_unchecked(win_offset + win_off.p00)
                - *sqsum.get_unchecked(win_offset + win_off.p10)
                - *sqsum.get_unchecked(win_offset + win_off.p01))
                as f64;

            // Precise floating-point variance calculation (f64 precision avoids underflow/overflow)
            let mean = total_sum / area_f64;
            let variance = (total_sqsum / area_f64) - (mean * mean);

            // Fast rejection of low-contrast/homogeneous regions (e.g. flat walls or background)
            if variance < 10.0 {
                return false;
            }

            let std_dev = variance.sqrt() as f32;

            // Sequential evaluation of cascade stages
            for stage in stages {
                let mut stage_sum = 0.0f32;

                for classifier in &stage.classifiers {
                    let mut feat_sum = 0.0f32;

                    for rect in &classifier.rects {
                        let r_sum = (*sum.get_unchecked(win_offset + rect.off11)
                            + *sum.get_unchecked(win_offset + rect.off00)
                            - *sum.get_unchecked(win_offset + rect.off10)
                            - *sum.get_unchecked(win_offset + rect.off01))
                            as f32;

                        feat_sum += r_sum * rect.weight;
                    }

                    // OpenCV Feature Normalization Formula: Divide total sum by scaled window area
                    let norm_feat_val = feat_sum * inv_area;

                    if norm_feat_val < classifier.threshold * std_dev {
                        stage_sum += classifier.left_val;
                    } else {
                        stage_sum += classifier.right_val;
                    }
                }

                // Stage early rejection: exit immediately if stage score requirement isn't met
                if stage_sum < stage.stage_threshold {
                    return false;
                }
            }

            true
        }
    }

    /// Performs multi-scale object detection across the integral image.
    ///
    /// Employs parallel row processing using Rayon (`into_par_iter`), adaptive step sizing,
    /// and pre-calculated memory offsets.
    ///
    /// # Arguments
    /// * `ii` - Reference to the processed `IntegralImage`.
    /// * `scale_factor` - Scaling factor between image pyramid levels (e.g., 1.2).
    /// * `base_step` - Minimum sliding window step size in pixels.
    /// * `min_neighbors` - Minimum surrounding detections required to keep a cluster during grouping.
    pub fn detect(
        &self,
        ii: &IntegralImage,
        scale_factor: f32,
        base_step: usize,
        min_neighbors: usize,
    ) -> Vec<Detection> {
        let mut raw_detections = Vec::new();
        let mut scale = 1.0f32;
        let stride = ii.stride;

        // Scale pyramid loop: gradually increase window dimensions until it exceeds frame size
        while (self.window_width as f32 * scale) as usize <= ii.width
            && (self.window_height as f32 * scale) as usize <= ii.height
        {
            let win_w = (self.window_width as f32 * scale).round() as usize;
            let win_h = (self.window_height as f32 * scale).round() as usize;

            if win_w == 0 || win_h == 0 {
                scale *= scale_factor;
                continue;
            }

            // Precompute scaled features and offsets for the current scale factor level
            let fast_stages = self.prepare_fast_stages(scale, stride);
            let inv_area = 1.0 / ((win_w * win_h) as f32);
            let area_f64 = (win_w * win_h) as f64;

            let win_off = WindowOffsets {
                p00: 0,
                p10: win_w,
                p01: win_h * stride,
                p11: win_h * stride + win_w,
            };

            // Dynamic adaptive step size proportional to window width
            let step_size = (win_w as f32 * 0.10).max(base_step as f32).round() as usize;
            let max_x = ii.width - win_w;
            let max_y = ii.height - win_h;

            let y_steps: Vec<usize> = (0..=max_y).step_by(step_size).collect();

            // Parallel multithreaded scanning across Y rows using Rayon worker thread pool
            let scale_detections: Vec<Detection> = y_steps
                .into_par_iter()
                .flat_map(|y| {
                    let mut local_dets = Vec::new();
                    let row_offset = y * stride;

                    for x in (0..=max_x).step_by(step_size) {
                        let win_offset = row_offset + x;

                        if Self::eval_window_fast(
                            &fast_stages,
                            &ii.sum,
                            &ii.sqsum,
                            win_offset,
                            &win_off,
                            inv_area,
                            area_f64,
                        ) {
                            local_dets.push(Detection {
                                x,
                                y,
                                width: win_w,
                                height: win_h,
                            });
                        }
                    }
                    local_dets
                })
                .collect();

            raw_detections.extend(scale_detections);
            scale *= scale_factor;
        }

        // Cluster and suppress duplicate overlapping candidate bounding boxes
        group_detections(raw_detections, min_neighbors, 0.2)
    }
}

/// Groups overlapping detection candidate rectangles into single averaged bounding boxes
/// based on Intersection over Union (IoU) clustering.
///
/// # Arguments
/// * `rects` - Unfiltered raw candidate detections.
/// * `min_neighbors` - Minimum number of overlapping detections required to confirm a cluster.
/// * `iou_threshold` - IoU threshold ratio above which rectangles are grouped together.
pub fn group_detections(
    rects: Vec<Detection>,
    min_neighbors: usize,
    iou_threshold: f32,
) -> Vec<Detection> {
    if rects.is_empty() {
        return Vec::new();
    }

    let mut clusters: Vec<Vec<Detection>> = Vec::new();

    for rect in rects {
        let mut matched = false;
        for cluster in &mut clusters {
            if intersection_over_union(&rect, &cluster[0]) > iou_threshold {
                cluster.push(rect);
                matched = true;
                break;
            }
        }
        if !matched {
            clusters.push(vec![rect]);
        }
    }

    let mut result = Vec::new();
    for cluster in clusters {
        if cluster.len() >= min_neighbors {
            let count = cluster.len();
            let avg_x = cluster.iter().map(|r| r.x).sum::<usize>() / count;
            let avg_y = cluster.iter().map(|r| r.y).sum::<usize>() / count;
            let avg_w = cluster.iter().map(|r| r.width).sum::<usize>() / count;
            let avg_h = cluster.iter().map(|r| r.height).sum::<usize>() / count;

            result.push(Detection {
                x: avg_x,
                y: avg_y,
                width: avg_w,
                height: avg_h,
            });
        }
    }

    result
}

/// Computes the Intersection over Union (IoU) overlap score between two detection rectangles.
fn intersection_over_union(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = ((x2 - x1) * (y2 - y1)) as f32;
    let area_a = (a.width * a.height) as f32;
    let area_b = (b.width * b.height) as f32;

    intersection / (area_a + area_b - intersection)
}

/// Draws detection bounding box outlines into a 32-bit pixel buffer.
///
/// # Arguments
/// * `pixel_buffer` - Target 0x00RRGGBB pixel buffer slice.
/// * `width` - Buffer frame width.
/// * `height` - Buffer frame height.
/// * `detections` - List of grouped detections to render.
/// * `color` - Packed 32-bit color value (e.g. `0x0000FF00` for green).
pub fn draw_detections(
    pixel_buffer: &mut [u32],
    width: usize,
    height: usize,
    detections: &[Detection],
    color: u32,
) {
    let thickness = 2;

    for det in detections {
        let x1 = det.x;
        let y1 = det.y;
        let x2 = (det.x + det.width).min(width);
        let y2 = (det.y + det.height).min(height);

        for t in 0..thickness {
            // Top and bottom lines
            let top_y = y1 + t;
            let bot_y = if y2 > t { y2 - 1 - t } else { 0 };

            if top_y < height {
                let row_offset = top_y * width;
                for x in x1..x2 {
                    pixel_buffer[row_offset + x] = color;
                }
            }
            if bot_y < height && bot_y >= y1 {
                let row_offset = bot_y * width;
                for x in x1..x2 {
                    pixel_buffer[row_offset + x] = color;
                }
            }

            // Left and right lines
            let left_x = x1 + t;
            let right_x = if x2 > t { x2 - 1 - t } else { 0 };

            for y in y1..y2 {
                let row_offset = y * width;
                if left_x < width {
                    pixel_buffer[row_offset + left_x] = color;
                }
                if right_x < width && right_x >= x1 {
                    pixel_buffer[row_offset + right_x] = color; 
                }
            }
        }
    }
}

mod haar;
mod web_cam;

use haar::{CascadeClassifier, IntegralImage, draw_detections};
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};
use std::error::Error;
use std::time::Instant;
use web_cam::CameraDriver;

fn main() -> Result<(), Box<dyn Error>> {
    let mut camera = CameraDriver::new(0)?;
    let width = camera.width();
    let height = camera.height();

    let mut window_options = WindowOptions::default();
    window_options.scale = Scale::X1;
    window_options.resize = true;
    // AspectRatioStretch adjust the frame to fit the window while preserving proportions
    window_options.scale_mode = ScaleMode::AspectRatioStretch;

    let mut window = Window::new(
        "Real-Time Face Detection (Rust)",
        width,
        height,
        window_options,
    )?;
    window.set_target_fps(120);

    let mut pixel_buffer = vec![0u32; width * height];
    let mut ii = IntegralImage::new(width, height);

    let cascade = CascadeClassifier::from_json_file("./assets/face_cascade.json")?;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        camera.capture_frame(&mut pixel_buffer)?;

        let start = Instant::now();

        ii.update(&pixel_buffer, width, height);
        let detections = cascade.detect(&ii, 1.2, 2, 2);

        let elapsed = start.elapsed();
        let fps = 1.0 / elapsed.as_secs_f32().max(0.0001);

        window.set_title(&format!(
            "Face Detection | FPS: {:.1} ({:.2?}) | Найдено лиц: {}",
            fps,
            elapsed,
            detections.len()
        ));

        draw_detections(&mut pixel_buffer, width, height, &detections, 0x0000FF00);

        window.update_with_buffer(&pixel_buffer, width, height)?;
    }

    Ok(())
}

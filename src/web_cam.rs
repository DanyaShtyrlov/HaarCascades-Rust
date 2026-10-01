use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
use std::error::Error;

pub struct CameraDriver {
    camera: Camera,
    width: usize,
    height: usize,
}

impl CameraDriver {
    pub fn new(index: u32) -> Result<Self, Box<dyn Error>> {
        let cam_index = CameraIndex::Index(index);
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        let mut camera = Camera::new(cam_index, requested)?;

        let _ = camera.set_resolution(Resolution::new(1920, 1080));

        camera.open_stream()?;

        let width = camera.resolution().width() as usize;
        let height = camera.resolution().height() as usize;

        Ok(Self {
            camera,
            width,
            height,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn capture_frame(&mut self, pixel_buffer: &mut [u32]) -> Result<(), Box<dyn Error>> {
        if let Ok(frame) = self.camera.frame() {
            if let Ok(rgb_image) = frame.decode_image::<RgbFormat>() {
                let raw_pixels = rgb_image.as_raw();

                for (i, chunk) in raw_pixels.chunks_exact(3).enumerate() {
                    if i < pixel_buffer.len() {
                        let r = chunk[0] as u32;
                        let g = chunk[1] as u32;
                        let b = chunk[2] as u32;
                        pixel_buffer[i] = (r << 16) | (g << 8) | b;
                    }
                }
            }
        }
        Ok(())
    }
}

impl Drop for CameraDriver {
    fn drop(&mut self) {
        let _ = self.camera.stop_stream();
    }
}

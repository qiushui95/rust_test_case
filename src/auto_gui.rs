use crate::assets::Assets;
use image::{codecs::png::PngEncoder, ColorType, DynamicImage, GenericImageView, ImageEncoder};
use std::error::Error;

use opencv::{
    core::{self, min_max_loc, Point, Rect, Scalar},
    imgcodecs, imgproc,
    prelude::*,
};

pub struct AutoGui {
    debug: bool,
}

pub struct FindImageRegion {
    left: u32,
    top: u32,
    width: u32,
    height: u32,
}

impl FindImageRegion {
    pub fn region(left: u32, top: u32, width: u32, height: u32) -> Self {
        assert!(width > 0, "width must > 0");
        assert!(height > 0, "height must > 0");
        Self { left, top, width, height }
    }
}

pub struct FindImageResult {
    pub left: u32,
    pub top: u32,
    pub precision: f32,
}

pub struct FindImageResults {
    pub width: u32,
    pub height: u32,
    pub list: Vec<FindImageResult>,
}

impl FindImageResults {
    pub fn first(&self) -> Option<&FindImageResult> {
        self.list.first()
    }
}

pub struct FindImageResultFilter {
    x_delta: u32,
    y_delta: u32,
}

impl FindImageResultFilter {
    pub fn new(x_delta: u32, y_delta: u32) -> Self {
        Self { x_delta, y_delta }
    }
}

impl FindImageResultFilter {
    fn need_filter(&self, result: &FindImageResult, info: (u32, u32, f32)) -> bool {
        if info.0 < result.left.saturating_sub(self.x_delta) {
            return false;
        }
        if info.0 > result.left + self.x_delta {
            return false;
        }
        if info.1 < result.top.saturating_sub(self.y_delta) {
            return false;
        }
        if info.1 > result.top + self.y_delta {
            return false;
        }
        true
    }
}

impl AutoGui {
    pub fn new(debug: bool) -> Result<Self, Box<dyn Error>> {
        Ok(Self { debug })
    }

    fn resize_image(img: DynamicImage, template_width: Option<u32>) -> DynamicImage {
        let Some(template_width) = template_width else {
            return img;
        };
        let (width, height) = img.dimensions();
        let ratio = template_width as f32 / width as f32;
        let resize_height = (height as f32 * ratio).round() as u32;
        img.resize(
            template_width,
            resize_height,
            image::imageops::FilterType::Lanczos3,
        )
    }

    fn _to_auto_gui_region(region: Option<FindImageRegion>) -> Option<(u32, u32, u32, u32)> {
        let Some(region) = region else {
            return None;
        };
        Some((region.left, region.top, region.width, region.height))
    }

    fn dynamic_image_to_png_bytes(img: &DynamicImage) -> Result<Vec<u8>, Box<dyn Error>> {
        let rgba = img.to_rgba8();
        let mut data = Vec::new();
        let encoder = PngEncoder::new(&mut data);
        encoder.write_image(&rgba, rgba.width(), rgba.height(), ColorType::Rgba8.into())?;
        Ok(data)
    }

    fn decode_png_to_mat_color(bytes: &[u8]) -> Result<core::Mat, Box<dyn Error>> {
        let buf = core::Vector::<u8>::from_slice(bytes);
        let mat = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR)?;
        Ok(mat)
    }

    fn clamp_rect(mut r: Rect, cols: i32, rows: i32) -> Rect {
        let x = r.x.max(0).min(cols - 1);
        let y = r.y.max(0).min(rows - 1);
        let w = (r.width.max(1)).min(cols - x);
        let h = (r.height.max(1)).min(rows - y);
        Rect::new(x, y, w, h)
    }

    pub fn find_image_on_screen(
        &mut self,
        assert_path: &str,
        precision: f32,
        region: Option<FindImageRegion>,
        template_width: Option<u32>,
        result_filter: Option<FindImageResultFilter>,
    ) -> Result<FindImageResults, Box<dyn Error>> {
        // Load template directly as Mat and resize via OpenCV if needed
        let Some(file) = Assets::get(assert_path) else {
            return Err(format!("assets加载{}失败", assert_path).into());
        };
        let mut template_color = Self::decode_png_to_mat_color(&file.data)?;
        if let Some(tw) = template_width {
            let cols = template_color.cols();
            let rows = template_color.rows();
            let ratio = tw as f64 / cols as f64;
            let new_h = (rows as f64 * ratio).round() as i32;
            let mut resized = core::Mat::default();
            imgproc::resize(&template_color, &mut resized, core::Size::new(tw as i32, new_h), 0.0, 0.0, imgproc::INTER_AREA)?;
            template_color = resized;
        }

        // Load screenshot from embedded assets
        let Some(screen_file) = Assets::get("screen.png") else {
            return Err("assets加载screen.png失败".into());
        };
        let mut screen_color = Self::decode_png_to_mat_color(&screen_file.data)?;

        // Apply region crop if provided
        if let Some((left, top, width, height)) = Self::_to_auto_gui_region(region) {
            let rect = Rect::new(left as i32, top as i32, width as i32, height as i32);
            let roi = core::Mat::roi(&screen_color, rect)?;
            screen_color = roi.try_clone()?;
        }

        // Convert both images to grayscale and enhance contrast/noise robustness
        let mut screen_gray = core::Mat::default();
        let mut template_gray = core::Mat::default();
        imgproc::cvt_color(&screen_color, &mut screen_gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        imgproc::cvt_color(&template_color, &mut template_gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        // Histogram equalization to reduce illumination variance
        let mut screen_eq = core::Mat::default();
        let mut template_eq = core::Mat::default();
        imgproc::equalize_hist(&screen_gray, &mut screen_eq)?;
        imgproc::equalize_hist(&template_gray, &mut template_eq)?;
        // Light Gaussian blur to denoise while preserving structure
        let mut screen_proc = core::Mat::default();
        let mut template_proc = core::Mat::default();
        imgproc::gaussian_blur(&screen_eq, &mut screen_proc, core::Size::new(3, 3), 0.0, 0.0, core::BORDER_DEFAULT, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        imgproc::gaussian_blur(&template_eq, &mut template_proc, core::Size::new(3, 3), 0.0, 0.0, core::BORDER_DEFAULT, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        if self.debug {
            println!(
                "screen: {}x{}, template: {}x{}, precision: {}",
                screen_proc.cols(),
                screen_proc.rows(),
                template_proc.cols(),
                template_proc.rows(),
                precision
            );
        }

        // Validate dimensions
        if template_proc.cols() > screen_proc.cols() || template_proc.rows() > screen_proc.rows() {
            return Ok(FindImageResults { width: template_proc.cols() as u32, height: template_proc.rows() as u32, list: vec![] });
        }

        // Run template matching on processed images
        let mut result = core::Mat::default();
        imgproc::match_template(
            &screen_proc,
            &template_proc,
            &mut result,
            imgproc::TM_CCOEFF_NORMED,
            &core::Mat::default(),
        )?;
        if self.debug {
            // Save the correlation map visualization
            let mut result_vis = core::Mat::default();
            core::normalize(&result, &mut result_vis, 0.0, 255.0, core::NORM_MINMAX, core::CV_8U, &core::Mat::default())?;
            let _ = imgcodecs::imwrite("target/result.png", &result_vis, &core::Vector::<i32>::new());
        }

        // Collect matches using iterative max suppression
        let mut list: Vec<FindImageResult> = vec![];
        let result_filter = result_filter.unwrap_or_else(|| FindImageResultFilter::new(5, 5));

        loop {
            let mut min_val: f64 = 0.0;
            let mut max_val: f64 = 0.0;
            let mut min_loc = Point::new(0, 0);
            let mut max_loc = Point::new(0, 0);

            min_max_loc(
                &result,
                Some(&mut min_val),
                Some(&mut max_val),
                Some(&mut min_loc),
                Some(&mut max_loc),
                &core::Mat::default(),
            )?;

            if max_val < precision as f64 {
                if self.debug {
                    println!("max: {:.4} < threshold {}", max_val, precision);
                }
                 break;
            }

            let left = max_loc.x.max(0) as u32;
            let top = max_loc.y.max(0) as u32;
            let candidate = (left, top, max_val as f32);

            let need_filter = list
                .iter()
                .any(|x| result_filter.need_filter(x, candidate));

            if !need_filter {
                list.push(FindImageResult {
                    left,
                    top,
                    precision: max_val as f32,
                });
                if self.debug {
                    println!("hit: ({}, {}), precision: {:.4}", left, top, max_val);
                    // Annotate top match area
                    let mut annotated = screen_color.try_clone()?;
                    let rect = Rect::new(max_loc.x, max_loc.y, template_proc.cols(), template_proc.rows());
                    imgproc::rectangle(&mut annotated, rect, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, 0)?;
                    let _ = imgcodecs::imwrite("target/annotated.png", &annotated, &core::Vector::<i32>::new());
                }
            }

            // Suppress region around current max to find next
            let suppress_rect = Rect::new(
                (max_loc.x - result_filter.x_delta as i32).max(0),
                (max_loc.y - result_filter.y_delta as i32).max(0),
                (template_proc.cols() as u32 + result_filter.x_delta * 2) as i32,
                (template_proc.rows() as u32 + result_filter.y_delta * 2) as i32,
            );
            let suppress_rect = Self::clamp_rect(suppress_rect, result.cols(), result.rows());
            imgproc::rectangle(
                &mut result,
                suppress_rect,
                Scalar::all(0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }

        Ok(FindImageResults {
            width: template_proc.cols() as u32,
            height: template_proc.rows() as u32,
            list,
        })
    }
}

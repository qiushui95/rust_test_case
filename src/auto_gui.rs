use crate::assets::Assets;
use image::{codecs::png::PngEncoder, ColorType, DynamicImage, GenericImageView};
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
        encoder.encode(&rgba, rgba.width(), rgba.height(), ColorType::Rgba8)?;
        Ok(data)
    }

    fn decode_png_to_mat_color(bytes: &[u8]) -> Result<core::Mat, Box<dyn Error>> {
        let bytes_mat = core::Mat::from_slice(bytes)?;
        let mat = imgcodecs::imdecode(&bytes_mat, imgcodecs::IMREAD_COLOR)?;
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
        // Load and optionally resize template from embedded assets
        let Some(file) = Assets::get(assert_path) else {
            return Err(format!("assets加载{}失败", assert_path).into());
        };
        let template_img = Self::resize_image(image::load_from_memory(&file.data)?, template_width);
        let template_png = Self::dynamic_image_to_png_bytes(&template_img)?;
        let template_color = Self::decode_png_to_mat_color(&template_png)?;

        // Load screenshot from embedded assets
        let Some(screen_file) = Assets::get("screen.png") else {
            return Err("assets加载screen.png失败".into());
        };
        let mut screen_color = Self::decode_png_to_mat_color(&screen_file.data)?;

        // Apply region crop if provided
        if let Some((left, top, width, height)) = Self::_to_auto_gui_region(region) {
            let rect = Rect::new(left as i32, top as i32, width as i32, height as i32);
            screen_color = core::Mat::roi(&screen_color, rect)?;
        }

        // Convert both images to grayscale for matching
        let mut screen_gray = core::Mat::default();
        let mut template_gray = core::Mat::default();
        imgproc::cvtColor(&screen_color, &mut screen_gray, imgproc::COLOR_BGR2GRAY, 0)?;
        imgproc::cvtColor(&template_color, &mut template_gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Validate dimensions
        if template_gray.cols() > screen_gray.cols() || template_gray.rows() > screen_gray.rows() {
            return Ok(FindImageResults { width: template_gray.cols() as u32, height: template_gray.rows() as u32, list: vec![] });
        }

        // Run template matching
        let mut result = core::Mat::default();
        imgproc::match_template(
            &screen_gray,
            &template_gray,
            &mut result,
            imgproc::TM_CCOEFF_NORMED,
            &core::Mat::default(),
        )?;

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
            }

            // Suppress region around current max to find next
            let suppress_rect = Rect::new(
                (max_loc.x - result_filter.x_delta as i32).max(0),
                (max_loc.y - result_filter.y_delta as i32).max(0),
                (template_gray.cols() as u32 + result_filter.x_delta * 2) as i32,
                (template_gray.rows() as u32 + result_filter.y_delta * 2) as i32,
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
            width: template_gray.cols() as u32,
            height: template_gray.rows() as u32,
            list,
        })
    }
}

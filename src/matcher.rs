use image::{DynamicImage, GenericImageView};
use opencv::{core, imgproc, prelude::*};
use std::error::Error;
use std::ops::Not;

pub struct ImageMatcher {
    template: Mat,
    use_gray: bool,
    width: u32,
    height: u32,
}

pub struct ImageMatchRegion {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

impl ImageMatchRegion {
    pub fn region(left: u32, top: u32, width: u32, height: u32) -> Self {
        assert!(width > 0, "width must > 0");

        assert!(height > 0, "height must > 0");

        Self {
            left,
            top,
            width,
            height,
        }
    }

    // pub fn from_window_rect(rect: &WindowRect) -> Self {
    //     let left = rect.left;
    //
    //     assert!(left >= 0, "left must >= 0");
    //
    //     let top = rect.top;
    //
    //     assert!(top >= 0, "top must >= 0");
    //
    //     Self::region(
    //         left as u32,
    //         top as u32,
    //         rect.width as u32,
    //         rect.height as u32,
    //     )
    // }
}

pub struct ImageMatchResult {
    pub left: u32,
    pub top: u32,
    pub precision: f32,
}

pub struct ImageMatchResults {
    pub width: u32,
    pub height: u32,
    pub list: Vec<ImageMatchResult>,
}

impl ImageMatchResults {
    fn new(width: u32, height: u32, list: Vec<ImageMatchResult>) -> Self {
        Self {
            width,
            height,
            list,
        }
    }
    pub fn first(&self) -> Option<&ImageMatchResult> {
        self.list.first()
    }
}

pub struct ImageMatchFilter {
    x_delta: u32,
    y_delta: u32,
}

impl ImageMatchFilter {
    pub fn new(x_delta: u32, y_delta: u32) -> Self {
        Self { x_delta, y_delta }
    }
    fn need_filter(&self, x: u32, y: u32, result: &ImageMatchResult) -> bool {
        if x < result.left - self.x_delta {
            return false;
        }

        if x > result.left + self.x_delta {
            return false;
        }

        if y < result.top - self.y_delta {
            return false;
        }

        if y > result.top + self.y_delta {
            return false;
        }

        true
    }
}

impl ImageMatcher {
    fn get_mat_from_dyn_image(image: DynamicImage) -> Result<Mat, Box<dyn Error>> {
        let rgb_image = image.to_rgb8();
        let data = rgb_image.as_raw();

        let mat = Mat::from_slice(data)?
            .reshape(3, image.height() as i32)?
            .try_clone()?;

        Ok(mat)
    }

    fn resize_mat(mat: Mat, width: Option<u32>) -> Result<Mat, Box<dyn Error>> {
        let Some(width) = width else {
            return Ok(mat);
        };

        let mut resized_mat = Mat::default();

        imgproc::resize(
            &mat,
            &mut resized_mat,
            core::Size::new(width as i32, 0i32),
            0.0,
            0.0,
            imgproc::INTER_LANCZOS4,
        )?;

        Ok(resized_mat)
    }

    fn gray_mat(mat: Mat, use_gray: bool) -> Result<Mat, Box<dyn Error>> {
        if use_gray.not() {
            return Ok(mat);
        }

        let mut gray_mat = Mat::default();

        imgproc::cvt_color(
            &mat,
            &mut gray_mat,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        Ok(gray_mat)
    }

    pub fn new(
        template: DynamicImage,
        use_gray: bool,
        width: Option<u32>,
    ) -> Result<Self, Box<dyn Error>> {
        let (mut w, mut h) = *&template.dimensions();

        if let Some(width) = width {
            w = width;
            h = w * h / w;
        }

        let mat = Self::get_mat_from_dyn_image(template)?;
        let mat = Self::resize_mat(mat, width)?;
        let mat = Self::gray_mat(mat, use_gray)?;

        let matcher = Self {
            template: mat,
            use_gray,
            width: w,
            height: h,
        };

        Ok(matcher)
    }
}

impl ImageMatcher {
    fn crop_mat(mat: Mat, region: Option<ImageMatchRegion>) -> Result<Mat, Box<dyn Error>> {
        let Some(region) = region else {
            return Ok(mat);
        };

        let rect = core::Rect {
            x: region.left as i32,
            y: region.top as i32,
            width: region.width as i32,
            height: region.height as i32,
        };

        let cropped_mat = mat.roi(rect)?.try_clone()?;

        Ok(cropped_mat)
    }
    fn need_filter(
        x: u32,
        y: u32,
        list: &Vec<ImageMatchResult>,
        filter: &ImageMatchFilter,
    ) -> bool {
        for result in list {
            if filter.need_filter(x, y, result) {
                return true;
            }
        }

        false
    }

    pub fn start_matching(
        &self,
        target_image: DynamicImage,
        precision: f32,
        region: Option<ImageMatchRegion>,
        filter: Option<ImageMatchFilter>,
    ) -> Result<ImageMatchResults, Box<dyn Error>> {
        let (mut width, mut height) = target_image.dimensions();

        if let Some(region) = &region {
            width = region.width;
            height = region.height;
        }

        let target_mat = Self::get_mat_from_dyn_image(target_image)?;
        let target_mat = Self::crop_mat(target_mat, region)?;
        let target_mat = Self::gray_mat(target_mat, self.use_gray)?;

        let mut result_mat = Mat::default();

        imgproc::match_template(
            &target_mat,
            &self.template,
            &mut result_mat,
            imgproc::TM_CCOEFF_NORMED,
            &core::no_array(),
        )?;

        let mut results = Vec::new();

        // 遍历结果矩阵，找到所有超过阈值的匹配点
        let result_size = result_mat.size().unwrap_or_default();

        let filter = filter.unwrap_or_else(|| ImageMatchFilter::new(5, 5));

        for y in 0..result_size.height as u32 {
            for x in 0..result_size.width as u32 {
                if Self::need_filter(x, y, &results, &filter) {
                    continue;
                }

                let Ok(threshold) = result_mat.at_2d::<f32>(y as i32, x as i32) else {
                    continue;
                };

                let threshold = *threshold;

                if threshold < precision {
                    continue;
                }

                results.push(ImageMatchResult {
                    left: x,
                    top: y,
                    precision: threshold,
                });
            }
        }

        // 按精度降序排序
        results.sort_by(|a, b| {
            b.precision
                .partial_cmp(&a.precision)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(ImageMatchResults::new(width, height, results))
    }
}
